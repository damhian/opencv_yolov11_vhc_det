import ipaddress
import socket
import cv2
import time
from ultralytics import YOLO
import numpy as np
from tkinter import Tk, Toplevel, Label, Button, filedialog, messagebox, simpledialog
from flask import Flask, Response, jsonify
from flask_cors import CORS
import threading
from options import choose_input, choose_mode
from speed import SpeedEstimator
from flask_socketio import SocketIO
from queue import Queue

from utils import is_crossing_line, is_inside_roi, seconds_to_hms

# Load YOLO model
model = YOLO("yolo11s.pt")  # Load your YOLO model
names = model.model.names  # Class names from the YOLO model

# Flask app for streaming and API
app = Flask(__name__)
CORS(app)

socketio = SocketIO(app, cors_allowed_origins="*")
socketio.init_app(app)

# Global variables
cap = None
current_stream = None
object_timers = {}
emit_queue = Queue(maxsize=50)
crossed_ids = set()
objects_in_roi_count = 0
frame_lock = threading.Lock()
latest_frame = None
line_pts = [(0, 288), (1019, 288)]  # Static line definition for speed mode
speed_obj = SpeedEstimator(reg_pts=line_pts, names=names)
vehicle_counts = {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0}
current_roi = None  # ROI for dwelling time mode
input_source = None  # "File" or "RTSP/Stream"

# Predefined RTSP streams
RTSP_STREAMS = {
    "Camera 1": "rtsp://admin:admin@192.168.2.71:554/unicaststream/1",
    "Camera 2 (MRKCCTVMERAK1)": {
        "url": "https://mitradarat-vidstream.kemenhub.go.id/stream/MRKCCTVMERAK1/stream.m3u8",
        "roi": [(130, 120), (710, 105), (980, 200), (5, 200)],
    },
    "Camera 3 (MRKCCTVMERAK2)": {
        "url": "https://mitradarat-vidstream.kemenhub.go.id/stream/MRKCCTVMERAK2/stream.m3u8",
        "roi": [(315, 150), (400, 150), (400, 400), (100, 400)],
    },
    "Camera 4 (Ambon)": "https://pantau.margamandala.co.id:3443/km97/ambon/ambon.m3u8",
    "Camera 5 (Merak Entrance)": "https://pantau.margamandala.co.id:3443/merak/entrance/entrance.m3u8",
    "Camera 6 (SP. Gadog)": "https://atcs-bptj.dephub.go.id/camera/gadog.m3u8",
    "(..Coming Soon..)": "(..Coming Soon..)",  # Example
}

# Function to select RTSP stream
def select_rtsp():
    global cap, current_stream, current_roi
    root = Tk()
    root.withdraw()

    rtsp_window = Toplevel(root)
    rtsp_window.title("Select RTSP Stream")
    rtsp_window.geometry("300x400")

    def set_stream(stream_name):
      global cap, current_stream, current_roi
      current_stream = RTSP_STREAMS[stream_name]
      
      # Check if current_stream is a dictionary (it may contain 'url' and 'roi' keys)
      if isinstance(current_stream, dict):
          current_stream_url = current_stream["url"]
          current_roi = current_stream.get("roi", None)  # Use default None if 'roi' doesn't exist
      else:
          current_stream_url = current_stream
          current_roi = None  # No ROI for simple URL-based stream
      
      # Open the stream
      cap = cv2.VideoCapture(current_stream_url, cv2.CAP_FFMPEG)
      if not cap.isOpened():
          messagebox.showerror("Error", "Failed to open RTSP stream. Exiting...")
          exit()

      rtsp_window.destroy()

    Label(rtsp_window, text="Select a Camera Stream:", font=("Arial", 14)).pack(pady=10)
    for name in RTSP_STREAMS.keys():
        Button(rtsp_window, text=name, font=("Arial", 12), command=lambda name=name: set_stream(name)).pack(pady=5)

    root.wait_window(rtsp_window)
    root.destroy()

    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to open RTSP stream. Exiting...")
        exit()

# Function to process video frames
def process_video():
    global cap, latest_frame, objects_in_roi_count

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame. Reconnecting...")
            time.sleep(2)
            continue

        frame = cv2.resize(frame, (1020, 500))
        results = model.track(frame, persist=True)

        if mode == "Dwelling Time":
            # Dwelling Time Mode
            for result in results:
                if result.boxes:
                    for box in result.boxes:
                        bbox = box.xyxy[0].cpu().numpy()
                        obj_center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
                        obj_class = int(box.cls)
                        obj_name = names[obj_class]
                        obj_id = int(box.id) if box.id is not None else -1

                        if is_inside_roi(obj_center, current_roi):
                            if obj_id not in object_timers:
                                object_timers[obj_id] = time.time()
                            if obj_id not in crossed_ids:
                                crossed_ids.add(obj_id)
                                objects_in_roi_count += 1
                            else:
                                objects_in_roi_count -= 1

                            dwelling_time = time.time() - object_timers[obj_id]
                            dwelling_time_str = seconds_to_hms(dwelling_time)

                            cv2.putText(
                                frame,
                                f"{obj_name} Timer: {dwelling_time_str}",
                                (int(bbox[0]), int(bbox[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2,
                            )
                        else:
                            if obj_id in object_timers:
                                del object_timers[obj_id]

            # Draw ROI
            cv2.polylines(frame, [np.array(current_roi, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.putText(
                frame,
                f"Jumlah: {objects_in_roi_count}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        elif mode == "Speed and Track":
            # Speed and Track Mode
            for result in results:
                if result.boxes:
                    for box in result.boxes:
                        bbox = box.xyxy[0].cpu().numpy()
                        obj_center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
                        obj_class = int(box.cls)
                        obj_name = names[obj_class]
                        track_id = int(box.id) if box.id is not None else -1

                        # Check if object crosses the line
                        if is_crossing_line(obj_center, line_pts[0], line_pts[1]):
                            if track_id not in crossed_ids:
                                crossed_ids.add(track_id)

                                # Update vehicle counts based on the object class
                                if obj_name in vehicle_counts:
                                    vehicle_counts[obj_name] += 1
                                else:
                                    vehicle_counts[obj_name] = 1

            # Draw the lane line
            cv2.line(frame, line_pts[0], line_pts[1], (0, 255, 0), 2)
            
            # Display counters on the frame
            y_offset = 30
            for obj, count in vehicle_counts.items():
                text = f"{obj}: {count}"
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 20

            # Estimate speed using the SpeedEstimator
            frame = speed_obj.estimate_speed(frame, results)

        with frame_lock:
            latest_frame = frame.copy()
        
        # print(f"Objects in ROI: {objects_in_roi_count}, Vehicle Counts: {vehicle_counts}")
        
        # Emit the frame via WebSocket
        if mode == "Dwelling Time":
            print("Emitting object count update...")
            emit_queue.put(("object_count_update", {"object_in_roi_count": objects_in_roi_count}))
        elif mode == "Speed and Track":
            print("Emitting vehicle count update...")
            emit_queue.put(("vehicle_count_update", {"vehicle_counts": vehicle_counts}))

@socketio.on("connect")
def handle_connect():
    print("Client connected")
    socketio.emit("stream_url", {"stream_url": current_stream})

@socketio.on("handle_client")
def handle_client(data):
    print("Client sent data:", data)
    
    # Assuming data contains specific event types like "object_count_update" and "vehicle_count_update"
    if "object_in_roi_count" in data:
        objects_in_roi_count = data["object_in_roi_count"]
        # Put the event in the emit_queue for object count updates
        emit_queue.put(("object_count_update", {"object_in_roi_count": objects_in_roi_count}))
        print(f"Object Count Update: {objects_in_roi_count}")

    if "vehicle_counts" in data:
        vehicle_counts = data["vehicle_counts"]
        # Put the event in the emit_queue for vehicle count updates
        emit_queue.put(("vehicle_count_update", {"vehicle_counts": vehicle_counts}))
        print(f"Vehicle Count Update: {vehicle_counts}")

    # You can also emit an acknowledgment or confirmation back to the client if needed
    socketio.emit("client_response", {"message": "Data received and processed"})
    
@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")    
        
def emit_updates():
    while True:
        event, data = emit_queue.get()
        socketio.emit(event, data)

threading.Thread(target=emit_updates, daemon=True).start()

# Flask route for video streaming
@app.route("/video_feed")
def video_feed():
    def generate():
        global latest_frame
        while True:
            with frame_lock:
                if latest_frame is None:
                    continue
                _, buffer = cv2.imencode(".jpg", latest_frame)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

# Flask API endpoint for object count
@app.route("/object_count", methods=["GET"])
def object_count():
    data = {"object_in_roi_count": objects_in_roi_count}    
    return jsonify(data)
  
@app.route("/vehicle_count", methods=["GET"])
def vehicle_count():
    data = {"vehicle_counts": vehicle_counts}    
    return jsonify(data)

def get_ip_address():
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        return local_ip
    except Exception as e: 
        return "127.0.0.1"

# Start Flask app in a separate thread
def start_flask():
    ipaddress = get_ip_address()
    video_feed_url = f"http://{ipaddress}:5000/video_feed"
    print(f"Video Feed Url: {video_feed_url}")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, log_output=True)

# Main execution
input_source = choose_input()

if input_source == "File": 
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mkv"), ("All Files", "*.*")]
    )
    
    if not video_path:
        print("No video file selected. Exiting...")
        exit()
        
    cap = cv2.VideoCapture(video_path)
    
elif input_source == "RTSP/Stream":
    select_rtsp()

mode = choose_mode()

threading.Thread(target=process_video, daemon=True).start()
threading.Thread(target=start_flask, daemon=True).start()

# Keep the main thread alive
try:
    while True:
        time.sleep(1)  # Sleep to reduce CPU usage
except KeyboardInterrupt:
    print("Exiting program...")
    cv2.destroyAllWindows()