import socket
import cv2
import time
import threading
import numpy as np
from flask import Flask, Response, abort, jsonify, render_template
from flask_cors import CORS
from ultralytics import YOLO
from tkinter import Tk, Toplevel, Label, Button, filedialog, messagebox, simpledialog
from queue import Queue
from options import choose_input, choose_mode
from speed import SpeedEstimator
from flask_socketio import SocketIO
from utils import is_crossing_line, is_inside_roi, seconds_to_hms

# # Check if CUDA is available
# if torch.cuda.is_available():
#     print("CUDA is available. PyTorch is using the GPU.")
# else:
#     print("CUDA is not available. PyTorch is using the CPU.")

# Load YOLO model
model = YOLO("yolo11s.pt")# Load your YOLO model
names = model.model.names  # Class names from the YOLO model

# Flask app for streaming and API
app = Flask(__name__)
CORS(app)

socketio = SocketIO(app, cors_allowed_origins="*")
socketio.init_app(app)

# print(cv2.getBuildInformation())

# Predefined RTSP streams
RTSP_STREAMS = {
    # "Camera_1": "rtsp://admin:admin@192.168.2.71:554/unicaststream/1",
    "Camera_2": {
        "code_name" : "MRKCCTVMERAK1",
        "url": "https://mitradarat-vidstream.kemenhub.go.id/stream/MRKCCTVMERAK1/stream.m3u8",
        "roi": [(130, 120), (710, 105), (980, 200), (5, 200)],
    },
    "Camera_3": {
        "code_name" : "MRKCCTVMERAK2",
        "url": "https://mitradarat-vidstream.kemenhub.go.id/stream/MRKCCTVMERAK2/stream.m3u8",
        "roi": [(315, 150), (400, 150), (400, 400), (100, 400)],
    },
    "Camera_4": {
        "code_name" : "Ambon",
        "url" : "https://pantau.margamandala.co.id:3443/km97/ambon/ambon.m3u8"
    },
    "Camera_5": {
        "code_name" : "Merak Entrance",
        "url" : "https://pantau.margamandala.co.id:3443/merak/entrance/entrance.m3u8",    
    },
    "Camera_6": {
        "code_name" : "SP. Gadog",
        "url": "https://atcs-bptj.dephub.go.id/camera/gadog.m3u8"
    },
}

# Global variables
cap = None
current_stream = None
object_timers = {}
emit_queue = Queue(maxsize=50)
processed_frames_queues = {stream_name: Queue(maxsize=10) for stream_name in RTSP_STREAMS}
crossed_ids = set()
objects_in_roi_count = 0
frame_lock = threading.Lock()
latest_frame = None
latest_frames = {}
line_pts = [(0, 288), (1019, 288)]  # Static line definition for speed mode
speed_obj = SpeedEstimator(reg_pts=line_pts, names=names)
vehicle_counts = {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0}
current_roi = None  # ROI for dwelling time mode
input_source = None  # "File" or "RTSP/Stream"
DEFAULT_MODE = 'speed_and_track'

def get_ip_address():
    print("Getting IP address...")
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        return local_ip
    except Exception as e: 
        print(f"Error getting IP address: {e}")
        return "127.0.0.1"

# Start Flask app in a separate thread
def start_flask():
    print("Starting Flask server...")
    ipaddress = get_ip_address()
    print(f"Server running at http://{ipaddress}:5000")
    
    for camera_name in RTSP_STREAMS:
        video_feed_url = f"http://{ipaddress}:5000/video_feed/{camera_name}"
        print(f"Video Feed Url for {camera_name}: {video_feed_url}")
    
    try:
        socketio.run(app, host="0.0.0.0", port=5000, debug=False, log_output=True)
    except Exception as e:
        print(f"Error starting Flask server: {e}")

def set_mode(mode):
    global current_mode
    current_mode = mode  # Update the mode globally
    print(f"Mode set to: {mode}")

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

def process_dwelling_time_mode(frame, results):
    # Process for dwelling time mode
    global objects_in_roi_count, object_timers, crossed_ids
    
    if current_roi is None:
        print("Current ROI is not set.")
        return
    
    current_ids_in_roi = set()
    
    for result in results:
        if result.boxes:
            for box in result.boxes:
                bbox = box.xyxy[0].cpu().numpy()
                obj_center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
                obj_class = int(box.cls)
                obj_name = names[obj_class]
                obj_id = int(box.id) if box.id is not None else -1

                if is_inside_roi(obj_center, current_roi):
                    current_ids_in_roi.add(obj_id)
                    if obj_id not in object_timers:
                        object_timers[obj_id] = time.time()
                    if obj_id not in crossed_ids:
                        crossed_ids.add(obj_id)
                        objects_in_roi_count += 1

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
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                else:
                    if obj_id in object_timers:
                        del object_timers[obj_id]

    # Update the count of objects in ROI
    for obj_id in list(crossed_ids):
        if obj_id not in current_ids_in_roi:
            crossed_ids.remove(obj_id)
            objects_in_roi_count -= 1

    # Ensure the count does not go below zero
    objects_in_roi_count = max(objects_in_roi_count, 0)

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
    
def process_speed_and_track_mode(frame, results):
    # Process for speed and tracking mode
    global vehicle_counts, crossed_ids
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

    # Draw the lane line and vehicle count text
    cv2.line(frame, line_pts[0], line_pts[1], (0, 255, 0), 2)
    y_offset = 30
    for obj, count in vehicle_counts.items():
        text = f"{obj}: {count}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 20

    # Estimate speed if applicable
    try:
        frame = speed_obj.estimate_speed(frame, results)
    except Exception as e:
        print(f"Error in estimate_speed: {e}")

    return frame

# Function to process video frames
def process_video(stream_name, stream_info, mode, cap):
# def process_video():
    # global cap, latest_frame, objects_in_roi_count, vehicle_counts
    
    global latest_frame, latest_frames, objects_in_roi_count, vehicle_counts, object_timers, crossed_ids, current_roi

    if isinstance(stream_info, dict) and "roi" in stream_info:
        current_roi = stream_info["roi"]
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1.0 / fps if fps > 0 else 1.0 / 30  # Minimum frame time of 30ms
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {stream_name}. Reconnecting...")
            time.sleep(2)
            continue

        frame = cv2.resize(frame, (1020, 500))
        results = model.track(frame, persist=True)

        if mode == "dwelling_time":
            process_dwelling_time_mode(frame, results)
        elif mode == "speed_and_track":
            process_speed_and_track_mode(frame, results)

        # wit:
        latest_frame = frame.copy()
        latest_frames[stream_name] = frame.copy()  # Store the frame in latest_frames
        print(f"Updated latest frame for {stream_name}")  # Debugging log
            
        # Emit the frame via WebSocket
        if mode == "dwelling_time":
            print("Emitting object count update...")
            emit_queue.put(("object_count_update", {"object_in_roi_count": objects_in_roi_count}))
        elif mode == "speed_and_track":
            print("Emitting vehicle count update...")
            emit_queue.put(("vehicle_count_update", {"vehicle_counts": vehicle_counts}))
            
         # Add the processed frame to the queue
        if not processed_frames_queues[stream_name].full():
            processed_frames_queues[stream_name].put(frame.copy())
        
        # Ensure consistent frame rate
        elapsed_time = time.time() - start_time
        sleep_time = max(0, frame_time - elapsed_time)
        time.sleep(sleep_time)

def stream_all_rtsp():
    """
    Function to start processing multiple RTSP streams concurrently.
    """
    threads = []

    # Initialize VideoCapture for each RTSP stream and create threads
    for stream_name, stream_info in RTSP_STREAMS.items():
        if isinstance(stream_info, dict):  # Check if stream_info is a dictionary
            mode = 'dwelling_time' if stream_info.get("roi", None) else DEFAULT_MODE
            stream_url = stream_info["url"]
        else:
            mode = DEFAULT_MODE  # Default mode for string URLs
            stream_url = stream_info
            
        print(f"Initializing stream: {stream_name} with URL: {stream_url}")  # Log the URL
        
        # Initialize VideoCapture for each camera stream
        # cap = cv2.VideoCapture(stream_info["url"] if isinstance(stream_info, dict) else stream_info)
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"Failed to open stream: {stream_name} : {stream_url}")
            continue
        
        # print(f"Initializing stream: {stream_name} with URL: {stream_info['url'] if isinstance(stream_info, dict) else stream_info}")
        # Start a new thread for each RTSP stream to process video
        thread = threading.Thread(target=process_video, args=(stream_name, stream_info, mode, cap))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish (this will run indefinitely)
    for thread in threads:
        thread.join()

def emit_updates():
    while True:
        event, data = emit_queue.get()
        socketio.emit(event, data)
print("Initializing emit_updates thread...")
threading.Thread(target=emit_updates, daemon=True).start()
print("emit_updates thread started.")

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

# Example of setting the mode dynamically via a WebSocket message
@socketio.on("set_mode")
def on_set_mode(data):
    new_mode = data.get("mode", "speed_and_track")
    set_mode(new_mode)
    
@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")    

# Flask route for video streaming
# @app.route("/video_feed")
# def video_feed():
#     def generate():
#         global latest_frame
#         while True:
#             wit:
#                 if latest_frame is None:
#                     continue
#                 _, buffer = cv2.imencode(".jpg", latest_frame)
#             yield (b"--frame\r\n"
#                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
#     return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed/<camera_name>")
def video_feed(camera_name):
    if camera_name not in RTSP_STREAMS:
        return "Camera stream not found!", 404

    def generate():
        # global latest_frames
        while True:
            # with frame_lock:
                if not processed_frames_queues[camera_name].empty():
                    frame = processed_frames_queues[camera_name].get()
                    _, buffer = cv2.imencode(".jpg", frame)
                    yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
                else:
                    time.sleep(0.01)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/select_camera")
def select_camera():
    try:
        cctv_streams = [{"name": name} for name in RTSP_STREAMS.keys()]
        return render_template("select_camera.html", cctv_streams=cctv_streams)
    except Exception as e:
        print(f"Error rendering select_camera: {e}")
        abort(500)

@app.route("/streams")
def get_streams():
    streams = [{"name": name, "url": info["url"] if isinstance(info, dict) else info} for name, info in RTSP_STREAMS.items()]
    return jsonify(streams)

@app.route("/object_count", methods=["GET"])
def object_count():
    data = {"object_in_roi_count": objects_in_roi_count}    
    return jsonify(data)
  
@app.route("/vehicle_count", methods=["GET"])
def vehicle_count():
    data = {"vehicle_counts": vehicle_counts}    
    return jsonify(data)

@app.route("/health")
def health_check():
    return "Flask server is running", 200

print("Starting main execution block...")
# Main execution
input_source = choose_input()

print(get_ip_address())

print("Starting Flask server thread...")
flask_thread = threading.Thread(target=start_flask, daemon=True)
flask_thread.start()
print("Flask server thread has been started.")

if input_source == "File": 
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mkv"), ("All Files", "*.*")]
    )
    
    if not video_path:
        print("No video file selected. Exiting...")
        exit()
        
    cap = cv2.VideoCapture(video_path)
    mode = DEFAULT_MODE
    threading.Thread(target=process_video, args=("File", {"url": video_path}, mode, cap), daemon=True).start()
    
elif input_source == "RTSP/Stream":
        # select_rtsp()
    stream_all_rtsp()

# mode = choose_mode()


# Keep the main thread alive
try:
    while True:
        time.sleep(1)  # Sleep to reduce CPU usage
except KeyboardInterrupt:
    print("Exiting program...")
    cv2.destroyAllWindows()