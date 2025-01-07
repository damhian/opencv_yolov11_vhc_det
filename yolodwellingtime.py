import cv2
import time
from ultralytics import YOLO
import numpy as np
from tkinter import Tk, Toplevel, Label, Button, messagebox
from flask import Flask, Response, jsonify
from flask_cors import CORS
import threading

# Load YOLOv11 model
model = YOLO("yolo11n.pt")  # Load your YOLOv11 model
names = model.model.names  # Class names from the YOLO model

# Flask app for streaming and API
app = Flask(__name__)
CORS(app)

# Global variables
cap = None
current_stream = None
object_timers = {}
crossed_ids = set()
objects_in_roi_count = 0
frame_lock = threading.Lock()
latest_frame = None

# Predefined RTSP streams
RTSP_STREAMS = {
    "Camera 1 (MRKCCTVMERAK1)": {
        "url": "https://mitradarat-vidstream.kemenhub.go.id/stream/MRKCCTVMERAK1/stream.m3u8",
        "roi": [(130, 120), (710, 105), (980, 200), (5, 200)],
    },
    "Camera 2 (MRKCCTVMERAK2)": {
        "url": "https://mitradarat-vidstream.kemenhub.go.id/stream/MRKCCTVMERAK2/stream.m3u8",
        "roi": [(315, 150), (400, 150), (400, 400), (100, 400)],
    },
}

current_roi = None  # Current ROI for the selected camera


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
        cap = cv2.VideoCapture(current_stream["url"], cv2.CAP_FFMPEG)
        current_roi = current_stream["roi"]
        rtsp_window.destroy()

    Label(rtsp_window, text="Select a Camera Stream:", font=("Arial", 14)).pack(pady=10)
    for name in RTSP_STREAMS.keys():
        Button(rtsp_window, text=name, font=("Arial", 12), command=lambda name=name: set_stream(name)).pack(pady=5)

    root.wait_window(rtsp_window)
    root.destroy()

    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to open RTSP stream. Exiting...")
        exit()


# Function to check if a point is inside the ROI
def is_inside_roi(point, roi):
    return cv2.pointPolygonTest(np.array(roi, dtype=np.int32), point, False) >= 0


# Function to start the processing loop
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

        with frame_lock:
            latest_frame = frame.copy()


# Function to stream video via Flask
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


# API endpoint to get object count
@app.route("/object_count", methods=["GET"])
def object_count():
    return jsonify({"objects_in_roi_count": objects_in_roi_count})


# Convert seconds to hh:mm:ss format
def seconds_to_hms(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


# Start Flask app in a separate thread
def start_flask():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


# Main execution
select_rtsp()
threading.Thread(target=process_video, daemon=True).start()
threading.Thread(target=start_flask, daemon=True).start()

# Keep the main thread alive
try:
    while True:
        time.sleep(1)  # Sleep to reduce CPU usage
except KeyboardInterrupt:
    print("Exiting program...")
    cv2.destroyAllWindows()
