import cv2
import time
from ultralytics import YOLO
import numpy as np
from tkinter import Tk, Toplevel, Label, Button, simpledialog, messagebox, filedialog

# Load YOLOv11 model
model = YOLO("yolo11n.pt")  # Load your YOLOv11 model
names = model.model.names  # Class names from the YOLO model

# Initialize global variables
roi = [(315, 150), (400, 150), (400, 400), (100, 400)]  # Example ROI
object_timers = {}  # Dictionary to track dwelling time for objects
crossed_ids = set()  # Set of IDs for tracked objects
objects_in_roi_count = 0  # Counter for objects that have entered the ROI

# Predefined RTSP streams
RTSP_STREAMS = {
    "Camera 1": "rtsp://admin:admin@192.168.2.71:554/unicaststream/1",
    "Camera 2 (MRKCCTVMERAK1)": "https://mitradarat-vidstream.kemenhub.go.id/stream/MRKCCTVMERAK1/stream.m3u8",
    "Camera 3 (MRKCCTVMERAK2)": "https://mitradarat-vidstream.kemenhub.go.id/stream/MRKCCTVMERAK2/stream.m3u8",
    "Camera 4 (Ambon)": "https://pantau.margamandala.co.id:3443/km97/ambon/ambon.m3u8",
    "Camera 5 (Merak Entrance)": "https://pantau.margamandala.co.id:3443/merak/entrance/entrance.m3u8",
    "(..Coming Soon..)": "(..Coming Soon..)",  # Example
}

# Function to select RTSP stream
def select_rtsp():
    root = Tk()
    root.withdraw()

    rtsp_window = Toplevel(root)
    rtsp_window.title("Select RTSP Stream")
    rtsp_window.geometry("300x400")

    def set_stream(url):
        global cap
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        rtsp_window.destroy()

    Label(rtsp_window, text="Select a Camera Stream:", font=("Arial", 14)).pack(pady=10)
    for name, url in RTSP_STREAMS.items():
        Button(rtsp_window, text=name, font=("Arial", 12), command=lambda url=url: set_stream(url)).pack(pady=5)

    root.wait_window(rtsp_window)
    root.destroy()

    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to open RTSP stream. Exiting...")
        exit()

# Select video source (RTSP)
select_rtsp()

# Function to check if a point is inside the ROI
def is_inside_roi(point, roi):
    return cv2.pointPolygonTest(np.array(roi, dtype=np.int32), point, False) >= 0

# Function to start counting objects when they enter the ROI
def count_object_in_roi(obj_id):
    global objects_in_roi_count
    if obj_id not in crossed_ids:
        crossed_ids.add(obj_id)  # Add the object ID to the set
        objects_in_roi_count += 1  # Increment the counter

# Start timer for an object when it enters the ROI
def start_timer(obj_id):
    if obj_id not in object_timers:
        object_timers[obj_id] = time.time()

# Stop timer when the object leaves the ROI and calculate dwelling time
def stop_timer(obj_id):
    if obj_id in object_timers:
        dwelling_time = time.time() - object_timers[obj_id]
        del object_timers[obj_id]
        return dwelling_time
    return 0
  
# Function to convert seconds to hh:mm:ss format
def seconds_to_hms(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

# Main processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame. Reconnecting...")
        cap.release()
        time.sleep(2)
        cap = cv2.VideoCapture(select_rtsp, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print("Failed to reconnect.")
            break
        continue

    # Resize frame for faster processing
    frame = cv2.resize(frame, (1020, 500))

    # Perform object detection with YOLO
    results = model.track(frame, persist=True)

    for result in results:
        if result.boxes:
            for box in result.boxes:
                # Extract object details
                bbox = box.xyxy[0].cpu().numpy()  # Bounding box
                obj_center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
                obj_class = int(box.cls)  # Class index
                obj_name = names[obj_class]  # Class name
                obj_id = int(box.id) if box.id is not None else -1  # Object ID

                # Check if the object is inside the ROI
                if is_inside_roi(obj_center, roi):
                    start_timer(obj_id)  # Start the timer for the object
                    count_object_in_roi(obj_id)  # Increment the counter when the object enters the ROI

                    # Calculate dwelling time
                    dwelling_time = time.time() - object_timers[obj_id]
                    cv2.putText(
                        frame,
                        f"{obj_name} Timer: {dwelling_time:.1f}s",
                        (int(bbox[0]), int(bbox[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                else:
                    # Stop timer if the object leaves the ROI
                    if obj_id in object_timers:
                        time_spent = stop_timer(obj_id)
                        print(f"{obj_name} ID {obj_id} spent {time_spent:.2f} seconds in the ROI.")

    # Draw the ROI
    cv2.polylines(frame, [np.array(roi, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

    # Display the counter of objects in the ROI
    cv2.putText(
        frame,
        f"Jumlah: {objects_in_roi_count}",
        (20, 30),  # Position at the top-left corner
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    # Show the frame
    cv2.imshow("Video Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
