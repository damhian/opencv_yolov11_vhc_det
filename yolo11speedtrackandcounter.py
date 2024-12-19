import cv2
from ultralytics import YOLO
from speed import SpeedEstimator
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from ultralytics.engine.results import Boxes

# Load YOLOv11 model
model = YOLO("yolo11s.pt")

# Initialize global variable to store cursor coordinates
line_pts = [(0, 288), (1019, 288)]  # Static line definition
names = model.model.names  # This is a dictionary

# Object counter per lane
object_counts = {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0}

# Set of IDs for vehicles that have already crossed the line
crossed_ids = set()

speed_obj = SpeedEstimator(reg_pts=line_pts, names=names)

# Predefined RTSP streams
RTSP_STREAMS = {
    "Camera 1": "rtsp://admin:admin@192.168.2.71:554/unicaststream/1",
    "(..Coming Soon..)": "(..Coming Soon..)",  # Example
}

# Mouse callback function to capture mouse movement
def RGB(event, x, y, flags, param):
    global cursor_point
    if event == cv2.EVENT_MOUSEMOVE:
        cursor_point = (x, y)
        print(f"Mouse coordinates: {cursor_point}")

# Initialize Tkinter and hide the main window
root = tk.Tk()
root.withdraw()

# Ask user to choose the input source
choice = simpledialog.askstring(
    "Input Source",
    "Choose input source:\n1 - File\n2 - RTSP/Stream",
    parent=root,
)

if choice == "1":
    # Open a file dialog to select a video file
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mkv"), ("All Files", "*.*")]
    )

    # Check if a file was selected
    if not video_path:
        print("No video file selected. Exiting...")
        exit()

    # Open the video file or webcam feed
    cap = cv2.VideoCapture(video_path)
    
elif choice == "2": 
    # Create a new window for RTSP stream selection
    rtsp_window = tk.Toplevel(root)
    rtsp_window.title("Select RTSP Stream")
    rtsp_window.geometry("300x200")

    # Function to handle RTSP selection
    def select_rtsp(stream_url):
        global cap
        cap = cv2.VideoCapture(stream_url)
        rtsp_window.destroy()

    tk.Label(rtsp_window, text="Select a Camera Stream:", font=("Arial", 14)).pack(pady=10)

    for name, url in RTSP_STREAMS.items():
        tk.Button(rtsp_window, text=name, font=("Arial", 12), command=lambda url=url: select_rtsp(url)).pack(pady=5)

    # Wait for user to select a stream
    root.wait_window(rtsp_window)
    
    # Check if the stream URL was selected
    if not cap or not cap.isOpened():
        messagebox.showerror("Error", "Failed to open RTSP stream. Exiting...")
        exit()
    
else:
    print("Invalid choice. Exiting...")
    exit()
    
# Ask user to choose the line orientation
line_orientation = simpledialog.askstring(
    "Line Orientation",
    "Choose line orientation(push H or V on Keyboard):\nH - Horizontal\nV - Vertical",
    parent=root,
)

# Define static line coordinates based on the selected orientation
if line_orientation == "H" or line_orientation.lower() == "h":
    # Horizontal line
    line_pts = [(0, 288), (1019, 288)]  # Example coordinates for horizontal
    is_horizontal = True
elif line_orientation == "V" or line_orientation.lower() == "v":
    # Vertical line
    line_pts = [(510, 0), (510, 500)]  # Example coordinates for vertical
    is_horizontal = False
else:
    print("Invalid choice for line orientation. Exiting...")
    exit()
    
# Set up the window and attach the mouse callback function
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

count = 0

def is_crossing_line(obj_center, line_start, line_end, is_horizontal):
    """Check if the object's center crosses the line."""
    x1, y1 = line_start
    x2, y2 = line_end
    cx, cy = obj_center

    # # Line equation check
    # if y1 - 5 <= cy <= y1 + 5:
    #     return True
    # return False
    
    if is_horizontal:
        # Horizontal line: Check y-coordinate proximity
        return y1 - 5 <= cy <= y1 + 5
    else:
        # Vertical line: Check x-coordinate proximity
        return x1 - 5 <= cx <= x1 + 5

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video stream ended or cannot be read.")
        break

    count += 1
    # if count % 2 != 0:  # Skip some frames for speed (optional)
    #     continue

    frame = cv2.resize(frame, (1020, 500))
    
    # Perform object tracking
    results_list = model.track(frame, persist=True)  # Returns a list
    
    # Filter results and exclude "person" class (class index 0)
    filtered_results_list = []
    for result in results_list:
        if result.boxes:  # Ensure there are detections in this result
            # Filter out "person" class
            filtered_boxes_tensor = result.boxes.cls != 0  # Exclude class index 0
            filtered_boxes = result.boxes[filtered_boxes_tensor]
            # Get the original image shape (frame shape)
            orig_shape = frame.shape[:2]  # Get the (height, width) of the frame
            # Replace result.boxes with filtered Boxes object
            result.boxes = Boxes(filtered_boxes.data, orig_shape=orig_shape)
            filtered_results_list.append(result)
            
    for result in filtered_results_list:
        if result.boxes:
            for box in result.boxes:
                # Extract class index, confidence, and bounding box coordinates
                obj_class = int(box.cls)  # Class index
                obj_name = names[obj_class]  # Map index to class name
                # track_id = int(box.id)  # Get the tracking ID
                track_id = int(box.id) if box.id is not None else -1  # Use -1 for missing IDs

                   
                # Object counting logic
                bbox = box.xyxy[0].cpu().numpy()  # Convert tensor to NumPy array
                obj_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

                # Check if the object crosses the defined line
                # if is_crossing_line(obj_center, line_pts[0], line_pts[1]):
                if is_crossing_line(obj_center, line_pts[0], line_pts[1], is_horizontal):
                    # if obj_name in object_counts:
                    if track_id not in crossed_ids and obj_name in object_counts:
                        object_counts[obj_name] += 1
                        crossed_ids.add(track_id)  # Add the ID to the set of crossed objects

    # Draw the lane line
    # cv2.line(frame, line_pts[0], line_pts[1], (0, 255, 0), 2)
    # Draw the lane line based on the selected orientation
    
    if is_horizontal:
    # Horizontal line
        cv2.line(frame, (line_pts[0][0], line_pts[0][1]), (line_pts[1][0], line_pts[0][1]), (0, 255, 0), 2)
    else:
        # Vertical line
        cv2.line(frame, (line_pts[0][0], line_pts[0][1]), (line_pts[0][0], line_pts[1][1]), (0, 255, 0), 2)


    # Display counters on the frame
    y_offset = 30
    for obj, count in object_counts.items():
        text = f"{obj}: {count}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 20

    im0 = speed_obj.estimate_speed(frame, results_list)

    # Display the frame with YOLOv11 results and counters
    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()