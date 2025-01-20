import cv2
import numpy as np


def seconds_to_hms(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"
  
  # Function to check if a point is inside the ROI
def is_inside_roi(point, roi):
    return cv2.pointPolygonTest(np.array(roi, dtype=np.int32), point, False) >= 0
  
def is_crossing_line(obj_center, line_start, line_end):
    x1, y1 = line_start
    cx, cy = obj_center
    return y1 - 5 <= cy <= y1 + 5  # Horizontal line check