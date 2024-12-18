import cv2
import json
import os
import shutil
from pathlib import Path

def capture_frame(camera_id):
    cap = cv2.VideoCapture(camera_id)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def select_rois(frame):
    rois = cv2.selectROIs("Select ROIs", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    return rois

def convert_to_min_max(rois):
    return {str(i+1): [int(roi[0]), int(roi[1]), int(roi[0] + roi[2]), int(roi[1] + roi[3])] for i, roi in enumerate(rois)}

l_file_path_str = Path(__file__).resolve()
l_parent_root_path_str = l_file_path_str.parents[0]

camera_id = int(input("Enter Complete path of video file or the camera ID:"))

frame = capture_frame(camera_id)

if frame is not None:

    rois = select_rois(frame)
    if rois is not None:
        print("Selected ROIs:", rois)
        
        # Convert ROIs to a list of tuples and save to a variable
        rois_dict = convert_to_min_max(rois)
        print("ROIs stored in variable:", rois_dict)
        
        # Check if coordinates.json exists
        if os.path.exists(os.path.join(l_parent_root_path_str, 'coordinates.json')):
            # Create a backup
            shutil.copy(os.path.join(l_parent_root_path_str, 'coordinates.json'), os.path.join(l_parent_root_path_str, 'coordinates_original.json'))
            print("Backup created: coordinates_original.json")
        
        # Load existing data
            with open(os.path.join(l_parent_root_path_str, 'coordinates.json'), 'r') as f:
                data = json.load(f)
            # Update ROI values
            data['roi'] = rois_dict
            data['camera_id'] = camera_id
            
            with open(os.path.join(l_parent_root_path_str, 'coordinates.json'), 'w') as f:
                json.dump(data, f, indent=2)
            print("ROIs updated in coordinates.json")
            
            
        print("ROIs saved to coordinates.json")
    else:
        print("No ROIs selected.")
        
else:
    print("Failed to capture frame from camera.")

# if __name__ == "__main__":
#     l_file_path_str = Path(__file__).resolve()
#     l_parent_root_path_str = l_file_path_str.parents[0]
#     print(f"l_parent_root_path_str :- {l_parent_root_path_str}")
    
#     print(f"coordinates path :- {os.path.join(l_parent_root_path_str, 'coordinates.json')}")

