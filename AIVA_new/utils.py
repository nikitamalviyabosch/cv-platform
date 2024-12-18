import json
import os
def write_to_json(file_path, main_region_roi, objects_roi, number_of_region, camera_id, hand_type, video_width, video_height, video_path, video_source, fps):
    try:
        with open(os.path.join(file_path,"input_config.json"), 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, create a new dictionary
        data = {}

    # Update with modified data
    data['main_region_roi'] = main_region_roi
    data['roi']             = objects_roi
    data['number_of_rois']  = number_of_region
    data['camera_id']       = camera_id
    data['webcam_flag']     = True if video_source == "Use Live Camera" else False
    data['scenario']        = 0 if hand_type == "Gloves" else 1
    data['video_width']     = video_width  # Add video width
    data['video_height']    = video_height  # Add video height.
    data['video_file_name'] = video_path
    data['fps'] = fps
    # Write updated JSON data back to the file
    with open(os.path.join(file_path,"input_config.json"), 'w') as file:
        json.dump(data, file, indent=4)

def convert_to_min_max(rois):
    return {str(i): [roi[0], roi[1], roi[2], roi[3]] for i, roi in enumerate(rois)}