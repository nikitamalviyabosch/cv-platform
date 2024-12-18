import os
from pathlib import Path

class Config:
    def __init__(self) -> None:
        # self.model_path = '/inference_model/yolov5s.pt'
        # self.json_path = 'coordinates.json'
        
        # self.image_output_dir = 'output/intrusion_screenshots'
        

        # Yolo Model Parameters
        self.model_conf_score = 0.5
        self.model_classes = [0]  # selecting 'person' class from COCO

        # Capturing Parameters
        ##self.live_capture = False
        ##self.live_capture_camera_id = 0

        # Video Output Configuration
        # video_saving_dir = 'output/output_videos'
        # video_name = 'output.mp4'
        # self.video_output_file = Path(video_saving_dir)/video_name
        self.org = (0, 30)
        self.fontScale = 1
        self.thickness = 1
        self.det_color_inside_roi = (0, 0, 255)  # Red
        self.det_color_outside_roi = (0, 255, 0)  # Green
        self.threshold_iou = 0.2
        self.saved_str = ''
        self.intrusion_save_time = 5 # in seconds


        # CSV Output Configuration
        # csv_saving_dir = 'output/output_csv'
        self.csv_name = 'output.csv'
        # self.csv_output_file = Path(csv_saving_dir)/csv_name
        self.csv_header = ['frame_number', 'zone_time', 'event_time']

