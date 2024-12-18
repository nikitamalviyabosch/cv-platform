import torch
import cv2
import os
from PIL import Image
from datetime import datetime
from pathlib import Path
# from src.utils.common import Common
# from src.intrusion_code.video_processing import video_processing_main_f #VideoProcessing #load_roi, process_video
from intrusion_detection.src.utils.common import Common
from intrusion_detection.src.intrusion_code.video_processing import video_processing_main_f

# DEVICE = torch.device('cpu')

class IntrusionDetection:
    def __init__(self, config,f_coord_pth_str,f_op_csv_path_str,f_mdl_pth_str,f_op_scr_path):
        # self.video_processing = VideoProcessing(config,f_coord_pth_str,f_op_csv_path_str,f_mdl_pth_str,f_op_scr_path)
        # Need to add LiveStreamProcessing and handle it using config.live_capture flag     
        self.config=  config
        self.f_coord_pth_str = f_coord_pth_str
        self.f_op_csv_path_str = f_op_csv_path_str
        self.f_mdl_pth_str = f_mdl_pth_str
        self.f_op_scr_path = f_op_scr_path

    def start_detection(self):
        video_processing_main_f(self.config,self.f_coord_pth_str,self.f_op_csv_path_str,self.f_mdl_pth_str,self.f_op_scr_path)
        # process_live_stream

# if __name__ == "__main__":
#     model_path = 'yolov5s.pt'
#     json_path = './coordinates.json'
#     video_output_dir = 'output_videos'
#     image_output_dir = 'intrusion_screenshots'
#     csv_output_dir = 'output_csv'

#     intrusion_detector = IntrusionDetection(model_path, json_path, video_output_dir, image_output_dir, csv_output_dir)
#     intrusion_detector.process_video()
