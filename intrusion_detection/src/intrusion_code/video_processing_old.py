import cv2
import numpy as np
import os
import csv
import torch
from datetime import datetime
from src.utils.vis_utils import Visualization
from src.utils.iou_utils import IOUUtils
from src.utils.common import Common
from pathlib import Path
import streamlit as st


# new_path = l_parent_root_path_str(os.sep)
# new_path.remove('src')
# new_joined_path = os.sep.join(new_path)

class VideoProcessing:
    def __init__(self, config,f_coord_pth_str,f_op_csv_path_str,f_mdl_pth_str,f_op_scr_path):
        """
        config:- Config file containing font thickness,font Scale,threshold_iou(IOU threshold),det_color_inside_roi(Color info),det_color_outside_roi(Color info),org,intrusion_save_time(Integer)
        f_coord_pth_str:- Path to JSON containing the bbox coordinates,webcam_flag(Whether feed is coming from the livefeed or offline video),csv_flag(Flag to save the csv),screenshot_flag(Flag to save the screenshot), inference_video_flag(Falg to save the video)
        f_op_csv_path_str:- Path to save the csv
        f_op_scr_path: Path to save the inference screenshots
        """
        self.config = config
        self.visualization = Visualization()
        self.iou_utils = IOUUtils()
        self.previous_save_time = datetime.min
        self.DEVICE = "0" if torch.cuda.is_available() else "cpu"
        print(self.DEVICE)
        # self.DEVICE = "cpu"

        self.l_op_scr_pth = f_op_scr_path

        common = Common()
        self.json_data = common.load_json(os.path.join(f_coord_pth_str,"coordinates.json"))
        self.roi = self.load_roi(self.json_data['roi'])
        self.webcam_flag = self.json_data.get('webcam_flag', False)
        self.camera_id = self.json_data.get('camera_id', 0)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        if self.webcam_flag:
            self.camera = cv2.VideoCapture(self.camera_id)
        else:
            self.camera = cv2.VideoCapture(self.json_data['video_file_name']) 
        self.fps = self.camera.get(cv2.CAP_PROP_FPS)

        self.out = None
        self.frame_number = 1

        self.model = torch.hub.load('.', 'custom', path=f_mdl_pth_str, source='local', force_reload=True, device=self.DEVICE)
        self.model.classes = config.model_classes
        self.model.conf = config.model_conf_score

        self.csv_flag = self.json_data.get('csv_flag', False)
        self.inference_video_flag = self.json_data.get('inference_video_flag', False)
        self.screenshot_flag = self.json_data.get('screenshot_flag', False)

        if self.csv_flag:
            self.csv_file_handle = open(os.path.join(f_op_csv_path_str,config.csv_name), mode='a', newline='')
            self.writer = csv.writer(self.csv_file_handle, delimiter=',')
            if os.stat(config.csv_output_file).st_size == 0:
                self.writer.writerow(config.csv_header)
        else:
            self.csv_file_handle = None
            self.writer = None
        
        self.fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.frame_number = 1

    def load_roi(self, roi_json):
        r_values = list(roi_json.values())
        r_correct = np.column_stack((np.array(r_values)[:, 0], np.array(r_values)[:, 1], np.array(r_values)[:, 0] + np.array(r_values)[:, 2], np.array(r_values)[:, 1] + np.array(r_values)[:, 3]))
        r_correct = np.column_stack((np.array(r_values)[:, 0], np.array(r_values)[:, 1], np.array(r_values)[:, 2], np.array(r_values)[:, 3]))

        roi = {'zone' + str(i + 1): r_correct[i] for i in range(len(r_correct))}
        return roi

    def process_video(self):
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print('Video Ended...')
                break

            img_h, img_w = frame.shape[:2]

            if self.out is None and self.inference_video_flag:
                self.out = cv2.VideoWriter(str(self.config.video_output_file), self.fourcc, self.fps, (img_w, img_h))

            result = self.model(frame, size=640)
            det_classes_series = result.pandas().xyxy[0]['name']
            det_classes = list(det_classes_series)
            df = result.pandas().xyxy[0]
            bboxes = df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()

            violation_in_frame = []
            output_frame = frame.copy()
            img_str = ''
            inside_roi = []
            outside_roi = []
            
            for roi_name, roi_coords in self.roi.items():
                cv2.rectangle(output_frame, (int(roi_coords[0]), int(roi_coords[1])), 
                            (int(roi_coords[2]), int(roi_coords[3])),
                            (0, 255, 255), self.config.thickness, self.config.fontScale)
                cv2.putText(output_frame, roi_name, (roi_coords[0], roi_coords[1] - 10), 
                            self.font, self.config.fontScale, (255, 0, 0), self.config.thickness, cv2.LINE_AA)

                for box in bboxes:
                    iou = self.iou_utils.calculate_iou(roi_coords, box)
                    x, y, w, h = box
                    if iou > self.config.threshold_iou:
                        violation_in_frame.append(True)
                        inside_roi.append([x, y, w, h])
                        if roi_name not in img_str:
                            img_str = img_str + str(roi_name) + "_"
                    else:
                        violation_in_frame.append(False)
                        outside_roi.append([x, y, w, h])

            outside_roi = self.iou_utils.remove_duplicates(outside_roi, inside_roi)
            self.visualization.draw_rectangles(output_frame, inside_roi, self.config.det_color_inside_roi)
            self.visualization.draw_rectangles(output_frame, outside_roi, self.config.det_color_outside_roi)

            if any(violation_in_frame):
                cv2.putText(output_frame, 'VIOLATION', self.config.org, self.font, self.config.fontScale, 
                            (0, 0, 255), self.config.thickness, cv2.LINE_AA)
                if (datetime.now() - self.previous_save_time).total_seconds() > self.config.intrusion_save_time or (img_str != self.saved_str):
                    self.saved_str = img_str
                    image_file_path = os.path.join(self.l_op_scr_pth, '{}_{}.jpg'.format(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"), img_str))
                    if self.screenshot_flag:
                        ppp = cv2.imwrite(image_file_path, output_frame)
                    csv_line = [self.frame_number, image_file_path, datetime.now().strftime("%d-%m-%Y_%H-%M-%S")]
                    if self.writer:
                        self.writer.writerow(csv_line)
                    self.previous_save_time = datetime.now()
            else:
                cv2.putText(output_frame, 'ALL GOOD', self.config.org, self.font, self.config.fontScale, 
                            self.config.det_color_outside_roi, self.config.thickness, cv2.LINE_AA)
            cv2.imshow('output_result', output_frame)
            if self.out:
                self.out.write(output_frame)
            self.frame_number += 1

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
 
        self.camera.release()
        if self.out:
            self.out.release()
        if self.csv_file_handle:
            self.csv_file_handle.close()
        cv2.destroyAllWindows()
