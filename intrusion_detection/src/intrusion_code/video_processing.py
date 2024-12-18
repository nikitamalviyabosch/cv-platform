import cv2  
import numpy as np  
import os  
import csv  
import torch  
from datetime import datetime  
from pathlib import Path
import streamlit as st  
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode  
# from src.utils.vis_utils import Visualization  
# from src.utils.iou_utils import IOUUtils  
# from src.utils.common import Common  
from intrusion_detection.src.utils.vis_utils import Visualization  
from intrusion_detection.src.utils.iou_utils import IOUUtils  
from intrusion_detection.src.utils.common import Common  
import av  
  
class VideoProcessor(VideoProcessorBase):  
    def __init__(self, config, f_coord_pth_str, f_op_csv_path_str, f_mdl_pth_str, f_op_scr_path):  
        self.config = config  
        self.visualization = Visualization()  
        self.iou_utils = IOUUtils()  
        self.previous_save_time = datetime.min  
        # self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  
        self.DEVICE = "0" if torch.cuda.is_available() else "cpu"
        print(self.DEVICE)  
  
        self.l_op_scr_pth = f_op_scr_path  
        common = Common()  
        self.json_data = common.load_json(os.path.join(f_coord_pth_str, "coordinates.json"))  
        self.roi = self.load_roi(self.json_data['roi'])  
        self.font = cv2.FONT_HERSHEY_SIMPLEX  
  
        self.model = torch.hub.load('intrusion_detection', 'custom', path=f_mdl_pth_str, source='local', force_reload=True, device=self.DEVICE)  
        self.model.classes = config.model_classes  
        self.model.conf = config.model_conf_score  
  
        self.csv_flag = self.json_data.get('csv_flag', False)  
        self.screenshot_flag = self.json_data.get('screenshot_flag', False)  
  
        if self.csv_flag:  
            self.csv_file_handle = open(os.path.join(f_op_csv_path_str, config.csv_name), mode='a', newline='')  
            self.writer = csv.writer(self.csv_file_handle, delimiter=',')  
            if os.stat(os.path.join(f_op_csv_path_str, config.csv_name)).st_size == 0:  
                self.writer.writerow(config.csv_header)  
        else:  
            self.csv_file_handle = None  
            self.writer = None  
  
        self.frame_number = 1  
  
    def load_roi(self, roi_json):  
        r_values = list(roi_json.values())
        r_correct = np.column_stack((np.array(r_values)[:, 0], np.array(r_values)[:, 1], np.array(r_values)[:, 0] + np.array(r_values)[:, 2], np.array(r_values)[:, 1] + np.array(r_values)[:, 3]))
        r_correct = np.column_stack((np.array(r_values)[:, 0], np.array(r_values)[:, 1], np.array(r_values)[:, 2], np.array(r_values)[:, 3]))

        roi = {'zone' + str(i + 1): r_correct[i] for i in range(len(r_correct))}
        return roi  
  
    def recv(self, frame):  
        img = frame.to_ndarray(format="bgr24")  
        # img_h, img_w = img.shape[:2]
        img = cv2.resize(img,(640,480))

        # flag = True

        # if flag:
        #     st.write(f"Width:- {img_w} & Height:- {img_h}")
        #     flag = False  
  
        result = self.model(img, size=640)  
        det_classes_series = result.pandas().xyxy[0]['name']  
        det_classes = list(det_classes_series)  
        df = result.pandas().xyxy[0]  
        bboxes = df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()  
  
        violation_in_frame = []  
        output_frame = img.copy()  
        img_str = ''  
        inside_roi = []  
        outside_roi = []  
  
        for roi_name, roi_coords in self.roi.items():  
            # roi_coords[0] = int(roi_coords[0]*0.75)
            # roi_coords[1] = int(roi_coords[1]*0.85714)
            # roi_coords[2] = int(roi_coords[2]*0.75)
            # roi_coords[3] = int(roi_coords[3]*0.85714)
            # st.write(f"roi_coords :- {roi_coords}")
            cv2.rectangle(output_frame, (roi_coords[0], roi_coords[1]),  
                          (roi_coords[2], roi_coords[3]),  
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
            if (datetime.now() - self.previous_save_time).total_seconds() > self.config.intrusion_save_time or (  
                    img_str != self.saved_str):  
                self.saved_str = img_str  
                image_file_path = os.path.join(self.l_op_scr_pth,  
                                               '{}_{}.jpg'.format(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"), img_str))  
                if self.screenshot_flag:  
                    cv2.imwrite(image_file_path, output_frame)  
                csv_line = [self.frame_number, image_file_path, datetime.now().strftime("%d-%m-%Y_%H-%M-%S")]  
                if self.writer:  
                    self.writer.writerow(csv_line)  
                self.previous_save_time = datetime.now()  
        else:  
            cv2.putText(output_frame, f'ALL GOOD', self.config.org, self.font, self.config.fontScale,  
                        self.config.det_color_outside_roi, self.config.thickness, cv2.LINE_AA)  
  
        return av.VideoFrame.from_ndarray(output_frame, format="bgr24")  
  
# Streamlit application setup  
def video_processing_main_f(config,f_coord_pth_str,f_op_csv_path_str,f_mdl_pth_str,f_op_scr_path):  
    # st.header("Real-time Video Processing with Streamlit and streamlit-webrtc")  
  
    # Load configuration  
    # config = ...  # Load your configuration here  
    # f_coord_pth_str = ...  # Set the path to your coordinates JSON  
    # f_op_csv_path_str = ...  # Set the path to save CSV files  
    # f_mdl_pth_str = ...  # Set the path to your model file  
    # f_op_scr_path = ...  # Set the path to save screenshots  
  
    # Initialize VideoProcessor with the appropriate arguments  
    video_processor = VideoProcessor(config, f_coord_pth_str, f_op_csv_path_str, f_mdl_pth_str, f_op_scr_path)  
  
    webrtc_ctx = webrtc_streamer(  
        key="example",  
        mode=WebRtcMode.SENDRECV,  
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},  
        video_processor_factory=lambda: video_processor,  
        media_stream_constraints={"video": True, "audio": False},  
        async_processing=True,  
    )  
  
if __name__ == "__main__":  
    video_processing_main_f()  
