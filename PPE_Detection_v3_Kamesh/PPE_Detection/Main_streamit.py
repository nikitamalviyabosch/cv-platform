import torch
import cv2
import numpy as np
from torchvision import transforms as T
from pathlib import Path
import sys
import os
import av
import json
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# sys.path.insert(0, 'yolov5')
# from yolov5.utils.general import non_max_suppression, scale_boxes
# from yolov5.models.experimental import attempt_load
# from yolov5.utils.augmentations import letterbox
from PPE_Detection_v3_Kamesh.PPE_Detection.yolov5.utils.general import non_max_suppression, scale_boxes
from PPE_Detection_v3_Kamesh.PPE_Detection.yolov5.models.experimental import attempt_load
from PPE_Detection_v3_Kamesh.PPE_Detection.yolov5.utils.augmentations import letterbox
import pathlib

# Ensure correct path class based on OS
if sys.platform.startswith('win'):
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath


class Main:
    def __init__(self,l_parent_root_path_str, config_file, img_size=640, conf_thres=0.25, iou_thres=0.45) -> None:
        try:
            self.img_size = img_size
            self.conf_thres = conf_thres
            self.iou_thres = iou_thres
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.config = config_file
            self.classes_dict = self.getConfigParam('threshold_dict', config_file['camera_list'][0])
            self.class_names = list(self.classes_dict.keys())
            self.ppe_model = attempt_load(os.path.join(l_parent_root_path_str,config_file["path_to_pt"]), device=self.device)
            self.ppe_model = self.ppe_model.to(self.device)
        except Exception as exp:
            print(str(exp))

    def getConfigParam(self, strKey, dict):
        try:
            if strKey in dict.keys():
                return dict[strKey]
            else:
                return ''
        except:
            print(str(strKey) + "not available")

    def preprocess(self, image):
        img = letterbox(image, new_shape=self.img_size)[0]
        img = np.ascontiguousarray(img.transpose((2, 0, 1)))
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        img = img[None]
        return img

    def postprocess(self, pred, img1, img0, tracker):
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        bbox = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img1.shape[2:], det[:, :4], img0.shape).round()
                for box in (det[:, :6].cpu().numpy()):
                    class_name = self.class_names[int(box[5])]
                    conf = box[4]
                    if class_name in self.classes_dict:
                        if conf > self.classes_dict[class_name]:
                            bbox.append((box[0], box[1], box[2], box[3], box[4], class_name))
        return bbox

    @torch.no_grad()
    def predict(self, image, tracker):
        img = self.preprocess(image)
        pred = self.ppe_model(img)[0]
        bboxes = self.postprocess(pred, img, image, tracker)
        return bboxes


class VideoProcessor(VideoProcessorBase):
    def __init__(self, model):
        self.model = model
        self.tracker = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        bboxes = self.model.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), self.tracker)
        for box in bboxes:
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)
            cv2.putText(img, str(box[5]), (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main_inf_ppe_f(l_key_str):
    # st.title("YOLOv5 Object Detection with Streamlit WebRTC")
    # st.write("Click the button below to start detection")

    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    # Load configuration file
    l_file_path_str = Path(__file__).resolve()
    l_parent_root_path_str = l_file_path_str.parents[0]
    with open(os.path.join(l_parent_root_path_str, "config/config.json")) as json_file:
        config_file = json.load(json_file)

    model = Main(l_parent_root_path_str,config_file, 640, 0.4, 0.5)

    # if st.button("Predict"):
    webrtc_streamer(key=l_key_str, video_processor_factory=lambda: VideoProcessor(model),
                    rtc_configuration=rtc_config)


if __name__ == "__main__":
    main_inf_ppe_f()
