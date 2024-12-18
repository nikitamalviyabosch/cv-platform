import torch
import cv2
from torchvision import transforms as T

import os
import sys
import streamlit as st
import time

sys.path.append(os.getcwd())
from preprocess import *
from postprocess import *
from common_utils import utilities

sys.path.insert(0, 'yolov5')
from yolov5.models.experimental import attempt_load

import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.set_page_config(
    page_title="PPE Detection",
    layout="wide",  
)

if "running" not in st.session_state:
    st.session_state.running = False
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False

# Define unique keys for buttons to avoid DuplicateWidgetID error
run_button_key = "Run Model"
stop_button_key = "Stop Model"


def handle_buttons():
    if st.button(run_button_key, key=run_button_key):
        if not st.session_state.running:
            st.session_state.running = True
            st.session_state.stop_requested = False 
             
    if st.button(stop_button_key, key=stop_button_key):
        st.session_state.running = False
        st.session_state.stop_requested = True  

handle_buttons()

@st.cache_resource()
def image_resize(image,width = None,height=None,inter=cv2.INTER_AREA):
    dim = None
    (h,w,_) = image.shape
    if width is None and height is None:
        return image
    if not width is None:
        r = width/float(w)
        dim = (width,int(h*r))
    resized = cv2.resize(image,dim,interpolation=inter)
    return resized

class Main:
    def __init__(self):
        try:
            self.json_path = "config/config.json"
            self.img_size = 640
            self.conf_thres = 0.25
            self.iou_thres = 0.45
            self.utils = utilities()
            self.config = self.utils.load_json(self.json_path)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
            self.ppe_model = attempt_load(self.utils.getConfigParam("path_to_pt", self.config), device=self.device)
            self.ppe_model = self.ppe_model.to(self.device)

            self.preprocessing = PreProcess(self.device,self.config,self.ppe_model,self.img_size,self.conf_thres,self.iou_thres)
            self.postprocessing = PostProcess(self.device,self.config)

            self.stframe = st.empty()
            self.inferwrite = st.empty()
            
        except Exception as exp:
            print(str(exp))

    

    def process(self):
        camera_prop = self.utils.getConfigParam("camera_list", self.config)
        url = self.utils.getConfigParam("url", camera_prop[0])
        cap = cv2.VideoCapture(url)

        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if ret is False:
                break
            if st.session_state.get("running", False):
                coords = self.preprocessing.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), self.ppe_model)
                frame = self.postprocessing.check_PPE_within_person(frame,coords)
            frame = cv2.resize(frame,(0,0),fx=0.8,fy=0.8)
            frame = image_resize(frame,width=1280)
            self.stframe.image(frame,channels='BGR',width=700)
            inference_time = time.time() - start_time
            self.inferwrite.write(f"Inference Time: {inference_time:.2f} seconds")
                  
            # cv2.imshow('test',cv2.resize(frame,(1280,720)))
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     cap.release()
            #     cv2.destroyAllWindows()

if __name__ == "__main__":
    Main().process()
    
