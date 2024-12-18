import torch
import cv2
from torchvision import transforms as T
import sys
from preprocess import *
from postprocess import *
from common_utils import utilities
sys.path.insert(0, 'yolov5')
from yolov5.models.experimental import attempt_load
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


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
        except Exception as exp:
            print(str(exp))

    def process(self):
        camera_prop = self.utils.getConfigParam("camera_list", self.config)
        webcam_flag = self.utils.getConfigParam("webcam", self.config)

        if webcam_flag:
            camera_id = self.utils.getConfigParam("camera_id", self.config)
            cap = cv2.VideoCapture(int(camera_id))
        else:
            url = self.utils.getConfigParam("url", camera_prop[0])
            cap = cv2.VideoCapture(url)

        while True:
            ret, frame = cap.read()
            if ret is False:
                break

            coords = self.preprocessing.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), self.ppe_model)

            frame = self.postprocessing.check_PPE_within_person(frame,coords)

            cv2.imshow('test',cv2.resize(frame,(1280,720)))

            if cv2.waitKey(1) & 0xFF == ord('q'):

                cap.release()

                cv2.destroyAllWindows()

if __name__ == "__main__":

    Main().process()
