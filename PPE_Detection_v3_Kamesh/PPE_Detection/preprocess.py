import torch
import cv2
import numpy as np
from torchvision import transforms as T

import sys

sys.path.insert(0, 'yolov5')
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.augmentations import letterbox

from common_utils import utilities

class PreProcess:
    def __init__(self,
                 device,
                 config_file,
                 ppe_model,
                 img_size=640,
                 conf_thres=0.25,
                 iou_thres=0.45
                 ) -> None:
        try:
            self.img_size = img_size
            self.conf_thres = conf_thres
            self.iou_thres = iou_thres
            self.device = device
            self.config = config_file
            self.utils = utilities()
            self.classes_dict = self.utils.getConfigParam('threshold_dict',self.config['camera_list'][0])
            self.class_names = self.utils.read_classes_file(self.utils.getConfigParam('classes',self.config['camera_list'][0]))
            self.ppe_model = ppe_model
            
        except Exception as exp:
            print(str(exp))

    def preprocess(self, image):
        img = letterbox(image, new_shape=self.img_size)[0]
        img = np.ascontiguousarray(img.transpose((2, 0, 1)))
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        img = img[None]
        return img

    def postprocess_coords(self,pred, img1, img0):

        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        coordinates = {}
        for classids in self.classes_dict:
            coordinates[classids] = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img1.shape[2:], det[:, :4], img0.shape).round()
                for box in (det[:, :6].cpu().numpy()):
                    class_name = self.class_names[int(box[5])]
                    conf = box[4]
                    if class_name in self.classes_dict:
                        if conf > self.classes_dict[class_name]:
                            coordinates[class_name].append((box[0], box[1], box[2], box[3]))
        return coordinates

    @torch.no_grad()
    def predict(self, image, model):
        img = self.preprocess(image)
        pred = model(img)[0]
        bboxes = self.postprocess_coords(pred, img, image)
        return bboxes
    
    