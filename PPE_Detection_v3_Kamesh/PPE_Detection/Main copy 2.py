import torch
import cv2
import numpy as np
from torchvision import transforms as T

import sys
import json

sys.path.insert(0, 'yolov5')
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.models.experimental import attempt_load
from yolov5.utils.augmentations import letterbox

import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

class Main:
    def __init__(self,
                 config_file,
                 img_size=640,
                 conf_thres=0.25,
                 iou_thres=0.45
                 ) -> None:
        try:
            self.img_size = img_size
            self.conf_thres = conf_thres
            self.iou_thres = iou_thres
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.config = config_file
            self.classes_dict = self.getConfigParam('threshold_dict',config_file['camera_list'][0])
            self.class_names = self.read_classes_file(self.getConfigParam('classes',config_file['camera_list'][0]))
            
        except Exception as exp:
            print(str(exp))

    def getConfigParam(self,strKey, dict):
        try:
            if strKey in dict.keys():
                return dict[strKey]
            else:
                return ''
        except:
            print(str(strKey) + "not available")

    def read_classes_file(self,file_path):
        with open(file_path, 'r') as file:
            # Read lines from the file and strip any extra whitespace
            class_list = [line.strip() for line in file]
        return class_list


    def preprocess(self, image):
        img = letterbox(image, new_shape=self.img_size)[0]
        img = np.ascontiguousarray(img.transpose((2, 0, 1)))
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        img = img[None]
        return img

    def postprocess(self,pred, img1, img0,tracker):

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
    def predict(self, image, model,tracker):
        img = self.preprocess(image)
        pred = model(img)[0]
        bboxes = self.postprocess(pred, img, image,tracker)
        return bboxes
    
    def check_Helmet_within_Person_Coords(self, PPE_rects, person_rect):
        PPE_status = False
        PPE_tracked = None
        try:
            for i, rect in enumerate(PPE_rects):
                startX, startY, endX, endY = rect
                pt = (round((startX + endX) / 2, 2), round((startY + endY) / 2, 2))
                logic = person_rect[0] < pt[0] < person_rect[2] and person_rect[1] < pt[1] < person_rect[3] / 2 + \
                        person_rect[1] / 2
                if logic == True:
                    PPE_status = True
                    PPE_tracked = rect
                    break

        except:
            print("error while mapping the Helmet")
        return PPE_status, PPE_tracked
    
    def check_PPE_within_Person_Coords(self,PPE_rects,person_rect):
        PPE_status = False
        PPE_tracked = None
        try:
            for i,rect in enumerate(PPE_rects):
                startX, startY, endX, endY = rect
                pt = (round((startX+endX)/2,2),round((startY+endY)/2,2))
                logic = person_rect[0] < pt[0] < person_rect[2] and person_rect[1] < pt[1] < person_rect[3]
                if logic==True:
                    PPE_status = True
                    PPE_tracked=rect
                    break

        except:
            print("error while mapping the PPE")
        return PPE_status,PPE_tracked
    
    def check_shoe_within_Person_Coords(self, PPE_rects, person_rect):
        PPE_status = False
        PPE_tracked = None
        try:
            for i, rect in enumerate(PPE_rects):
                startX, startY, endX, endY = rect
                pt = (round((startX + endX) / 2, 2), round((startY) / 2, 2))
                logic = person_rect[0] < pt[0] < person_rect[2] and person_rect[1] < pt[1] < person_rect[3]
                if logic == True:
                    PPE_status = True
                    PPE_tracked = rect
                    break

        except:
            print("error while mapping the Shoe")
        return PPE_status, PPE_tracked
    
    def process(self):
        camera_prop = self.getConfigParam("camera_list", config_file)
        url = self.getConfigParam("url", camera_prop[0])
        cap = cv2.VideoCapture(url)

        tracker = None
        self.ppe_model = attempt_load(self.getConfigParam("path_to_pt", config_file), device=self.device)
        self.ppe_model = self.ppe_model.to(self.device)

        while True:
            ret, frame = cap.read()
            if ret is False:
                break

            coords = self.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), self.ppe_model, tracker)
            if any(coords["person"]):
                for person_rect in coords["person"]:
                    PPE_list = []
                    keysfound = [key for key in coords.keys() if key != "person"]
                    if 'helmet' in keysfound:
                        with_helmet_status, helmet_coords = self.check_Helmet_within_Person_Coords(coords['helmet'], person_rect)
                        if with_helmet_status == True:
                            PPE_list.append('helmet')

                    if 'shoe' in keysfound:
                        with_shoe_status, shoe_coords = self.check_shoe_within_Person_Coords(coords['shoe'], person_rect)
                        if with_shoe_status == True:
                            PPE_list.append('shoe')
                    
                    if 'vest' in keysfound:
                        with_vest_status, vest_coords = self.check_PPE_within_Person_Coords(coords['vest'], person_rect)
                        if with_vest_status == True:
                            PPE_list.append('vest')
                # for box in bboxes:
                #     cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)
                #     cv2.putText(frame, str(box[5]), (int(box[0]), int(box[1]) - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    if len(PPE_list) > 0:
                        (startX, startY, endX, endY) = person_rect
                        pt1 = int(startX), int(startY)
                        pt2 = int(endX), int(endY)
                        for PPE_no, PPE_status in enumerate(PPE_list):
                            color = (0, 255, 0)
                            background_color = (0, 0, 0)
                            cv2.rectangle(frame, pt1, pt2, color, 2)
                            # cv2.putText(frame, PPE_status, (int(startX) + 4, int(startY) - (10 + (PPE_no * 20))),
                            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                            (text_width, text_height), baseline = cv2.getTextSize(PPE_status, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                            # Calculate the bottom-left corner of the text box
                            text_x = int(startX) + 4
                            text_y = int(startY) - (10 + (PPE_no * 20))

                            background_pt1 = (text_x - 2, text_y - text_height - 5)
                            background_pt2 = (text_x + text_width + 2, text_y + 2)
                            cv2.rectangle(frame, background_pt1, background_pt2, background_color, cv2.FILLED)
                            cv2.putText(frame, PPE_status, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


            cv2.imshow('test',cv2.resize(frame,(1280,720)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()

if __name__ == "__main__":
    with open("config/config.json") as json_file:
        config_file = json.load(json_file)
    ppe = Main(config_file,640,
    0.4,
    0.5)
    ppe.process()
    



