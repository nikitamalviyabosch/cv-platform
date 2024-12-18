import datetime

import torch
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms as T
import math
# from shapely.geometry import Point,Polygon

import sys
import json

sys.path.insert(0, 'yolov5')
from yolov5.utils.general import non_max_suppression, scale_boxes,cv2,xyxy2xywh
from yolov5.models.experimental import attempt_load
from yolov5.utils.augmentations import letterbox

# sys.path.insert(0, 'strong_sort')
# from strong_sort import StrongSORT

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
            self.classes = ["helmet","vest","shoe","person"]

        except Exception as exp:
            print(str(exp))

    def preprocess(self, image):
        # print(image.shape)
        img = letterbox(image, new_shape=self.img_size)[0]
        img = np.ascontiguousarray(img.transpose((2, 0, 1)))
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        img = img[None]
        return img

    def postprocess(self,pred, img1, img0,tracker):

        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        bbox = []
        ppe_dict = {}
        for det in pred:
            if len(det):
                
                det[:, :4] = scale_boxes(img1.shape[2:], det[:, :4], img0.shape).round()
                for box in (det[:, :6].cpu().numpy()):
                    
                    # if self.classes[int(box[5])] == 'person':
                    #     tracker.update(xywhs.cpu(), confs.cpu(), clss.cpu(), img0)
                    # else:
                    bbox.append((box[0], box[1], box[2], box[3], box[4],self.classes[int(box[5])]))
                    
                    # cv2.putText(frame, f"person : {track_id}", (startX, startY - 5),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return bbox

    @torch.no_grad()
    def predict(self, image, model,tracker):
        img = self.preprocess(image)
        pred = model(img)[0]
        bboxes = self.postprocess(pred, img, image,tracker)
        return bboxes

    # def person_within_rect(self,person_foot_coord, queue_zone_dict,value_sent):
    #     person_status = False
    #     try:
    #         count_foot_inside_polygon = 1
    #         if len(queue_zone_dict) > 0:
    #             count_foot_inside_polygon = 0
    #             for keyvalue in queue_zone_dict.keys():
    #                 if keyvalue == value_sent:
    #                     poly = queue_zone_dict[keyvalue][0]
    #                     if poly.contains(Point(person_foot_coord)):
    #                         count_foot_inside_polygon = count_foot_inside_polygon + 1
    #         if count_foot_inside_polygon > 0:
    #             person_status = True

    #     except Exception as exp:
    #         print("error while mapping the person_track_box" + str(exp))
    #     return person_status
    
    def getConfigParam(self,strKey, dict):
            try:
                if strKey in dict.keys():
                    return dict[strKey]
                else:
                    return ''
            except:
                print(str(strKey) + "not available")


    # def get_zone_cords(self,camera_prop):
    #     queue_zone_dict = {}
    #     if self.getConfigParam("zone_cords", camera_prop[0]) != '':
    #         queue_zone_dict_raw = self.getConfigParam("zone_cords", camera_prop[0])
    #         for key, value in queue_zone_dict_raw.items():
    #             zone = value[0]
    #             color = tuple(value[1])
    #             pts = np.array(zone, np.int32)
    #             pts = pts.reshape((-1, 1, 2))
    #             # if getConfigParam("is_green_box_enabled", data) == "true":
    #             #     frame = cv2.polylines(frame, [pts], True, color, 2)
    #             zonepts = []
    #             for z in zone:
    #                 zonepts.append(tuple(z))
    #             poly = Polygon(zonepts)
    #             queue_zone_dict[int(key)] = [poly, color]
            
    def check_ppe_inside_person(self,track,person_rect,ppe_bboxes):
        ppe_status = None
        with_ppe = False
        without_ppe = False
        for ppe in ppe_bboxes:
            startX, startY, endX, endY, clss = ppe
            pt = (round((startX + endX) / 2, 2), round((startY + endY) / 2, 2))
            logic = person_rect[0] < pt[0] < person_rect[2] and person_rect[1] < pt[1] < person_rect[3] / 2 + person_rect[1] / 2
            if logic == True and clss.split("_")[0] == "with":
                with_ppe = True
            elif logic == True and clss.split("_")[0] == "without":
                without_ppe = True
        if with_ppe == True:
            ppe_status = True
        elif without_ppe == True and with_ppe == False:
            ppe_status = False

        if len(track.ppe_list) >= 5:
            ppe = max(track.ppe_list,key=track.ppe_list.count)
            if str(ppe).split("_")[0] == "with":
                ppe_status == True
            else:
                ppe_status == False
        else:
            ppe_status = None

        return ppe_status

    def process(self):
        camera_prop = self.getConfigParam("camera_list", config_file)
        url = self.getConfigParam("url", camera_prop[0])
        cap = cv2.VideoCapture(url)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width, frame_height))


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # tracker = StrongSORT(
        #     self.getConfigParam("tracker_model", config_file),
        #     device,
        #     max_dist=0.2,
        #     max_iou_distance=0.7,
        #     max_age=30,
        #     n_init=3,
        #     nn_budget=100,
        #     mc_lambda=0.995,
        #     ema_alpha=0.9,

        # )

        tracker = None
        self.ppe_model = attempt_load(self.getConfigParam("path_to_pt", config_file), device=self.device)
        self.ppe_model = self.ppe_model.to(self.device)

        while True:
            ret, frame = cap.read()
            if ret is False:
                break

            bboxes = self.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), self.ppe_model, tracker)
            for box in bboxes:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)
                cv2.putText(frame, str(box[5]), (int(box[0]), int(box[1]) - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # for track in tracker.tracker.tracks:
            #     if not track.is_confirmed() or track.time_since_update > 1:
            #         continue

            #     track_id = track.track_id
            #     class_id = track.class_id
            #     conf = track.conf
            #     box = track.to_tlwh()
            #     startX, startY, endX, endY = tracker._tlwh_to_xyxy(box)
            #     ppe_status = self.check_ppe_inside_person(tracker._tlwh_to_xyxy(box), bboxes)
            #     if ppe_status == True:
            #         cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 1)
            #         cv2.putText(frame, f"person : {track_id}", (startX,startY- 5),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                

            # out.write(frame)
            cv2.imshow('test',cv2.resize(frame,(640,480)))
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
    



