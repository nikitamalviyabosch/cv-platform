import cv2
import mediapipe as mp
import numpy as np
from skimage.metrics import structural_similarity
import time
from collections import Counter
from src.utils.config import Config
from src.utils.common import Common
from imutils.video import FileVideoStream
from imutils.video import FPS
import os
import csv
from datetime import datetime

cfg = Config()
common = Common()


class HandTracking:

    def __init__(self):
        self.json_data = common.load_json(cfg.json_path)
        self.number_of_region = self.json_data['number_of_rois']
        # Left hand keypoint
        self.MCP1 = {}
        # Right hand keypoint
        self.MCP2 = {}
        self.left_hand_time = 0
        self.right_hand_time = 0
        self.total_time = False
        self.total_time_count = 0
        self.left_hand_time_in_sec = "0"
        self.right_hand_time_in_sec = "0"
        self.roi_timing = 0
        # x, y points of Roi's
        self.roi = []
        # x, y points of main region
        self.main_region_roi = []
        self.old_state = None
        self.count = 0
        self.input_path = self.json_data['video_file_name']
        self.output_path = cfg.video_output_file
        self.ret = 1
        self.image = None
        # Hand key points object
        self.no_of_hands = 0
        self.mp_drawing = None
        self.mp_hands = None
        self.hands = None
        self.fps = 0
        self.image_similarity_score = 0
        self.list_of_image_similarity_score = []
        self.roi_timing_in_sec = None
        self.roi_visited = None
        self.image_at_t0 = None
        self.visited_flags = ['x']*self.json_data['number_of_rois']

    def reset_timers(self):
        self.left_hand_time = 0
        self.right_hand_time = 0
        self.total_time = False
        self.total_time_count = 0
        self.left_hand_time_in_sec = "0"
        self.right_hand_time_in_sec = "0"
        # self.roi_timing = 0
        self.roi_timing = [0] * self.number_of_region
        self.count = -1
        # self.roi_timing_in_sec = None
        self.roi_timing_in_sec = [0] * self.number_of_region
        # self.roi_visited = None
        self.roi_visited = [None] * self.number_of_region
        self.image_at_t0 = None
        self.visited_flags = ['x']*self.json_data['number_of_rois']
        print("Reset successfull...")

    @staticmethod
    def is_intersection_of_keypoint_and_roi(rect1, cir):
        # checks if key point intersects with roi, if yes return True else False

        px = cir["x"]
        py = cir["y"]
        if rect1[0] <= px <= rect1[2] and rect1[1] <= py <= rect1[3]:
            return True
        return False

    def get_hand_keypoint_info(self, index, hand, results, width, height):
        # It returns hand keypoint information like hand coordinates, left hand or right hand etc
        output = None
        for ix, classification in enumerate(results.multi_handedness):
            if classification.classification[0].index == index:
                label = classification.classification[0].label
                score = classification.classification[0].score
                text = "{} {} ".format(label, round(score, 2))
                coords = tuple(np.multiply(
                        np.array((hand.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].x,
                                  hand.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y)),
                    [width, height]).astype(int))
                output = text, coords, label
        return output

    @staticmethod
    def compare_images(image1, image2):
        # Comparing two images for similarity

        gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        (score, diff) = structural_similarity(gray_image1, gray_image2, full=True)
        return score

    def input_video(self):
        # Video processing input and video writer
        if self.json_data['webcam_flag']:
            cap = cv2.VideoCapture(self.json_data['camera_id'])
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            if not cap.isOpened():
                print("Error: Could not open webcam.")
                exit()
        else:
            cap = cv2.VideoCapture(self.input_path)

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (width, height))
        return cap, video_writer

    def define_rois(self):
        # defining Roi's variables

        self.roi_timing = [0] * self.number_of_region
        self.roi_timing_in_sec = [0] * self.number_of_region
        self.roi_visited = [None] * self.number_of_region

    def define_hand_keypoints(self):
        # Defining hand keypoints object

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils

    def draw_bbox(self):
        # Drawing Bounding box's for main region and ROI's and storing x,y coordinates

        self.main_region_roi = self.json_data['main_region_roi']
        # cv2.namedWindow("select main region:", flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        # r = cv2.selectROI("select main region:", self.image, showCrosshair=False, fromCenter=False)
        # self.main_region_roi = [r[0], r[1], (r[0] + r[2]), (r[1] + r[3])]
        # main_region = self.image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        # cv2.imshow("Main image", main_region)
        # cv2.waitKey(10)
        # cv2.destroyAllWindows()

        for region in range(self.number_of_region):
            # cv2.namedWindow(f"select the ROI{region}", flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
            # r = cv2.selectROI(f"select the ROI{region}", self.image, showCrosshair=False, fromCenter=False)
            # self.roi.append([r[0], r[1], (r[0] + r[2]), (r[1] + r[3])])
            self.roi.append(self.json_data['roi'][str(region)])
            # cropped_image = self.image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
            # cv2.imshow("Cropped image", cropped_image)
            # cv2.waitKey(10)
            # cv2.destroyAllWindows()
            self.count += 1

    def drawing_rect(self):
        # drawing rectangle of ROI's and main region on each frame

        cv2.rectangle(self.image, (self.main_region_roi[0], self.main_region_roi[1]), (self.main_region_roi[2], self.main_region_roi[3]),
                      (0, 0, 255), 1)
        for i in range(len(self.roi)):
            cv2.rectangle(self.image, (self.roi[i][0], self.roi[i][1]), (self.roi[i][2], self.roi[i][3]), (0, 0, 255), 1)
            cv2.putText(self.image, (f"ROI " + str(i)), (self.roi[i][0], self.roi[i][1]), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 1, cv2.LINE_AA)

        return self.roi

    def check_intersection_left_keypoints_and_ROI(self, opt):
        # checks left hand info

        # if self.no_of_hands == 0 and opt != None and opt[2] == 'Left':
        if opt != None and opt[2] == 'Left':

            self.MCP1["x"] = int(opt[1][0])
            self.MCP1["y"] = int(opt[1][1])
            for i in range(self.number_of_region):
                if not self.is_intersection_of_keypoint_and_roi(self.main_region_roi, self.MCP1):
                    self.left_hand_time += 1
                    self.left_hand_time_in_sec = str(round(self.left_hand_time / self.fps, 2)) + "Sec"
                else:
                    self.left_hand_time = 0
                    self.left_hand_time_in_sec = str(round(self.left_hand_time / self.fps, 2)) + "Sec"
                    self.image_at_t0 = self.image[self.main_region_roi[0]:self.main_region_roi[2],
                                       self.main_region_roi[1]:self.main_region_roi[3]]

                if self.is_intersection_of_keypoint_and_roi(self.roi[i], self.MCP1):
                    self.total_time = True
                    self.roi_timing[i] += 1

                    if round(self.roi_timing[i] / self.fps, 2) > cfg.roi_object_grabbing_threshold_sec:
                        self.visited_flags[i] = 'Y'
                        self.roi_timing_in_sec[i] = str(round(self.roi_timing[i] / self.fps, 2)) + "Sec"
                        self.image = cv2.putText(self.image, (f"ROI " + str(i) + ":" + self.roi_timing_in_sec[i]),
                                            (self.roi[i][0], self.roi[i][1]),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1, (255, 0, 0), 1, cv2.LINE_AA)
                        cv2.rectangle(self.image, (self.roi[i][0], self.roi[i][1]), (self.roi[i][2], self.roi[i][3]),
                                      (0, 255, 0), 1)

                        if self.roi_visited[-1] != i and self.roi_visited[-2] != i:
                            self.roi_visited.append(i)

    def check_intersection_right_keypoints_and_ROI(self, opt):
        # checks right hand info

        # if self.no_of_hands == 1 and opt != None and opt[2] == 'Right':
        if opt != None and opt[2] == 'Right':

            self.MCP2["x"] = int(opt[1][0])
            self.MCP2["y"] = int(opt[1][1])
            for i in range(self.number_of_region):
                if not self.is_intersection_of_keypoint_and_roi(self.main_region_roi, self.MCP2):
                    self.right_hand_time += 1
                    self.right_hand_time_in_sec = str(round(self.right_hand_time / self.fps, 2)) + "Sec"
                else:
                    self.right_hand_time = 0
                    self.right_hand_time_in_sec = str(round(self.right_hand_time / self.fps, 2)) + "Sec"
                    self.image_at_t0 = self.image[self.main_region_roi[0]:self.main_region_roi[2],
                                       self.main_region_roi[1]:self.main_region_roi[3]]

                if self.is_intersection_of_keypoint_and_roi(self.roi[i], self.MCP2):
                    self.total_time = True
                    self.roi_timing[i] += 1

                    if round(self.roi_timing[i] / self.fps, 2) > cfg.roi_object_grabbing_threshold_sec:
                        self.visited_flags[i] = 'Y'
                        self.roi_timing_in_sec[i] = str(round(self.roi_timing[i] / self.fps, 2)) + "Sec"
                        self.image = cv2.putText(self.image, (f"ROI " + str(i) + ":" + self.roi_timing_in_sec[i]),
                                            (self.roi[i][0], self.roi[i][1]), cv2.FONT_HERSHEY_SIMPLEX,
                                            1, (255, 0, 0), 1, cv2.LINE_AA)
                        cv2.rectangle(self.image, (self.roi[i][0], self.roi[i][1]), (self.roi[i][2], self.roi[i][3]),
                                      (0, 255, 0), 1)
                        if self.roi_visited[-1] != i and self.roi_visited[-2] != i:
                            self.roi_visited.append(i)

            if len(self.MCP1) != 0 and len(self.MCP2) != 0:
                for i in range(self.number_of_region):
                    if self.is_intersection_of_keypoint_and_roi(self.roi[i], self.MCP2) or \
                            self.is_intersection_of_keypoint_and_roi(self.roi[i], self.MCP1):
                        self.total_time = True
                        self.roi_timing[i] += 1
                        if round(self.roi_timing[i] / self.fps, 2) > cfg.roi_object_grabbing_threshold_sec:
                            self.roi_timing_in_sec[i] = str(round(self.roi_timing[i] / self.fps, 2)) + "Sec"
                            self.image = cv2.putText(self.image, (f"ROI " + str(i) + ":" + self.roi_timing_in_sec[i]),
                                                (self.roi[i][0], self.roi[i][1]), cv2.FONT_HERSHEY_SIMPLEX,
                                                1, (255, 0, 0), 1, cv2.LINE_AA)
                            cv2.rectangle(self.image, (self.roi[i][0], self.roi[i][1]),
                                          (self.roi[i][2], self.roi[i][3]), (0, 255, 0), 1)
                            if self.roi_visited[-1] != i and self.roi_visited[-2] != i:
                                self.roi_visited.append(i)
                    else:
                        self.roi_timing[i] = 0
                        self.roi_timing_in_sec[i] = str(round(self.roi_timing[i] / self.fps, 2)) + "Sec"

    def draw_hand_keypoints(self, hand_landmarks):
        # drawing hand key-points on each frame

        self.mp_drawing.draw_landmarks(
            self.image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=6),
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))

    def counts_of_hand_visited(self):
        # its displaying how many times hand visited which ROI
        rois_visited_dict = {}
        count = Counter(self.roi_visited)
        for i in count:
            if i != None:
                print("ROI" + str(i) + " visited " + str(count[i]) + " times.")
                
        for region in range(self.number_of_region):
            if region in count.keys():
                rois_visited_dict.update({"ROI" + str(region): True})
            else:
                rois_visited_dict.update({"ROI" + str(region): False})
        return rois_visited_dict

    def calculate_total_time(self):
        # its calculating total time required for completion of process
        # calculating total time if total_time true else total time close

        if self.total_time:
            self.total_time_count += 1
            text = str(round(self.total_time_count / self.fps, 2)) + "Sec"
            self.image = cv2.putText(self.image, (f"Total time : " + text),
                                (200, 200), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 1, cv2.LINE_AA)
        elif (self.image_similarity_score >= 0.96) and self.total_time_count != 0:
            self.total_time = False
            text = str(round(self.total_time_count / self.fps, 2)) + "Sec"
            print("cycle time ends - total time value: ", text)

    def input_video_for_optimized_det(self):
        # json_data = common.load_json(cfg.json_path)
        if self.json_data['webcam_flag']:
            cap = cv2.VideoCapture(self.json_data['camera_id'])
            if not cap.isOpened():
                print("Error: Could not open webcam.")
                exit()
            video_loc = self.json_data['camera_id']
            self.fps = cap.get(cv2.CAP_PROP_FPS)

        else:
            cap = cv2.VideoCapture(self.input_path)
            video_loc = self.input_path
            self.fps = cap.get(cv2.CAP_PROP_FPS)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (width, height)
        video_writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, size)

        return video_loc, cap, video_writer


def hand_key_point_tracking_algorithm():
    ht = HandTracking()
    cap, out = ht.input_video()
    ht.define_rois()
    ht.define_hand_keypoints()
    ret = 1
    first_image = None
    # start_time = time.time()
    records_dict = dict()
    counter = 0
    key_point_buffer = 0
    wait_buffer_for_missed_detection = 0
    first_img_flag = True

    while ret:
        ret, ht.image = cap.read()
        if ht.image is None:
            ht.count += 1
            continue

        counter += 1
        if counter <= 120:
            continue

        if first_img_flag:

            part_number = input('Enter part number: ')
            records_dict.update({'Part Number': part_number})
            first_img_flag = False
            time_entry_flag = True

        rgb_frame = cv2.cvtColor(ht.image, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = ht.image.shape

        # Detect hand keypoints
        results = ht.hands.process(rgb_frame)

        if ht.count == 0 and ht.number_of_region != 0:
            ht.draw_bbox()
            print("___________________Taking New Image___________________")
            first_image = ht.image[ht.main_region_roi[0]:ht.main_region_roi[2],
                          ht.main_region_roi[1]:ht.main_region_roi[3]]

        rois_val = ht.drawing_rect()
        text_to_print1 = 'Part no.: ' + str(part_number) + '\n'
        text_to_print2 = ''
        for region in range(ht.number_of_region):
            text_to_print2 = text_to_print2 + 'ROI' + str(region) + ' visited: ' + str(ht.visited_flags[region]) + '\n'
        text_to_print = text_to_print1 + text_to_print2
        y0, dy = 50, 30
        for i, line in enumerate(text_to_print.split('\n')):
            y = y0 + i * dy
            ht.image = cv2.putText(ht.image, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX,
                                   1, (128, 0, 128), 1, cv2.LINE_AA)

        start_time = time.time()
        image_roi = ht.image[ht.main_region_roi[0]:ht.main_region_roi[2],
                    ht.main_region_roi[1]:ht.main_region_roi[3]]

        # check for image similarities
        ht.image_similarity_score = ht.compare_images(first_image, image_roi)
        ht.list_of_image_similarity_score.append(ht.image_similarity_score)

        if time_entry_flag:
            # Get the current timestamp
            current_timestamp = datetime.now()
            # Format the timestamp
            formatted_timestamp = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            records_dict.update({'Timestamp (YYYY/MM/DD HH:MM:SS)':formatted_timestamp}) 
            # disable the entry time flag.         
            time_entry_flag = False

        if results.multi_hand_landmarks:

            for no_of_hands, hand_landmarks in enumerate(results.multi_hand_landmarks):
                ht.draw_hand_keypoints(hand_landmarks)
                opt = ht.get_hand_keypoint_info(no_of_hands, hand_landmarks, results, image_width, image_height)
                ht.check_intersection_left_keypoints_and_ROI(opt)
                ht.check_intersection_right_keypoints_and_ROI(opt)

            ht.calculate_total_time()

        elif (ht.image_similarity_score >= cfg.similarity_threshold) and ht.total_time_count != 0:
            ht.total_time = False
            text = str(round(ht.total_time_count / ht.fps, 2)) + "Sec"
            print("cycle time ends - total time value: ", text)
            key_point_buffer += 1
            if key_point_buffer == 30:
                ht.total_time = False
                text = str(round(ht.total_time_count / ht.fps, 2)) + "Sec"
                print("cycle time ends - total time value: ", text)
                records_dict.update({'Cycle Time':text})
                print(f"{part_number} cycle time logged successfully...")

                rois_visited_dict = ht.counts_of_hand_visited()
                records_dict.update(rois_visited_dict)
                with open("output_log.csv", "a", newline="") as f:
                    w = csv.DictWriter(f, records_dict.keys())
                    w.writeheader()
                    w.writerow(records_dict)

                ht.reset_timers()

                first_image = None
                counter = 0
                key_point_buffer = 0

            # break

        out.write(ht.image)
        cv2.namedWindow("Hand Keypoint Detection", flags=cv2.WINDOW_FREERATIO)
        cv2.imshow('Hand Keypoint Detection', ht.image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        ht.count += 1
    if not ht.json_data['webcam_flag']:
        # print(ht.visited_flags, ht.roi_visited)
        text = str(round(ht.total_time_count / ht.fps, 2)) + "Sec"
        print("cycle time ends - total time value: ", text)
        records_dict.update({'Cycle Time': text})
        print(f"{part_number} cycle time logged successfully...")

        rois_visited_dict = ht.counts_of_hand_visited()
        records_dict.update(rois_visited_dict)
        with open("output_log.csv", "a", newline="") as f:
            w = csv.DictWriter(f, records_dict.keys())
            w.writeheader()
            w.writerow(records_dict)
    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

