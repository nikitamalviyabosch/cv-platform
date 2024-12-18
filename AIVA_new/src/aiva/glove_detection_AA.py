import cv2
from skimage.metrics import structural_similarity
import time
import os
from collections import Counter
# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import FPS
from src.utils.config import Config
from src.utils.common import Common
import torch
from ultralytics import YOLO
import csv
from datetime import datetime

cfg = Config()
common = Common()

class ObjectDetection:
    def __init__(self):
        # self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("using device:", self.device)
        self.model = YOLO(cfg.glove_det_model_path)


    def load_model(self):
        self.model.fuse()
        return  self.model

    def predict(self, frame):
        results = self.model(frame)
        return results

    def plot_bboxes(self, results, frame):
        xyxys = []
        for res in results:
            boxes = res.boxes.cpu().numpy()

            xyxys = boxes.xyxy

            for xyxy in xyxys:
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,255,0))

        return frame, xyxys


class GloveTracking:

    def __init__(self):
        self.json_data = common.load_json(cfg.json_path)
        self.number_of_region = self.json_data['number_of_rois']
        self.left_hand_time = 0
        self.right_hand_time = 0
        self.total_time = False
        self.total_time_count = 0
        self.left_hand_time_in_sec = "0"
        self.right_hand_time_in_sec = "0"
        self.roi_timing = 0
        self.roi = []
        self.main_region_roi = []
        self.old_state = None
        self.count = 0
        self.input_path = self.json_data['video_file_name']
        self.output_path = cfg.video_output_file
        self.ret = 1
        self.image = None
        self.no_of_hands = 0
        self.mp_drawing = None
        self.mp_hands = None
        self.hands = None
        self.fps = 0
        self.image_similarity_score = 0
        self.list_of_image_similarity_score = []
        self.norm_hist1 = None
        self.glove_bboxes = [[0, 0, 0, 0], [0, 0, 0, 0], [0,0,0,0]]
        self.iou_threshold = 0.3
        self.roi_timing_in_sec = None
        self.roi_visited = None
        self.image_at_t0 = None
        self.visited_flags = ['x']*self.json_data['number_of_rois']

    def reset_timers(self):
        print("self.number_of_region : ", self.number_of_region)
        self.left_hand_time = 0
        self.right_hand_time = 0
        self.total_time = False
        self.total_time_count = 0
        self.left_hand_time_in_sec = "0"
        self.right_hand_time_in_sec = "0"
        self.roi_timing = [0] * self.number_of_region
        self.count = -1
        self.roi_timing_in_sec = [0] * self.number_of_region
        self.roi_visited = [None] * self.number_of_region
        self.image_at_t0 = None
        self.visited_flags = ['x'] * self.json_data['number_of_rois']
        print("Reset successfull...")

    @staticmethod
    def find_bounding_box_center(x_min, y_min, x_max, y_max):
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        return (center_x, center_y)

    def find_bounding_box_one_third_center(self, x_min, y_min, x_max, y_max):
        center_x, center_y = self.find_bounding_box_center(x_min, y_min, x_max, y_max)
        center_x = x_min + 2 * (center_x - x_min) // 3
        center_y = y_min + 2 * (center_y - y_min) // 3
        return (center_x, center_y)

    @staticmethod
    def is_point_inside_bbox(x, y, x_min, y_min, x_max, y_max):
        return x_min <= x <= x_max and y_min <= y <= y_max

    def is_intersection_of_gloves_and_roi(self, box_a, box_b,):

        # determine the (x, y)-coordinates of the intersection rectangle
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])

        # compute the area of intersection rectangle
        intersection_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
        box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area

        # iou = intersection_area / float(box_a_area + box_b_area - intersection_area)
        iou = intersection_area / max(min(box_a_area, box_b_area), 1e-6)

        center_bbox_a = self.find_bounding_box_center(box_a[0], box_a[1], box_a[2], box_a[3])
        # center_bbox_b = self.find_bounding_box_center(box_b[0], box_b[1], box_b[2], box_b[3])

        center_bbox_b = self.find_bounding_box_one_third_center(box_b[0], box_b[1], box_b[2], box_b[3])
        cv2.circle(self.image, center_bbox_b, 5, (0, 255, 0), -1)

        if iou >= self.iou_threshold and self.is_point_inside_bbox(center_bbox_b[0], center_bbox_b[1],
                                                                   box_a[0], box_a[1], box_a[2], box_a[3]):
            return True
        else:
            return False

    @staticmethod
    def compare_images(image1, image2):
        # Comparing two image similarity

        gray_image_1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray_image_2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        (score, diff) = structural_similarity(gray_image_1, gray_image_2, full=True)
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
        size = (width, height)
        video_writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, size)

        return cap, video_writer

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

    def define_rois(self):
        # defining Roi's variables

        self.roi_timing = [0] * self.number_of_region
        self.roi_timing_in_sec = [0] * self.number_of_region
        self.roi_visited = [None] * self.number_of_region

    def draw_bbox(self):
        # Drawing Bounding box's for main region and ROI's and storing x,y coordinates

        self.main_region_roi = self.json_data['main_region_roi']
        # cv2.namedWindow("select main region:", flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        # r = cv2.selectROI("select main region:", self.image, showCrosshair=False, fromCenter=False)
        # self.main_region_roi = [r[0], r[1], (r[0] + r[2]), (r[1] + r[3])]
        # main_region = self.image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        # cv2.imshow("Main image", main_region)
        # cv2.waitKey(1)
        # cv2.destroyAllWindows()

        for region in range(self.number_of_region):
            # cv2.namedWindow(f"select the ROI{region}", flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
            # r = cv2.selectROI(f"select the ROI{region}", self.image, showCrosshair=False, fromCenter=False)
            # self.roi.append([r[0], r[1], (r[0] + r[2]), (r[1] + r[3])])
            # cropped_image = self.image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
            # cv2.imshow("Cropped image", cropped_image)
            # cv2.waitKey(10)
            # cv2.destroyAllWindows()

            self.roi.append(self.json_data['roi'][str(region)])

            self.count += 1

    def drawing_rect(self):
        # drawing rectangle of ROI's and main region on each frame

        cv2.rectangle(self.image, (self.main_region_roi[0], self.main_region_roi[1]),
                      (self.main_region_roi[2], self.main_region_roi[3]),
                      (0, 0, 255), 1)

        for i in range(len(self.roi)):
            cv2.rectangle(self.image, (self.roi[i][0], self.roi[i][1]), (self.roi[i][2], self.roi[i][3]), (0, 0, 255), 1)
            cv2.putText(self.image, (f"ROI " + str(i)), (self.roi[i][0], self.roi[i][1]), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 1, cv2.LINE_AA)
        return self.roi

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
            print("cycle time ends - total time value: ", self.total_time_count)

    def check_intersection_of_left_gloves_and_ROI(self):
        for i in range(self.number_of_region):
            if not self.is_intersection_of_gloves_and_roi(self.main_region_roi, self.glove_bboxes[0]):
                self.left_hand_time += 1
                self.left_hand_time_in_sec = str(round(self.left_hand_time / self.fps, 2)) + "Sec"
            else:
                self.left_hand_time = 0
                self.left_hand_time_in_sec = str(round(self.left_hand_time / self.fps, 2)) + "Sec"
                self.image_at_t0 = self.image[self.main_region_roi[0]:self.main_region_roi[2],
                              self.main_region_roi[1]:self.main_region_roi[3]]

            if self.is_intersection_of_gloves_and_roi(self.roi[i], self.glove_bboxes[0]):
                self.total_time = True
                self.roi_timing[i] += 1
                if round(self.roi_timing[i] / self.fps, 2) > cfg.roi_object_grabbing_threshold_sec:
                    self.visited_flags[i] = 'Y'
                    self.roi_timing_in_sec[i] = str(round(self.roi_timing[i] / self.fps, 2)) + "Sec"
                    self.image = cv2.putText(self.image, (f"ROI " + str(i) + ":" + self.roi_timing_in_sec[i]),
                                            (self.roi[i][0], self.roi[i][1]),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1, (255, 0, 0), 1, cv2.LINE_AA)
                    cv2.rectangle(self.image, (self.roi[i][0], self.roi[i][1]),
                                  (self.roi[i][2], self.roi[i][3]), (0, 255, 0), 1)

                    if self.roi_visited[-1] != i and self.roi_visited[-2] != i:
                        self.roi_visited.append(i)

    def check_intersection_of_right_gloves_and_ROI(self):
        for i in range(self.number_of_region):
            if not self.is_intersection_of_gloves_and_roi(self.main_region_roi, self.glove_bboxes[1]):
                self.right_hand_time += 1
                self.right_hand_time_in_sec = str(round(self.right_hand_time / self.fps, 2)) + "Sec"
            else:
                self.right_hand_time = 0
                self.right_hand_time_in_sec = str(round(self.right_hand_time / self.fps, 2)) + "Sec"
                self.image_at_t0 = self.image[self.main_region_roi[0]:self.main_region_roi[2],
                                   self.main_region_roi[1]:self.main_region_roi[3]]

            if self.is_intersection_of_gloves_and_roi(self.roi[i], self.glove_bboxes[1]):
                self.total_time = True
                self.roi_timing[i] += 1
                if round(self.roi_timing[i] / self.fps, 2) > cfg.roi_object_grabbing_threshold_sec:
                    self.visited_flags[i] = 'Y'
                    self.roi_timing_in_sec[i] = str(round(self.roi_timing[i] / self.fps, 2)) + "Sec"
                    self.image = cv2.putText(self.image, (f"ROI " + str(i) + ":" + self.roi_timing_in_sec[i]),
                                        (self.roi[i][0], self.roi[i][1]), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (255, 0, 0), 1, cv2.LINE_AA)
                    cv2.rectangle(self.image, (self.roi[i][0], self.roi[i][1]),
                                  (self.roi[i][2], self.roi[i][3]), (0, 255, 0), 1)
                    if self.roi_visited[-1] != i and self.roi_visited[-2] != i:
                        self.roi_visited.append(i)

        if len(self.glove_bboxes[0]) != 0 and len(self.glove_bboxes[1]) != 0:
            for i in range(self.number_of_region):
                if self.is_intersection_of_gloves_and_roi(self.roi[i], self.glove_bboxes[0]) or \
                        self.is_intersection_of_gloves_and_roi(self.roi[i], self.glove_bboxes[1]):
                    self.total_time = True
                    self.roi_timing[i] += 1
                    if round(self.roi_timing[i] / self.fps, 2) > cfg.roi_object_grabbing_threshold_sec:
                        self.visited_flags[i] = 'Y'
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


def glove_tracking_algorithm():
    obj = ObjectDetection()
    gt = GloveTracking()
    cap, out = gt.input_video()
    gt.define_rois()
    ret = 1
    first_image = None
    # start_time = time.time()
    records_dict = dict()
    counter = 0
    key_point_buffer = 0
    wait_buffer_for_missed_detection = 0
    first_img_flag = True
    
    while ret:
        ret, gt.image = cap.read()
        if gt.image is None:
            gt.count += 1
            continue

        counter += 1
        if counter <= 120:
            continue

        if first_img_flag:
            part_number = input('Enter part number: ')
            records_dict.update({'Part Number': part_number})
            first_img_flag = False
            time_entry_flag = True

        image_height, image_width, _ = gt.image.shape

        # Run inference on the frame
        results = obj.predict(gt.image)

        # get bounding boxes on the frame
        gt.image, xyxys = obj.plot_bboxes(results, gt.image)
        box_xyxy = xyxys

        if gt.count == 0 and gt.number_of_region != 0:
            gt.draw_bbox()
            print("___________________Taking New Image___________________")
            first_image = gt.image[gt.main_region_roi[0]:gt.main_region_roi[2],
                          gt.main_region_roi[1]:gt.main_region_roi[3]]

        image_roi = gt.image[gt.main_region_roi[0]:gt.main_region_roi[2],
                    gt.main_region_roi[1]:gt.main_region_roi[3]]

        # check for image similarities
        gt.image_similarity_score = gt.compare_images(first_image, image_roi)
        gt.list_of_image_similarity_score.append(gt.image_similarity_score)

        b_boxes_tracked = box_xyxy
        rois_val = gt.drawing_rect()
        text_to_print1 = 'Part no.: ' + str(part_number) + '\n'
        text_to_print2 = ''
        for region in range(gt.number_of_region):
            text_to_print2 = text_to_print2 + 'ROI' + str(region) + ' visited: ' + str(gt.visited_flags[region]) + '\n'
        text_to_print = text_to_print1 + text_to_print2
        y0, dy = 50, 30
        for i, line in enumerate(text_to_print.split('\n')):
            y = y0 + i * dy
            gt.image = cv2.putText(gt.image, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX,
                                   1, (128, 0, 128), 1, cv2.LINE_AA)

        start_time = time.time()
        
        if time_entry_flag:
            # Get the current timestamp
            current_timestamp = datetime.now()

            # Format the timestamp
            formatted_timestamp = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            records_dict.update({'Timestamp (YYYY/MM/DD HH:MM:SS)':formatted_timestamp}) 
            # disable the entry time flag.
            time_entry_flag = False

        if len(box_xyxy) > 0:
            # draw tracked objects
            for i, newbox in enumerate(b_boxes_tracked):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[2]), int(newbox[3]))

                if i<len(gt.glove_bboxes):
                    gt.glove_bboxes[i] = [int(newbox[0]), int(newbox[1]),int(newbox[2]), int(newbox[3])]
                else:
                    continue

                # print(gt.glove_bboxes, "\n")
                cv2.rectangle(gt.image, p1, p2, (255, 151, 0), 1)

                # main_region_center_pt = (gt.main_region_roi[0] + ((gt.main_region_roi[2] - gt.main_region_roi[0]) // 2),

                #                          gt.main_region_roi[1] + ((gt.main_region_roi[3] - gt.main_region_roi[1]) // 2))

                # ROI_center_pt = (newbox[0] + ((newbox[2] - newbox[0]) // 2),

                #                  newbox[1] + ((newbox[3] - newbox[1]) // 2))

                # if ROI_center_pt[0] < main_region_center_pt[0]:
                #     gt.check_intersection_of_left_gloves_and_ROI()
                # else:
                #     gt.check_intersection_of_right_gloves_and_ROI()

                gt.check_intersection_of_left_gloves_and_ROI()
                gt.check_intersection_of_right_gloves_and_ROI()

            gt.calculate_total_time()

        elif len(box_xyxy) == 0:
            wait_buffer_for_missed_detection += 1
            continue

        elif (gt.image_similarity_score >= cfg.similarity_threshold) and gt.total_time_count != 0:
            gt.total_time = False
            text = str(round(gt.total_time_count / gt.fps, 2)) + "Sec"
            print("cycle time ends - total time value: ", text)

            key_point_buffer += 1
            if key_point_buffer == 30:
                gt.total_time = False
                text = str(round(gt.total_time_count / gt.fps, 2)) + "Sec"
                print("cycle time ends - total time value: ", text)
                records_dict.update({'Cycle Time': text})
                print(f"{part_number} cycle time logged successfully...")

                rois_visited_dict = gt.counts_of_hand_visited()
                records_dict.update(rois_visited_dict)
                with open("output_log.csv", "a", newline="") as f:
                    w = csv.DictWriter(f, records_dict.keys())
                    w.writeheader()
                    w.writerow(records_dict)

                gt.reset_timers()

                first_image = None
                first_img_flag = True
                counter = 0
                key_point_buffer = 0
                wait_buffer_for_missed_detection = 0
            # break

        out.write(gt.image)
        cv2.namedWindow("Hand Keypoint Detection", flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.imshow('Hand Keypoint Detection', gt.image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    gt.count += 1

    if not gt.json_data['webcam_flag']:
        text = str(round(gt.total_time_count / gt.fps, 2)) + "Sec"
        print("cycle time ends - total time value: ", text)
        records_dict.update({'Cycle Time': text})
        print(f"{part_number} cycle time logged successfully...")

        rois_visited_dict = gt.counts_of_hand_visited()
        records_dict.update(rois_visited_dict)
        with open("output_log.csv", "a", newline="") as f:
            w = csv.DictWriter(f, records_dict.keys())
            w.writeheader()
            w.writerow(records_dict)

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

#def optimized_glove_tracking_algorithm():
#     gt = GloveTracking()
#     video_loc, cap, out = gt.input_video_for_optimized_det()
#     gt.define_rois()
#
#     first_image = None
#     start_time = time.time()
#
#     fvs = FileVideoStream(video_loc).start()
#     time.sleep(1.0)
#     # start the FPS timer
#     fps = FPS().start()
#     # loop over frames from the video file stream
#     while fvs.more():
#         # grab the frame from the threaded video file stream, resize
#         # it, and convert it to grayscale (while still retaining 3
#         # channels)
#
#         frame = fvs.read()
#         if frame is None:
#             break
#
#         gt.image = frame
#         if gt.image is None:
#             gt.count += 1
#             continue
#
#         rgb_frame = cv2.cvtColor(gt.image, cv2.COLOR_BGR2RGB)
#         image_height, image_width, _ = gt.image.shape
#
#         working_dir_path = os.getcwd()
#         temp_loc = os.path.join(working_dir_path, 'src/aiva/output/temp.jpg')
#         # print(temp)
#         # temp_loc = r'../../../src/aiva/output/temp.jpg'
#
#         cv2.imwrite(temp_loc, gt.image)
#
#         box_xyxy = d.run(weights=cfg.glove_det_model_path, source=temp_loc, conf_thres=cfg.model_conf_score,
#                          box_xyxy=[])
#         del temp_loc
#
#         if gt.count == 0 and gt.number_of_region != 0:
#             gt.draw_bbox()
#             first_image = gt.image[gt.main_region_roi[0]:gt.main_region_roi[2],
#                           gt.main_region_roi[1]:gt.main_region_roi[3]]
#
#         start_time = time.time()
#         image_roi = gt.image[gt.main_region_roi[0]:gt.main_region_roi[2],
#                     gt.main_region_roi[1]:gt.main_region_roi[3]]
#
#         # check for image similarities
#         gt.image_similarity_score = gt.compare_images(first_image, image_roi)
#         gt.list_of_image_similarity_score.append(gt.image_similarity_score)
#
#         b_boxes_tracked = box_xyxy
#         gt.drawing_rect()
#
#         if len(box_xyxy) > 0:
#             # draw tracked objects
#             for i, newbox in enumerate(b_boxes_tracked):
#                 p1 = (int(newbox[0]), int(newbox[1]))
#                 p2 = (int(newbox[2]), int(newbox[3]))
#
#                 gt.glove_bboxes[i] = [int(newbox[0]), int(newbox[1]), int(newbox[2]), int(newbox[3])]
#
#                 print(gt.glove_bboxes, "\n")
#                 cv2.rectangle(gt.image, p1, p2, (255, 151, 0), 1)
#
#                 main_region_center_pt = (
#                 gt.main_region_roi[0] + ((gt.main_region_roi[2] - gt.main_region_roi[0]) // 2),
#
#                 gt.main_region_roi[1] + ((gt.main_region_roi[3] - gt.main_region_roi[1]) // 2))
#
#                 ROI_center_pt = (newbox[0] + ((newbox[2] - newbox[0]) // 2),
#
#                                  newbox[1] + ((newbox[3] - newbox[1]) // 2))
#
#                 if ROI_center_pt[0] < main_region_center_pt[0]:
#                     # cv2.putText(gt.image, 'Left Hand', ROI_center_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
#                     gt.check_intersection_of_left_gloves_and_ROI()
#
#                 else:
#                     # cv2.putText(gt.image, 'Right Hand', ROI_center_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
#                     gt.check_intersection_of_right_gloves_and_ROI()
#
#                 # gt.check_intersection_of_left_gloves_and_ROI()
#                 # gt.check_intersection_of_right_gloves_and_ROI()
#
#             gt.calculate_total_time()
#
#         elif (gt.image_similarity_score >= 0.96) and gt.total_time_count != 0:
#             (gt.total_time) = False
#
#         out.write(gt.image)
#         # display the size of the queue on the frame
#         # cv2.putText(gt.image, "Queue Size: {}".format(fvs.Q.qsize()),
#         #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#         cv2.namedWindow("Hand Keypoint Detection", flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
#         cv2.imshow('Hand Keypoint Detection', gt.image)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#
#         # frame = imutils.resize(frame, width=450)
#         # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         # frame = np.dstack([frame, frame, frame])
#
#
#         fps.update()
#
#     gt.count += 1
#     end_time = time.time()
#     tt = abs(end_time - start_time)
#     gt.total_time += 1
#     print(gt.counts_of_hand_visited())
#
#     # Release the capture and close windows
#     cap.release()
#     cv2.destroyAllWindows()
#     gt.counts_of_hand_visited()
#
#     # stop the timer and display FPS information
#     fps.stop()
#     print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
#     print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
#     # do a bit of cleanup
#     cv2.destroyAllWindows()
#     fvs.stop()

# if __name__ == '__main__':
#     gt = GloveTracking()
#     cap, out = gt.input_video()
#     gt.define_rois()
#
#
#     ret = 1
#     first_image = None
#     start_time = time.time()
#     while ret:
#         ret, cap_current_image = cap.read()
#         gt.image = cap_current_image
#
#         if gt.image is None:
#             gt.count += 1
#             continue
#
#         rgb_frame = cv2.cvtColor(gt.image, cv2.COLOR_BGR2RGB)
#         image_height, image_width, _ = gt.image.shape
#
#         temp_loc = '../../src/aiva/output/temp.jpg'
#         cv2.imwrite(temp_loc, gt.image)
#
#         box_xyxy = d.run(weights=cfg.glove_det_model_path, source=temp_loc, conf_thres=cfg.model_conf_score, box_xyxy=[])
#         del temp_loc
#
#         if gt.count == 0 and gt.number_of_region != 0:
#
#             gt.draw_bbox()
#             first_image = gt.image[gt.main_region_roi[0]:gt.main_region_roi[2],
#                           gt.main_region_roi[1]:gt.main_region_roi[3]]
#
#         start_time = time.time()
#         image_roi = gt.image[gt.main_region_roi[0]:gt.main_region_roi[2],
#                     gt.main_region_roi[1]:gt.main_region_roi[3]]
#
#         # check for image similarities
#         gt.image_similarity_score = gt.compare_images(first_image, image_roi)
#         gt.list_of_image_similarity_score.append(gt.image_similarity_score)
#
#         b_boxes_tracked = box_xyxy
#         gt.drawing_rect()
#
#         if len(box_xyxy) > 0:
#             # draw tracked objects
#             for i, newbox in enumerate(b_boxes_tracked):
#                 p1 = (int(newbox[0]), int(newbox[1]))
#                 p2 = (int(newbox[2]), int(newbox[3]))
#                 # print(p1, p2)
#
#                 gt.glove_bboxes[i] = [int(newbox[0]), int(newbox[1]),int(newbox[2]), int(newbox[3])]
#
#                 print(gt.glove_bboxes, "\n")
#                 cv2.rectangle(gt.image, p1, p2, (255, 151, 0), 1)
#
#                 gt.check_intersection_of_left_gloves_and_ROI()
#                 gt.check_intersection_of_right_gloves_and_ROI()
#
#             gt.calculate_total_time()
#
#         elif (gt.image_similarity_score >= 0.96) and gt.total_time_count != 0:(gt.total_time) = False
#
#         out.write(gt.image)
#         cv2.namedWindow("Hand Keypoint Detection", flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
#         cv2.imshow('Hand Keypoint Detection', gt.image)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     gt.count += 1
#     end_time = time.time()
#     tt = abs(end_time - start_time)
#     gt.total_time += 1
#     print(gt.counts_of_hand_visited())
#
#     # Release the capture and close windows
#     cap.release()
#     cv2.destroyAllWindows()
#     gt.counts_of_hand_visited()