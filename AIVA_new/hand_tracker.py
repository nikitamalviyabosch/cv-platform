import streamlit as st
import cv2
import numpy as np
import json
import os
import tempfile
import mediapipe as mp
import time
from skimage.metrics import structural_similarity
from collections import Counter

from .src.utils.config import Config
cfg = Config()

class HandTracker:
    def __init__(self,json_path):
        # Existing initialization code
        # json_path = "input_config.json"
        if not os.path.isfile(json_path):
            st.error(f"ROI JSON file not found at {json_path}")
            return

        with open(json_path, "r") as f:
            self.roi_data = json.load(f)

        self.number_of_region = self.roi_data.get("number_of_rois", 0)
        self.roi = []
        self.main_region_roi = []
        self.count = 0
        self.input_path = self.roi_data.get('video_file_name', '')
        self.output_path = cfg.video_output_file
        self.ret = 1
        self.image = None
        self.mp_drawing = None
        self.mp_hands = None
        self.hands = None
        self.fps = 0
        self.total_time_count = 0
        self.left_hand_time = 0
        self.right_hand_time = 0
        self.left_hand_time_in_sec = "0"
        self.right_hand_time_in_sec = "0"
        self.total_time = False
        self.image_similarity_score = 0
        self.list_of_image_similarity_score = []
        self.roi_timing_in_sec = None
        self.roi_timing = 0
        self.roi_visited = None
        self.image_at_t0 = None
        self.visited_flags = ['x'] * self.number_of_region

        # Sidebar initialization
        st.sidebar.title("Hand Tracking Info")
        st.sidebar.subheader("Visited ROIs")
        self.sidebar_info = {}

        # Left hand keypoint
        self.MCP1 = {}
        # Right hand keypoint
        self.MCP2 = {}
        self.no_of_hands = 0
        self.old_state = None

    def reset_timers(self):
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
        self.visited_flags = ['x'] * self.number_of_region
        self.sidebar_info.clear()  # Clear sidebar info on reset
        print("Reset successful...")

    def get_camera_fps(self):
        self.fps=30
        # camera_index = self.roi_data['camera_id']
        # # Initialize the VideoCapture object
        # cap = cv2.VideoCapture(camera_index)
        # if not cap.isOpened():
        #     print("Error: Could not open video capture.")
        #     return None
        # # Retrieve FPS
        # self.fps = cap.get(cv2.CAP_PROP_FPS)
        # # Release the video capture object
        # cap.release()
        return self.fps
    @staticmethod
    def is_intersection_of_keypoint_and_roi(rect1, cir):
        px = cir["x"]
        py = cir["y"]
        return rect1[0] <= px <= rect1[2] and rect1[1] <= py <= rect1[3]

    def get_hand_keypoint_info(self, index, hand, results, width, height):
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
        gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        (score, diff) = structural_similarity(gray_image1, gray_image2, full=True)
        return score

    def define_rois(self):
        self.roi_timing = [0] * self.number_of_region
        self.roi_timing_in_sec = [0] * self.number_of_region
        self.roi_visited = [None] * self.number_of_region

    def define_hand_keypoints(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils

    def get_bbox(self):
        self.main_region_roi = self.roi_data['main_region_roi']
        for region in range(self.number_of_region):
            self.roi.append(self.roi_data['roi'][str(region)])
            self.count += 1
    def drawing_rect_live(self, image,x_scale,y_scale):
        # Apply scaling factors
        scaled_main_region_roi = [
            int(self.main_region_roi[0] * x_scale),
            int(self.main_region_roi[1] * y_scale),
            int(self.main_region_roi[2] * x_scale),
            int(self.main_region_roi[3] * y_scale)
        ]
        # Draw the scaled main region rectangle
        cv2.rectangle(image, (scaled_main_region_roi[0], scaled_main_region_roi[1]),
                      (scaled_main_region_roi[2], scaled_main_region_roi[3]),
                      (0, 0, 255), 1)
        # Apply scaling and draw rectangles for each ROI
        for i in range(len(self.roi)):
            scaled_roi = [
                int(self.roi[i][0] * x_scale),
                int(self.roi[i][1] * y_scale),
                int(self.roi[i][2] * x_scale),
                int(self.roi[i][3] * y_scale)
            ]
            cv2.rectangle(image, (scaled_roi[0], scaled_roi[1]), (scaled_roi[2], scaled_roi[3]), (0, 0, 255), 1)
            cv2.putText(image, f"ROI {i}", (scaled_roi[0], scaled_roi[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 1, cv2.LINE_AA)

        return image

    def drawing_rect(self):
        cv2.rectangle(self.image, (self.main_region_roi[0], self.main_region_roi[1]), (self.main_region_roi[2], self.main_region_roi[3]),
                      (0, 0, 255), 1)
        for i in range(len(self.roi)):
            cv2.rectangle(self.image, (self.roi[i][0], self.roi[i][1]), (self.roi[i][2], self.roi[i][3]), (0, 0, 255), 1)
            cv2.putText(self.image, (f"ROI " + str(i)), (self.roi[i][0], self.roi[i][1]), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 1, cv2.LINE_AA)

        return self.roi

    def draw_hand_keypoints(self, hand_landmarks):
        self.mp_drawing.draw_landmarks(
            self.image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=6),
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))

    def check_intersection_left_keypoints_and_ROI(self, opt):
        image_height, image_width, _ = self.image.shape
        x_scale = image_width / 600
        y_scale = image_height / 400

        if opt and opt[2] == 'Left':
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
                        # Apply scaling factors to ROI coordinates for text placement
                        text_x = int(self.roi[i][0] * x_scale)
                        text_y = int(self.roi[i][1] * y_scale)
                        # Draw text on the image
                        self.image = cv2.putText(self.image,
                                                 f"ROI {i}:{self.roi_timing_in_sec[i]}",
                                                 (text_x, text_y),
                                                 cv2.FONT_HERSHEY_SIMPLEX,
                                                 1,
                                                 (255, 0, 0),
                                                 1,
                                                 cv2.LINE_AA)

                        if len(self.roi_visited) < 2 or (self.roi_visited[-1] != i and self.roi_visited[-2] != i): #changed
                            self.roi_visited.append(i)


    def check_intersection_right_keypoints_and_ROI(self, opt):
        image_height, image_width, _ = self.image.shape
        x_scale = image_width / 600
        y_scale = image_height / 400
        if opt and opt[2] == 'Right':
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
                        # Apply scaling factors to ROI coordinates for text placement
                        text_x = int(self.roi[i][0] * x_scale)
                        text_y = int(self.roi[i][1] * y_scale)
                        # Draw text on the image
                        self.image = cv2.putText(self.image,
                                                 f"ROI {i}:{self.roi_timing_in_sec[i]}",
                                                 (text_x, text_y),
                                                 cv2.FONT_HERSHEY_SIMPLEX,
                                                 1,
                                                 (255, 0, 0),
                                                 1,
                                                 cv2.LINE_AA)

                        # cv2.rectangle(self.image, (self.roi[i][0], self.roi[i][1]), (self.roi[i][2], self.roi[i][3]),
                        #               (0, 255, 0), 1)

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
                            # Apply scaling factors to ROI coordinates for text placement
                            text_x = int(self.roi[i][0] * x_scale)
                            text_y = int(self.roi[i][1] * y_scale)
                            # Draw text on the image
                            self.image = cv2.putText(self.image,
                                                     f"ROI {i}:{self.roi_timing_in_sec[i]}",
                                                     (text_x, text_y),
                                                     cv2.FONT_HERSHEY_SIMPLEX,
                                                     1,
                                                     (255, 0, 0),
                                                     1,
                                                     cv2.LINE_AA)
                            # cv2.rectangle(self.image, (self.roi[i][0], self.roi[i][1]),
                            #               (self.roi[i][2], self.roi[i][3]), (0, 255, 0), 1)
                            if self.roi_visited[-1] != i and self.roi_visited[-2] != i:
                                self.roi_visited.append(i)
                    else:
                        self.roi_timing[i] = 0
                        self.roi_timing_in_sec[i] = str(round(self.roi_timing[i] / self.fps, 2)) + "Sec"


    def hand_detection_live(self, image,hands):
        print("hellaida")
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)
        print(len(results))
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, w, c = image.shape



    def hand_detection(self, image):
        image.flags.writeable = False
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(self.image)
        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        h, w, c = self.image.shape

        if results.multi_hand_landmarks:
            for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                self.draw_hand_keypoints(hand_landmarks)
                self.no_of_hands = len(results.multi_handedness)
                self.update_sidebar()
                opt1 = self.get_hand_keypoint_info(0, hand_landmarks, results, w, h)
                opt2 = self.get_hand_keypoint_info(1, hand_landmarks, results, w, h)

                self.check_intersection_left_keypoints_and_ROI(opt1)
                self.check_intersection_right_keypoints_and_ROI(opt2)

        if self.no_of_hands == 0:
            self.left_hand_time = 0
            self.right_hand_time = 0
        return self.image

    def input_video(self):
        # Video processing input and video writer
        if self.roi_data['webcam_flag']:
            cap = cv2.VideoCapture(self.roi_data['camera_id'])
            if not cap.isOpened():
                print("Error: Could not open webcam.")
                exit()
            else:
                pass
        else:
            cap = cv2.VideoCapture(self.input_path)

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        x_scale = frame_width / 600
        y_scale = frame_height / 400
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (int(frame_width), int(frame_height)))
        return cap, video_writer,x_scale,y_scale, self.fps

    def calculate_total_time(self):
        # its calculating total time required for completion of process
        # calculating total time if total_time true else total time close

        if self.total_time:
            self.total_time_count += 1
            text = str(round(self.total_time_count / self.fps, 2)) + "Sec"
            self.image = cv2.putText(self.image, (f"Total time : " + text),
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 1, cv2.LINE_AA)
        elif (self.image_similarity_score >= 0.96) and self.total_time_count != 0:
            self.total_time = False
            text = str(round(self.total_time_count / self.fps, 2)) + "Sec"
            print("cycle time ends - total time value: ", text)

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

    def input_video_for_optimized_det(self):
        if self.roi_data['webcam_flag']:
            cap = cv2.VideoCapture(self.roi_data['camera_id'])
            if not cap.isOpened():
                print("Error: Could not open webcam.")
                exit()
            video_loc = self.roi_data['camera_id']
        else:
            cap = cv2.VideoCapture(self.input_path)
            video_loc = self.input_path

        self.fps = cap.get(cv2.CAP_PROP_FPS)

        # Set a default size if width and height are not used
        size = (640, 480)  # You can adjust this to your preferred default size
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, size)

        return video_loc, cap, video_writer

    def apply_rois(self, frame, x_scale, y_scale):
        # Apply ROI scaling to the frame
        if self.main_region_roi:
            xmin, ymin, xmax, ymax = self.main_region_roi
            xmin = int(xmin * x_scale)
            ymin = int(ymin * y_scale)
            xmax = int(xmax * x_scale)
            ymax = int(ymax * y_scale)
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

        for roi in self.roi:
            xmin, ymin, xmax, ymax = roi
            xmin = int(xmin * x_scale)
            ymin = int(ymin * y_scale)
            xmax = int(xmax * x_scale)
            ymax = int(ymax * y_scale)
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        return frame
    def start(self):
        self.define_rois()
        self.define_hand_keypoints()

        cap = cv2.VideoCapture(self.input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps
        st.sidebar.info(f"Video: {self.input_path}")
        st.sidebar.info(f"FPS: {fps}")

        output_file = os.path.join(cfg.video_output_file)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_file, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        p_time = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = self.hand_detection(frame)
            out.write(image)

            c_time = time.time()
            fps = 1 / (c_time - p_time)
            p_time = c_time

            cv2.imshow('Hand Tracking', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        st.sidebar.success(f"Video processing completed and saved to {output_file}")

# Running the HandTracker
if __name__ == '__main__':
    st.title("Hand Tracking Application")
    hand_tracker = HandTracker()
    hand_tracker.start()
