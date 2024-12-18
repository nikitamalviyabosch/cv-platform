import streamlit as st
import cv2
from PIL import Image
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import time
import csv
from datetime import datetime
import av
from .src.utils.config import Config
from .src.utils.common import Common
from .hand_tracker import HandTracker
from collections import Counter
import mediapipe as mp
import os
import cv2
import json
import numpy as np
from skimage.metrics import structural_similarity
cfg = Config()
common = Common()

class LHandTracker:
    # This class should contain all functions
    def __init__(self,json_path):
        # Initialize MediaPipe Hands module
        self.hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.drawing = mp.solutions.drawing_utils

        # json_path = "input_config.json"
        # json_path = r"C:\CV_Platform\input_config.json"
        if not os.path.isfile(os.path.join(json_path,"input_config.json")):
            st.error(f"ROI JSON file not found at {json_path}")
            return

        with open(os.path.join(json_path,"input_config.json"), "r") as f:
            self.roi_data = json.load(f)
        self.number_of_region = self.roi_data.get("number_of_rois", 0)
        # Left hand keypoint
        self.MCP1 = {}
        # Right hand keypoint
        self.MCP2 = {}
        self.roi = []
        self.main_region_roi = []
        self.count = 0
        self.mp_hands = None
        self.mp_drawing = None
        self.input_path = self.roi_data.get('video_file_name', '')
        self.image_similarity_score = 0
        self.list_of_image_similarity_score = []

        self.output_path = None
        self.ret = 1
        self.image = None
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
        self.visited_flags = ['x'] * self.roi_data['number_of_rois']

    def get_camera_fps(self):
        self.fps=30
        #self.fps = self.roi_data['fps']
        return self.fps

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
        self.visited_flags = ['x']*self.roi_data['number_of_rois']
        print("Reset successfull...")

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
    def update_image(self, new_image):
        self.image = new_image

    def compare_images(self, image1, image2):
        # Comparing two images for similarity

        gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        (score, diff) = structural_similarity(gray_image1, gray_image2, full=True)
        return score

    def define_rois(self):
        # defining Roi's variables
        self.roi_timing = [0] * self.number_of_region
        self.roi_timing_in_sec = [0] * self.number_of_region
        self.roi_visited = [None] * self.number_of_region

    def draw_hand_keypoints(self, hand_landmarks):
        if hand_landmarks:
            self.mp_drawing.draw_landmarks(
                self.image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=6),
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))

    def drawing_rect_live(self, x_scale, y_scale):
        # Apply scaling factors
        scaled_main_region_roi = [
            int(self.main_region_roi[0] * x_scale),
            int(self.main_region_roi[1] * y_scale),
            int(self.main_region_roi[2] * x_scale),
            int(self.main_region_roi[3] * y_scale)
        ]
        # Draw the scaled main region rectangle
        cv2.rectangle(self.image, (scaled_main_region_roi[0], scaled_main_region_roi[1]),
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
            cv2.rectangle(self.image, (scaled_roi[0], scaled_roi[1]), (scaled_roi[2], scaled_roi[3]), (0, 255, 0), 1)
            cv2.putText(self.image, f"ROI {i}", (scaled_roi[0], scaled_roi[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1, cv2.LINE_AA)

        return self.roi
    def get_bbox(self):
        self.main_region_roi = self.roi_data['main_region_roi']
        if len(self.main_region_roi) != 0:
            print("roi success")
        for region in range(self.number_of_region):
            self.roi.append(self.roi_data['roi'][str(region)])
            self.count += 1

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

    def calculate_total_time(self):
        if self.total_time:
            self.total_time_count += 1
            text = str(round(self.total_time_count / self.fps, 2)) + "Sec"

            self.image = cv2.putText(self.image, (f"Total time : " + text),
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 1, cv2.LINE_AA)
        elif (self.image_similarity_score >= 0.96) and self.total_time_count != 0:
            self.total_time = False
            text = str(round(self.total_time_count / self.fps, 2)) + "Sec"
            print("cycle time ends - total time value: ", text)

    def is_intersection_of_keypoint_and_roi(self,rect1, cir):
        px = cir["x"]
        py = cir["y"]
        return rect1[0] <= px <= rect1[2] and rect1[1] <= py <= rect1[3]

    def check_intersection_right_keypoints_and_ROI(self, opt):
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
                                            0.5, (255, 0, 0), 1, cv2.LINE_AA)
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
                            self.image = cv2.putText(self.image, (f"ROI " + str(i) + ":" + self.roi_timing_in_sec[i]),
                                                (self.roi[i][0], self.roi[i][1]), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.5, (255, 0, 0), 1, cv2.LINE_AA)
                            # cv2.rectangle(self.image, (self.roi[i][0], self.roi[i][1]),
                            #               (self.roi[i][2], self.roi[i][3]), (0, 255, 0), 1)
                            if self.roi_visited[-1] != i and self.roi_visited[-2] != i:
                                self.roi_visited.append(i)
                    else:
                        self.roi_timing[i] = 0
                        self.roi_timing_in_sec[i] = str(round(self.roi_timing[i] / self.fps, 2)) + "Sec"


    def check_intersection_left_keypoints_and_ROI(self, opt):
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
                                            0.5, (255, 0, 0), 1, cv2.LINE_AA)
                        # cv2.rectangle(self.image, (self.roi[i][0], self.roi[i][1]), (self.roi[i][2], self.roi[i][3]),
                        #               (0, 255, 0), 1)

                        if self.roi_visited[-1] != i and self.roi_visited[-2] != i:
                            self.roi_visited.append(i)

    def define_hand_keypoints(self):
        # Defining hand keypoints object

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils

class HandTrackingProcessor(VideoProcessorBase):
    def __init__(self,part_number,l_json_pth_str,l_op_log_csv_pth_str):
        super().__init__()
        self.ht = LHandTracker(l_json_pth_str)
        self.frame = None
        self.frame_count = 0  # Initialize a frame counter for live
        self.l_op_log_csv_pth_str = l_op_log_csv_pth_str
        # inside while/recv
        self.first_image = None
        self.time_entry_flag = None
        self.first_img_flag = True
        self.records_dict = dict()
        self.key_point_buffer = 0
        self.counter = 0

        self.ht.fps = self.ht.get_camera_fps()
        self.part_number = part_number
        self.ht.define_rois()
        self.ht.define_hand_keypoints()
        print(self.ht.number_of_region)

    def recv(self, frame):
        self.ht.image = frame.to_ndarray(format="bgr24")
        if self.ht.image is None:
            self.ht.count += 1

        self.counter += 1
        # skip few frames
        if self.counter <= 50:
            return frame

        # start
        if self.first_img_flag:
            self.records_dict.update({'Part Number': self.part_number})
            self.first_img_flag = False
            self.time_entry_flag = True

        rgb_frame = cv2.cvtColor(self.ht.image, cv2.COLOR_BGR2RGB)
        frame_height, frame_width = rgb_frame.shape[:2]  #640/480

        x_scale = frame_width / 600
        y_scale = frame_height / 400

        results = self.ht.hands.process(rgb_frame)  # rgb frame

        if self.ht.count == 0 and self.ht.number_of_region != 0:
            self.ht.get_bbox()
            print("___________________Taking New Image___________________")
            self.first_image = self.ht.image[self.ht.main_region_roi[0]:self.ht.main_region_roi[2],
                          self.ht.main_region_roi[1]:self.ht.main_region_roi[3]]

        rois_val = self.ht.drawing_rect_live(x_scale,y_scale)

        start_time = time.time()

        image_roi = self.ht.image[self.ht.main_region_roi[0]:self.ht.main_region_roi[2],
                    self.ht.main_region_roi[1]:self.ht.main_region_roi[3]]


        print("comapring images...")
        self.ht.image_similarity_score = self.ht.compare_images(self.first_image, image_roi)
        self.ht.list_of_image_similarity_score.append(self.ht.image_similarity_score)

        if self.time_entry_flag:
            current_timestamp = datetime.now()
            formatted_timestamp = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            self.records_dict.update({'Timestamp (YYYY/MM/DD HH:MM:SS)': formatted_timestamp})
            time_entry_flag = False
        print("calculations****:")
        print(self.ht.image_similarity_score)
        print(cfg.similarity_threshold)
        print(self.ht.total_time_count)

        if results.multi_hand_landmarks:
            print("hands detected")
            for no_of_hands, hand_landmarks in enumerate(results.multi_hand_landmarks):
                self.ht.draw_hand_keypoints(hand_landmarks)
                opt = self.ht.get_hand_keypoint_info(no_of_hands, hand_landmarks, results, frame_width, frame_height)
                self.ht.check_intersection_left_keypoints_and_ROI(opt)
                self.ht.check_intersection_right_keypoints_and_ROI(opt)

            self.ht.calculate_total_time()

        elif (self.ht.image_similarity_score >= cfg.similarity_threshold) and self.ht.total_time_count != 0:
            self.ht.total_time = False
            text = str(round(self.ht.total_time_count / self.ht.fps, 2)) + "Sec"
            print("cycle time ends - total time value: ", text)
            self.key_point_buffer += 1
            if self.key_point_buffer == 30:
                self.ht.total_time = False
                text = str(round(self.ht.total_time_count / self.ht.fps, 2)) + "Sec"
                print("cycle time ends - total time value: ", text)
                self.records_dict.update({'Cycle Time':text})
                print(f"{self.part_number} cycle time logged successfully...")

                rois_visited_dict = self.ht.counts_of_hand_visited()
                self.records_dict.update(rois_visited_dict)

                # ROI visits
                #st.sidebar.title("ROI Visits History")
                # Display results dynamically based on the ROI visit data

                dict_str = "\n".join([f"{key}: {'✅ Visited' if visited else '❌ Not Visited'}" for key, visited in
                                      rois_visited_dict.items()])
                st.toast(f"100 frames processed!\n\nROI Visit Status:\n{dict_str}", icon="ℹ️")

                # Display the ROI status on the image
                y0, dy = 30, 30
                for i, line in enumerate(dict_str.split('\n')):
                    y = y0 + i * dy
                    cv2.putText(self.ht.image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                with open(os.path.join(self.l_op_log_csv_pth_str,"output_log.csv"), "a", newline="") as f:
                    print("writing to csv ---")
                    w = csv.DictWriter(f, self.records_dict.keys())
                    w.writeheader()
                    w.writerow(self.records_dict)

                self.ht.reset_timers()

                self.first_image = None
                self.counter = 0
                self.key_point_buffer = 0

        self.ht.count += 1

        if not self.ht.roi_data['webcam_flag']:
            # print(ht.visited_flags, ht.roi_visited)
            text = str(round(self.ht.total_time_count / self.ht.fps, 2)) + "Sec"
            print("cycle time ends - total time value: ", text)
            self.records_dict.update({'Cycle Time': text})
            print(f"{self.part_number} cycle time logged successfully...")

            rois_visited_dict = self.ht.counts_of_hand_visited()
            self.records_dict.update(rois_visited_dict)
            with open(os.path.join(self.l_op_log_csv_pth_str,"output_log.csv"), "a", newline="") as f:
                w = csv.DictWriter(f, self.records_dict.keys())
                w.writeheader()
                w.writerow(self.records_dict)

        print(f"Processed frame count: {self.ht.count}")
        return av.VideoFrame.from_ndarray(self.ht.image, format="bgr24")


def video_tracking_page(part_number,l_json_pth_str,l_op_log_csv_pth_str):
    st.title("Hand Keypoints tracking")
    # part_number = st.text_input('Enter part number:', key='part_number_input')
    # source_type = st.radio("Select Video Source", ("Upload a video file", "Live video stream"))
    source_type = "Live video stream"
    st.markdown(
        """
        <style>
        iframe[title="streamlit_webrtc_media_stream"] {
            width: 800px;
            height: 600px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # check the user selection
    if source_type == "Upload a video file":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            ht = HandTracker()
            cap, out, x_scale, y_scale, fps = ht.input_video()
            ht.define_rois()
            ht.define_hand_keypoints()
            st.sidebar.write(f"Number of ROIs: :blue[{ht.number_of_region}]")
            cap = cv2.VideoCapture(temp_file_path)
            process_video(ht, cap, part_number, out, fps)
        else:
            st.write("Please upload a video file to proceed.")

    elif source_type == "Live video stream":
        # Set up WebRTC stream
        webrtc_ctx = webrtc_streamer(
            key="hand-tracking",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: HandTrackingProcessor(part_number,l_json_pth_str,l_op_log_csv_pth_str),  # Use a lambda to create the processor
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True  # Ensures asynchronous processing of frames
        )
        if webrtc_ctx.state.playing:
            st.write("Streaming live video with hand tracking...")


def process_video(ht, cap, part_number, out, fps, is_live_stream=False, webrtc_ctx=None):
    ret = 1
    first_image = None
    records_dict = dict()
    counter = 0
    key_point_buffer = 0
    time_entry_flag = True
    first_img_flag = True
    # Frame skipping counter for live streams
    initial_frame_skip = 30
    frame_skip_counter = 0
    video_display = st.empty()

    # Play and stop buttons are only needed for file uploads, not live streams
    if not is_live_stream:
        play_button = st.button("Play")
        stop_button = st.button("Stop")

        if "playing" not in st.session_state:
            st.session_state.playing = False
        if play_button:
            st.session_state.playing = True
        if stop_button:
            st.session_state.playing = False

    while ret or (is_live_stream and webrtc_ctx and webrtc_ctx.state.playing):
        if cap is not None:
            ret, ht.image = cap.read()
            if not ret:
                break
        elif is_live_stream and webrtc_ctx and webrtc_ctx.video_processor:
            ht.image = webrtc_ctx.video_processor.get_frame()
            if ht.image is None:
                if frame_skip_counter < initial_frame_skip:
                    frame_skip_counter += 1
                    continue
                else:
                    print("No valid frame received from WebRTC.")
                    break
        else:
            st.write("No video frame available.")
            continue

        if is_live_stream or st.session_state.playing:
            if ht.image is None or ht.image.size == 0:
                ht.count += 1
                continue

            counter += 1
            if counter <= 120:
                continue
            if first_img_flag:
                records_dict.update({'Part Number': part_number})
                first_img_flag = False
                time_entry_flag = True

            rgb_frame = cv2.cvtColor(ht.image, cv2.COLOR_BGR2RGB)
            image_height, image_width, _ = ht.image.shape
            x_scale = image_width / 600
            y_scale = image_height / 400
            results = ht.hands.process(rgb_frame)
            if ht.count == 0 and ht.number_of_region != 0:
                ht.get_bbox()
                print("___________________Taking New Image___________________")
                first_image = ht.image[ht.main_region_roi[0]:ht.main_region_roi[2],
                              ht.main_region_roi[1]:ht.main_region_roi[3]]

            start_time = time.time()
            image_roi = ht.image[ht.main_region_roi[0]:ht.main_region_roi[2],
                        ht.main_region_roi[1]:ht.main_region_roi[3]]

            if first_image is not None and first_image.size > 0 and image_roi.size > 0:
                ht.image_similarity_score = ht.compare_images(first_image, image_roi)
                ht.list_of_image_similarity_score.append(ht.image_similarity_score)
            else:
                print("Error: first_image or image_roi is None or empty")

            if time_entry_flag:
                current_timestamp = datetime.now()
                formatted_timestamp = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                records_dict.update({'Timestamp (YYYY/MM/DD HH:MM:SS)': formatted_timestamp})
                time_entry_flag = False

            if results.multi_hand_landmarks:
                for no_of_hands, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    ht.draw_hand_keypoints(hand_landmarks)
                    opt = ht.get_hand_keypoint_info(no_of_hands, hand_landmarks, results, 600, 400)
                    ht.check_intersection_left_keypoints_and_ROI(opt)
                    ht.check_intersection_right_keypoints_and_ROI(opt)
                ht.calculate_total_time()

            elif (ht.image_similarity_score >= cfg.similarity_threshold) and ht.total_time_count != 0:
                ht.total_time = False
                text = str(round(ht.total_time_count / fps, 2)) + "Sec"
                print("cycle time ends - total time value: ", text)
                key_point_buffer += 1
                if key_point_buffer == 30:
                    ht.total_time = False
                    text = str(round(ht.total_time_count / fps, 2)) + "Sec"
                    print("cycle time ends - total time value: ", text)
                    records_dict.update({'Cycle Time': text})
                    print(f"{part_number} cycle time logged...")

                    rois_visited_dict = ht.counts_of_hand_visited()
                    records_dict.update(rois_visited_dict)
                    # ROI visits
                    st.sidebar.title("ROI Visits History")

                    # Display results dynamically based on the ROI visit data
                    for key, visited in rois_visited_dict.items():
                        st.sidebar.checkbox(f"{key}: {'✅ Visited' if visited else '❌ Not Visited'}", value=visited)


                    with open("output_log.csv", "a", newline="") as f:
                        w = csv.DictWriter(f, records_dict.keys())
                        w.writeheader()
                        w.writerow(records_dict)
                    ht.reset_timers()

                    first_image = None
                    counter = 0
                    key_point_buffer = 0

            if not is_live_stream:
                ht.image = ht.apply_rois(ht.image, x_scale, y_scale)

            if out is not None:
                if ht.image is not None and ht.image.size > 0:
                    out.write(ht.image)
                else:
                    print("Error: ht.image is None or empty")

            if ht.image is not None and ht.image.size > 0:
                video_display.image(cv2.cvtColor(ht.image, cv2.COLOR_BGR2RGB))
            else:
                print("Error: ht.image is None or empty")

    if cap is not None:
        cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    video_tracking_page()
