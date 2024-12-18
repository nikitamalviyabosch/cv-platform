import os
import configparser
from pathlib import Path
import datetime
import time
import cv2
from PIL import Image
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase
from anomalib.deploy import OpenVINOInferencer
from anomalib.utils.visualization.image import ImageVisualizer, VisualizationMode
from anomalib import TaskType


class CInference:
    def __init__(self, f_prjct_name, f_cfg_obj, f_prnt_str):
        self.l_prjct_name = f_prjct_name
        self.l_wght_info_dict = eval(f_cfg_obj['DEFAULT']['WEIGHT_INFO_DICT'])
        self.l_resources_pth_str = os.path.join(f_prnt_str, f_cfg_obj['DEFAULT']['RESOURCES_PATH'])
        self.l_mdl_dir_str = os.path.join(f_prnt_str, f_cfg_obj['DEFAULT']['MODEL_PATH'])
        self.l_inf_op_pth_str = os.path.join(f_prnt_str, f_cfg_obj['DEFAULT']['INFERENCE_PATH'])
        self.l_opnvino_pth_str = os.path.join(self.l_mdl_dir_str, self.l_prjct_name,
                                              self.l_wght_info_dict["OPENVINO_MODEL_PATH"])
        self.l_mdl_metadata_pth_str = os.path.join(self.l_mdl_dir_str, self.l_prjct_name,
                                                   self.l_wght_info_dict["METADATA_PATH"])
        self.model = None
        if not self.model:
            self.load_model_f()

    def load_model_f(self):
        if os.path.exists(self.l_opnvino_pth_str) and os.path.exists(self.l_mdl_metadata_pth_str):
            self.model = OpenVINOInferencer(
                path=self.l_opnvino_pth_str,  # Path to the OpenVINO IR model.
                metadata=self.l_mdl_metadata_pth_str,  # Path to the metadata file.
                device="CPU",  # We would like to run it on an Intel CPU.
            )
        else:
            self.model = OpenVINOInferencer(
                path=self.l_opnvino_pth_str,  # Path to the OpenVINO IR model.
                metadata=self.l_mdl_metadata_pth_str,  # Path to the metadata file.
                device="CPU",  # We would like to run it on an Intel CPU.
            )

    def inference_f(self, f_image_obj):
        l_predictions_obj = self.model.predict(image=f_image_obj)
        visualizer = ImageVisualizer(mode=VisualizationMode.FULL, task=TaskType.SEGMENTATION)
        output_image = visualizer.visualize_image(l_predictions_obj)
        l_op_img_obj = Image.fromarray(output_image)

        # Save the PIL image in the output folder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
        output_image_path = os.path.join(self.l_inf_op_pth_str,
                                         f"{self.l_prjct_name}_{l_predictions_obj.pred_label.name}_{timestamp}.jpg")
        l_op_img_obj.save(output_image_path)

        return l_predictions_obj.pred_label.name, output_image


class VideoProcessor(VideoTransformerBase):
    def __init__(self, inference_obj):
        self.inference_obj = inference_obj
        self.frame = None
        self.result = None
        self.result_time = None  # Time when the result was set

    def transform(self, frame):
        self.frame = frame.to_ndarray(format="bgr24")

        # Check if result should still be displayed
        if self.result and self.result_time:
            elapsed_time = time.time() - self.result_time
            if elapsed_time <= 2:  # Display for 2 seconds
                if self.result=='OK':
                    cv2.putText(self.frame, self.result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(self.frame, self.result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                self.result = None  # Clear the result after 2 seconds

        return self.frame


def main(f_part_num_str,f_prj_str):
    # st.write(f"f_part_num_str:- {f_part_num_str}")
    # st.write(f"f_prj_str:- {f_prj_str}")
    # Path setup
    l_file_path_str = Path(__file__).resolve()
    l_parent_root_path_str = l_file_path_str.parents[0]

    # Configuration setup
    # l_prj_str = st.text_input("Enter the part number:")
    l_config_file_path_str = os.path.join(l_parent_root_path_str, "config.cfg")
    l_config_str = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    l_config_str.read(l_config_file_path_str)

    # if not f_part_num_str:
    #     st.warning("Part number not provided")
    #     return

        # Initialize inference object
    l_inf_obj = CInference(f_prj_str, l_config_str, l_parent_root_path_str)

    # Create VideoProcessor instance
    ctx = webrtc_streamer(
        key=f"{f_prj_str}_{f_part_num_str}_infer_streamer",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: VideoProcessor(l_inf_obj),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        if st.button("Predict",key=f"{f_prj_str}_{f_part_num_str}_predict"):
            if ctx.video_processor.frame is not None:
                frame_rgb = cv2.cvtColor(ctx.video_processor.frame, cv2.COLOR_BGR2RGB)
                l_op_str, output_image = l_inf_obj.inference_f(frame_rgb)
                result_str = "NOT OK" if l_op_str == "ABNORMAL" else "OK"
                ctx.video_processor.result = result_str
                ctx.video_processor.result_time = time.time()  # Set the time when the result is set
                st.image(output_image, caption=f"Inference Result: {result_str}")
            else:
                st.warning("No frame captured yet.")


if __name__ == "__main__":
    main()
