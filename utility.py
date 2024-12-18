import av
import cv2
import numpy as np
import streamlit as st
import threading
from typing import Union
import os
from PIL import Image
from drawing_roi_streamlit import draw_roi_id_main_f 
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode, VideoTransformerBase,
    webrtc_streamer,
)

from vqi_acq_label import VideoTransformer,save_frame
from Anomaly_detection.files.train_app import train_f

# Set the directory where your images are stored
# IMAGE_DIR = 'path/to/your/image/folder'

def list_images(directory):
    """List all image files in the given directory."""
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    return [f for f in os.listdir(directory) if f.lower().endswith(valid_extensions)]

def show_logs_f(l_dir_str):
    # st.title("Image Display App")

    # Get the list of image files
    image_files = list_images(l_dir_str)

    if not image_files:
        st.write("No images found in the directory.")
        return

    # Dropdown to select an image
    selected_image = st.selectbox("Select relevant log", image_files)

    # Display the selected image
    if selected_image:
        image_path = os.path.join(l_dir_str, selected_image)
        image = Image.open(image_path)
        st.image(image, caption=selected_image, use_column_width=True)

def menu_list_f(f_use_case_str):
    if f_use_case_str == "VQI":
        return ["","Data acquisition & labeling", "Training"]
    elif f_use_case_str == "Intrusion detection":
        return ["","Data acquisition", "Data Labeling"]
    elif f_use_case_str == "Assembly assistance":
        return ["","Data acquisition & labeling"]
    else:
        return None
    
    
def data_acquisition_f(l_pth_str,l_key_str):
    """
    Data acquisition function to be called when the option:- 'Data acquisition' is selected
    as a task in the existing projects page. 
    
    This function captures a singe frame upon the press of the capture button. 
    
    Usecases where this is applicable are Intrusion detection & assembly 
    assistance.

    :param None
    """
    ctx = webrtc_streamer(key=f"{l_key_str}_streamer", video_transformer_factory=VideoTransformer)

    if ctx.video_transformer:
        
        if st.button("Capture"):  
            with ctx.video_transformer.frame_lock:
                out_image = ctx.video_transformer.out_image
        
            if out_image is not None:
                cv2.imwrite(os.path.join(l_pth_str,"roi_frame.jpg"),out_image)
                st.success("Frame saved !")
            else:
                st.warning("No frames available yet.")
            

def data_labeling_f(f_coord_pth_str, f_img_path_str,f_key_str):
    """
    Data labeling function to be called when the option:- 'Data Labeling' is selected
    as a task in the existing projects page. 
    
    This function is used to draw ROIs on the frame captured during the data acquisition process.

    Usecases where this is applicable are Intrusion detection & assembly 
    assistance.

    :param None
    """
    draw_roi_id_main_f(f_coord_pth_str, f_img_path_str,f_key_str)
    # pass

def vqi_training_f(f_prj_nm_str):
    """
    Function is used to trigger the training pipeline for the VQI use case.
    :param None
    """
    train_f(f_prj_nm_str)

def vqi_ds_acq_label_f(l_notok_pth_str,l_ok_pth_str,l_key_str):
    """
    Function is used to trigger the data acquisition and labeling for projects with the use case VQI configured in them.
    :param None
    """
    ctx = webrtc_streamer(key=f"{l_key_str}_streamer", video_transformer_factory=VideoTransformer)

    if ctx.video_transformer:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("OK"):  
                with ctx.video_transformer.frame_lock:
                    out_image = ctx.video_transformer.out_image
            
                if out_image is not None:
                    prefix = "good"
                    # file_path = save_frame(out_image, l_ok_pth_str, prefix)
                    save_frame(out_image, l_ok_pth_str, prefix)
                    # st.write("Good Frame saved")
                    # st.image(out_image, caption=f"Snapshot saved as {file_path}", use_column_width=True)
                    # st.write(f"Image stored at: {file_path}")
                else:
                    st.warning("No frames available yet.")
            # good = st.button("Good")
        with col2:
            if st.button("Not OK"):
                with ctx.video_transformer.frame_lock:
                    out_image = ctx.video_transformer.out_image
            
                if out_image is not None:
                    prefix = "defect"
                    # file_path = save_frame(out_image, l_notok_pth_str, prefix)
                    save_frame(out_image, l_notok_pth_str, prefix)
                    # st.write("Bad Frame saved")
                    # st.image(out_image, caption=f"Snapshot saved as {file_path}", use_column_width=True)
                    # st.write(f"Image stored at: {file_path}")
                else:
                    st.warning("No frames available yet.")