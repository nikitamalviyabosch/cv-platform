import cv2
import os
import json
import os
import shutil
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from intrusion_detection.config import Config
from intrusion_detection.src.utils.common import Common

def convert_to_min_max(rois):
    return {str(i+1): [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])] for i, roi in enumerate(rois)}

def draw_roi_id_main_f(f_coord_pth_str, f_img_path_str,f_key_str):

    l_coord_dict = {
    "webcam_flag": True,
    "roi": {},
    "csv_flag": True,
    "screenshot_flag": True,
    "inference_video_flag": False
    }


    # img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_pil = Image.open(os.path.join(f_img_path_str,"roi_frame.jpg"))
    # st.image(img_pil, caption='Captured Frame with ROIs', use_column_width=True)

    st.subheader("Draw ROIs")

    canvas_result = st_canvas(
        fill_color="rgba(0, 255, 0, 0.3)",
        stroke_width=2,
        stroke_color="#ff0000",
        background_color="#ffffff",
        background_image=img_pil,
        update_streamlit=True,
        height=480,
        width=640,
        drawing_mode="rect",
        key=f"{f_key_str}_canvas",
    )

    if canvas_result.json_data is not None:
        shapes = canvas_result.json_data.get("objects", [])
        rois = []
        for shape in shapes:

            # Convert to xmin, ymin, xmax, ymax
            # xmin = int(shape['left']*1.333)
            # ymin = int(shape['top']*1.166)
            # xmax = xmin + int(shape['width']*1.333)
            # ymax = ymin + int(shape['height']*1.166)

            xmin = int(shape['left'])
            ymin = int(shape['top'])
            xmax = xmin + int(shape['width'])
            ymax = ymin + int(shape['height'])
            
            rois.append((xmin, ymin, xmax, ymax))
        
        if len(rois)!=0:
            rois_dict = convert_to_min_max(rois)
            l_coord_dict['roi'] = rois_dict

            with open(os.path.join(f_coord_pth_str,'coordinates.json'), 'w') as f:
                json.dump(l_coord_dict, f, indent=2)
            st.success("ROIs updated")
        

if __name__ == "__main__":
    draw_roi_id_main_f()
