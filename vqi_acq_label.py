import os
import cv2
import threading
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import av
from typing import Union

# # Ensure the directories exist
# output_folder = "images"
# os.makedirs(output_folder, exist_ok=True)

def Image_name_gen():
    from datetime import datetime
    now = datetime.now()
    year_time = now.strftime("%Y%m%d")
    time_stamp = now.strftime("%H%M%S")
    return year_time, time_stamp

class VideoTransformer(VideoTransformerBase):
    frame_lock: threading.Lock  # Thread-safe lock
    out_image: Union[np.ndarray, None]

    def __init__(self) -> None:
        self.frame_lock = threading.Lock()
        self.out_image = None

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        out_image = frame.to_ndarray(format="bgr24")
        # out_image = cv2.resize(out_image, (640, 480))
        with self.frame_lock:
            self.out_image = out_image
        return out_image

def save_frame(image: np.ndarray, folder: str, prefix: str):
    source = "Camera"
    year_time, time_stamp = Image_name_gen()
    file_path = os.path.join(folder, f'{source}_{year_time}_{time_stamp}_{prefix}.jpg')
    cv2.imwrite(file_path, image)
    # return file_path

def vqi_ds_acq_label_f(l_ok_pth_str,l_notok_pth_str):
    # st.set_page_config(page_title="Data Acquisition and Labeling")
    # st.title("Live Camera Stream with Snapshot Capture")

    # Initialize the WebRTC stream and transformer
    ctx = webrtc_streamer(key="snapshot", video_transformer_factory=VideoTransformer)

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


        # if good or defect:
        #     with ctx.video_transformer.frame_lock:
        #         out_image = ctx.video_transformer.out_image
            
        #     if out_image is not None:
        #         prefix = "good" if good else "defect"
        #         file_path = save_frame(out_image, output_folder, prefix)
        #         st.image(out_image, caption=f"Snapshot saved as {file_path}", use_column_width=True)
        #         st.write(f"Image stored at: {file_path}")
        #     else:
        #         st.warning("No frames available yet.")
    
    # st.stop()

if __name__ == "__main__":
    vqi_ds_acq_label_f()
