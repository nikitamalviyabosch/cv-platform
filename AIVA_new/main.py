import streamlit as st
from roi_capture import roi_capture_page
from video_tracking import video_tracking_page
import warnings
warnings.filterwarnings("ignore")

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page:", ("ROI Capture", "Video Tracking"))

    if page == "ROI Capture":
        roi_capture_page()
    elif page == "Video Tracking":
        video_tracking_page()


if __name__ == "__main__":
    main()
