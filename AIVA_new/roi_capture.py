import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2
import json
import os
from .utils import write_to_json, convert_to_min_max


def capture_frame(camera_id):
    cap = cv2.VideoCapture(camera_id)
    # Capture frame-by-frame
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def read_frame_from_video(video_path, frame_number=60):
    """Read a specific frame from a video file. By default, it captures the 60th frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Video file could not be opened.")
        return None

    # Set the video position to the specified frame (default is the 60th frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)

    ret, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frame if ret else None,fps


def streamlit_region_rois(image_pil, title, num_regions, canvas_key):
    """Allow the user to draw ROIs on a given image and return the coordinates."""
    st.write(title)
    canvas_result = st_canvas(
        fill_color="rgba(0, 255, 0, 0.3)",
        stroke_width=2,
        stroke_color="#ff0000",
        background_color="#ffffff",
        background_image=image_pil,
        update_streamlit=True,
        drawing_mode="rect",
        key=canvas_key
    )

    rois = []
    if canvas_result.json_data:
        shapes = canvas_result.json_data.get("objects", [])
        if len(shapes) == num_regions:
            for shape in shapes:
                xmin = int(shape['left'])
                ymin = int(shape['top'])
                xmax = xmin + int(shape['width'])
                ymax = ymin + int(shape['height'])
                rois.append((xmin, ymin, xmax, ymax))
            return convert_to_min_max(rois), True

    return {}, False


def save_roi_data(frame, main_region_roi, objects_roi, number_of_region, camera_id, hand_type, video_source,
                  video_path, fps,f_file_path_str):
    """Save the ROI data and frame details to a JSON file."""
    video_resolution = (frame.shape[1], frame.shape[0])
    # file_path = "input_config.json"

    write_to_json(
        f_file_path_str,
        main_region_roi['0'],
        objects_roi,
        number_of_region,
        camera_id,
        hand_type,
        video_resolution[0],
        video_resolution[1],
        video_path,
        video_source,
        fps
    )

    st.session_state.captured_frame = frame
    st.session_state.roi_data = objects_roi

    st.success("JSON file updated successfully.")


def handle_frame_operations(frame, camera_id, video_source, video_path, fps, f_file_path_str):
    """Handle the process of capturing ROIs and saving the data."""
    if frame is None:
        st.error("No frame available for ROI operations.")
        return

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    st.image(img_pil, caption='Frame with ROIs', use_column_width=True)

    main_region_roi, roi_done = streamlit_region_rois(img_pil, "Draw Main Region ROI", 1, "main_region_canvas")
    if roi_done:
        number_of_region = st.number_input("Please enter number of regions: ", value=1, min_value=1)
        hand_type = st.selectbox("Select Type", ("Bare Hands", "Gloves"),
                                 help="Select Gloves if black gloves have been worn.")

        objects_roi, object_roi_done = streamlit_region_rois(img_pil, "Draw objects ROI", number_of_region,
                                                             "objects_canvas")
        if object_roi_done:
            save_roi_data(
                frame, main_region_roi, objects_roi, number_of_region,
                camera_id if video_source == "Use Live Camera" else "N/A",
                hand_type, video_source, video_path, fps,f_file_path_str
            )
        else:
            st.warning(f"Please draw exactly {number_of_region} sub-regions.")
    else:
        st.error("Please draw the main region ROI.")


def roi_capture_page(f_file_path_str):
    """Main function for the ROI capture page."""
    # st.title("ROI Capture")

    # st.sidebar.header("Settings")

    camera_id, video_path, fps = None, None, None

 
    # # st.sidebar.subheader("Camera Settings")
    # camera_id = st.sidebar.number_input("Enter the camera ID:", min_value=0, value=0)
    camera_id = 2 #For the real sense camera
    if st.button('Capture Frame'):
        frame = capture_frame(camera_id)
        fps = 30 # Default FPS for live camera (adjust if necessary)

        if frame is not None:
            st.session_state.frame = frame
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption='Captured Frame', use_column_width=True)
            st.success(f"Frame captured successfully. FPS: {fps}")
        else:
            st.error("Failed to capture frame from camera.")

    if 'frame' not in st.session_state:
        st.error("No frame captured or extracted. Please capture or upload a video frame first.")
        return

    handle_frame_operations(
        st.session_state.frame,
        camera_id,
        "Use Live Camera",
        video_path,
        fps,
        f_file_path_str
    )


if __name__ == "__main__":
    roi_capture_page()
