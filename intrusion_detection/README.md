# Intrusion Detection using YOLOv5

This project aims to detect intrusions using the YOLOv5 object detection model. The code is modularized into different files for better maintainability and scalability.

## Project Structure

- `main.py`: Main script to initiate the intrusion detection process.
- `config.py`: Static inputs like model path, config file path, etc.
- `drawing_rois.py`: Drawing the ROIs by selecting camera or video file.

- `src/intrusion_code`
  - `intrusion_detection.py`: feed initiation, processing call
  - `video_processing.py`: process the frames, inference model, display output
  - 
- `src/utils`
  - `common.py`: loading json file, creating directories for output saving
  - `iou_utils.py`: IOU calculation and removing the BB after processing
  - `vis_utils.py`: drawing the BB with coordinates parsed
  - 
- `inference_model`: Contains the YOLOv5 model weight file.
- `output`:
  - `intrusion_screenshots`: saves the intrusion events with time stamp.
  - `output_video`: saves the inference video of the stream.
  - `output_csv`: saves the events in a .csv file.

## Setup

### Prerequisites

- Python 3.7+
- pip install -r requirements.txt



### Licensing
- Source code is dependent/built using Yolov5 architecture. So the Yolov5 licensing applies here.


