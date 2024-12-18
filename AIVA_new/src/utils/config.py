from pathlib import Path
import os
# from src.utils.common import Common
from .common import Common

common = Common()


class Config:

    def __init__(self) -> None:
        self.glove_det_model_path = 'inference_model/last_150.pt'
        self.json_path = 'input_config.json'

        self.roi_object_grabbing_threshold_sec = 1
        self.similarity_threshold = 0.8

        # Video Output Configuration
        self.video_saving_dir = 'output/output_videos'

        if not os.path.exists('/tmp/test'):
            common.create_directories(self.video_saving_dir)

        video_name = 'output.avi'
        self.video_output_file = Path(self.video_saving_dir) / video_name
        self.threshold_iou = 0.2
        self.saved_str = ''

