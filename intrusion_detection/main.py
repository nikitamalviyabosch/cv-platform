# from config import Config
# from src.intrusion_code.intrusion_detection import IntrusionDetection
# from src.utils.common import Common

from intrusion_detection.config import Config
from .src.intrusion_code.intrusion_detection import IntrusionDetection
from .src.utils.common import Common
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main_intrusion_det_f(f_coord_pth_str,f_op_csv_path_str,f_mdl_pth_str,f_op_scr_path):
    config = Config()

    intrusion_detector = IntrusionDetection(config,f_coord_pth_str,f_op_csv_path_str,f_mdl_pth_str,f_op_scr_path)
    intrusion_detector.start_detection()

if __name__ == "__main__":
    l_op_csv_path_str = r"C:\\CV_Platform\\intrusion_detection/resources\\id_12\\output_csv"
    l_coord_pth_str = r"C:\\CV_Platform\\intrusion_detection/resources\\id_12\\coordinates"
    l_mdl_pth_str = r"C:\\CV_Platform\\intrusion_detection/inference_model/yolov5s.pt"
    l_op_scr_path =  r"C:\\CV_Platform\\intrusion_detection/resources\\id_12\\output_screenshots"
    main_intrusion_det_f(l_coord_pth_str,l_op_csv_path_str,l_mdl_pth_str,l_op_scr_path)

