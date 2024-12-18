import os
import shutil
import configparser
from pathlib import Path
import datetime
import cv2
from PIL import Image
import numpy as np

from anomalib.deploy import OpenVINOInferencer
from anomalib.utils.visualization.image import ImageVisualizer, VisualizationMode
from anomalib import TaskType
# from azure.storage.blob import BlobServiceClient

# from inf_acquisition import inf_acquisition_main_f

class CInference():
    def __init__(self, f_prjct_name, f_cfg_obj, f_prnt_str):
        self.l_prjct_name = f_prjct_name
        self.l_wght_info_dict = eval(f_cfg_obj['DEFAULT']['WEIGHT_INFO_DICT'])
        self.l_resources_pth_str = os.path.join(f_prnt_str, f_cfg_obj['DEFAULT']['RESOURCES_PATH'])
        self.l_mdl_dir_str = os.path.join(f_prnt_str, f_cfg_obj['DEFAULT']['MODEL_PATH'])
        self.l_inf_op_pth_str = os.path.join(f_prnt_str, f_cfg_obj['DEFAULT']['INFERENCE_PATH'])
        # self.l_mdl_dwnld_dir_str = os.path.join(self.l_mdl_dir_str, self.l_prjct_name)
        self.l_opnvino_pth_str = os.path.join(self.l_mdl_dir_str, self.l_prjct_name,
                                              self.l_wght_info_dict["OPENVINO_MODEL_PATH"])
        print(f"self.l_opnvino_pth_str :- {self.l_opnvino_pth_str}")
        self.l_mdl_metadata_pth_str = os.path.join(self.l_mdl_dir_str, self.l_prjct_name,
                                                   self.l_wght_info_dict["METADATA_PATH"])
        print(f"self.l_mdl_metadata_pth_str :- {self.l_mdl_metadata_pth_str}")
        self.model = None

        # if os.path.isdir(os.path.join(self.l_mdl_dir_str, self.l_blob_str)):
        #     self.cleanup_folder(os.path.join(self.l_mdl_dir_str, self.l_blob_str))
        #
        # if os.path.isdir(self.l_inf_op_pth_str):
        #     self.cleanup_folder(self.l_inf_op_pth_str)

        if not self.model:
            self.load_model_f()

    def cleanup_folder(self, folder_path):
        """
        Deletes a specified folder and all its contents.

        :param None
        """
        # Construct the full path to the folder
        # folder_path = os.path.join(self.l_mdl_dir_str, self.l_blob_str)

        # Check if the folder exists
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            try:
                # Delete the folder and all its contents
                shutil.rmtree(folder_path)
                print(f"The folder '{folder_path}' has been deleted successfully.")
            except Exception as e:
                print(f"An error occurred while trying to delete the folder: {e}")
        else:
            print(f"The folder '{folder_path}' does not exist.")

    def load_model_f(self):
        if os.path.exists(self.l_opnvino_pth_str) and os.path.exists(self.l_mdl_metadata_pth_str):
            print(f"Entered if condition of load_model_f()")
            self.model = OpenVINOInferencer(
                path=self.l_opnvino_pth_str,  # Path to the OpenVINO IR model.
                metadata=self.l_mdl_metadata_pth_str,  # Path to the metadata file.
                device="CPU",  # We would like to run it on an Intel CPU.
            )
            print(f"If condition:- Model downloaded")
        else:
            print(f"Entered else condition of load_model_f()")
            # self.download_model_f()
            self.model = OpenVINOInferencer(
                path=self.l_opnvino_pth_str,  # Path to the OpenVINO IR model.
                metadata=self.l_mdl_metadata_pth_str,  # Path to the metadata file.
                device="CPU",  # We would like to run it on an Intel CPU.
            )
            print(f"Else condition:- Model downloaded")

    def inference_f(self, f_image_obj):
        l_predictions_obj = self.model.predict(image=f_image_obj)
        visualizer = ImageVisualizer(mode=VisualizationMode.FULL, task=TaskType.SEGMENTATION)
        output_image = visualizer.visualize_image(l_predictions_obj)
        l_op_img_obj = Image.fromarray(output_image)

        # Save the PIL image in the output folder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
        output_image_path = os.path.join(self.l_inf_op_pth_str, f"{self.l_prjct_name}_{l_predictions_obj.pred_label.name}_{timestamp}.jpg")
        l_op_img_obj.save(output_image_path)
        # upload_directory_to_container(self.l_cncn_dict["CONNECTION_STRING"], self.l_cncn_dict["CONTAINER_NAME"],
        #                               self.l_inf_op_pth_str, f"results")

        # return {"label": l_predictions_obj.pred_label.name, "score": l_predictions_obj.pred_score}
        return l_predictions_obj.pred_label.name

    def inf_acquisition_main_f(self):
        cap = cv2.VideoCapture(3) # 3 for Intel real sense camera
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return

        l_op_str = None
        show_result_until = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame")
                break

            # Get the width and height of the frame
            frame_height, frame_width = frame.shape[:2]
            # Define the position for the text (top right corner)
            text_position = (frame_width - 200, 30)
            # Add the part number string to the frame
            cv2.putText(frame, self.l_prjct_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2,
                        cv2.LINE_AA)

            if l_op_str and show_result_until and datetime.datetime.now() < show_result_until:
                cv2.putText(frame, f"Result: {l_op_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

            cv2.imshow('Video Capture', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                # Convert the captured frame to RGB format
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert the RGB frame to a NumPy array
                frame_array = np.array(frame_rgb)
                print("Frame captured and converted to numpy array")
                print("Shape of the array:", frame_array.shape)
                print("Array data type:", frame_array.dtype)
                l_op_str = self.inference_f(frame_array)
                if l_op_str == "ABNORMAL":
                    l_op_str = "NOT OK"
                else:
                    l_op_str = "OK"
                print(f"l_op_str:- {l_op_str}")
                show_result_until = datetime.datetime.now() + datetime.timedelta(seconds=1)

            elif key == ord('q'):
                print("Logging...")
                print(f"Inference results for {self.l_prjct_name} logged...")
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    l_file_path_str = Path(__file__).resolve()
    l_parent_root_path_str = l_file_path_str.parents[0]


    l_prj_str = input("Enter the part number: ")

    l_config_file_path_str = os.path.join(l_parent_root_path_str, "config.cfg")
    l_config_str = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    l_config_str.read(l_config_file_path_str)



    print(f"Inference started")
    l_inf_obj = CInference(l_prj_str,l_config_str, l_parent_root_path_str)
    l_inf_obj.inf_acquisition_main_f()
    print(f"Inference completed")

