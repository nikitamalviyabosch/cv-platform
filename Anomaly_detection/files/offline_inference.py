import os
import shutil
import configparser
from pathlib import Path
from PIL import Image
import numpy as np

from anomalib.deploy import OpenVINOInferencer
from anomalib.utils.visualization.image import ImageVisualizer, VisualizationMode
from anomalib import TaskType
from azure.storage.blob import BlobServiceClient

from utility import upload_directory_to_container

class CInference():
    def __init__(self,f_blob_str,f_cfg_obj,f_prnt_str):
        self.l_blob_str = f_blob_str
        self.l_cncn_dict = eval(f_cfg_obj['DEFAULT']['BLOB_INFO_DICT'])
        self.l_wght_info_dict = eval(f_cfg_obj['DEFAULT']['WEIGHT_INFO_DICT'])
        self.l_resources_pth_str = os.path.join(f_prnt_str, f_cfg_obj['DEFAULT']['RESOURCES_PATH'])
        self.l_mdl_dir_str = os.path.join(f_prnt_str, f_cfg_obj['DEFAULT']['MODEL_PATH'])
        self.l_inf_op_pth_str = os.path.join(f_prnt_str, f_cfg_obj['DEFAULT']['INFERENCE_PATH'])
        self.l_mdl_dwnld_dir_str = os.path.join(self.l_mdl_dir_str,self.l_blob_str)
        self.l_opnvino_pth_str = os.path.join(self.l_mdl_dir_str, self.l_blob_str,self.l_wght_info_dict["OPENVINO_MODEL_PATH"])
        print(f"self.l_opnvino_pth_str :- {self.l_opnvino_pth_str}")
        self.l_mdl_metadata_pth_str = os.path.join(self.l_mdl_dir_str, self.l_blob_str,
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

    def cleanup_folder(self,folder_path):
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
            self.download_model_f()
            self.model = OpenVINOInferencer(
                path=self.l_opnvino_pth_str,  # Path to the OpenVINO IR model.
                metadata=self.l_mdl_metadata_pth_str,  # Path to the metadata file.
                device="CPU",  # We would like to run it on an Intel CPU.
            )
            print(f"Else condition:- Model downloaded")

    def inference_f(self,f_image_obj):
        l_predictions_obj = self.model.predict(image=f_image_obj)
        visualizer = ImageVisualizer(mode=VisualizationMode.FULL, task=TaskType.SEGMENTATION)
        output_image = visualizer.visualize_image(l_predictions_obj)
        l_op_img_obj = Image.fromarray(output_image)

        # Save the PIL image in the output folder
        output_image_path = os.path.join(self.l_inf_op_pth_str, "output_image.png")
        l_op_img_obj.save(output_image_path)
        upload_directory_to_container(self.l_cncn_dict["CONNECTION_STRING"], self.l_cncn_dict["CONTAINER_NAME"], self.l_inf_op_pth_str, f"results")

        return {"label":l_predictions_obj.pred_label.name,"score":l_predictions_obj.pred_score}

    def download_model_f(self):
        """
        Download all blobs from a specific folder in an Azure Blob Storage container.

        :param None
        """
        try:
            # Create the BlobServiceClient object
            blob_service_client = BlobServiceClient.from_connection_string(self.l_cncn_dict["CONNECTION_STRING"])
            container_client = blob_service_client.get_container_client(container=self.l_cncn_dict["CONTAINER_NAME"])

            # Ensure the local directory exists
            os.makedirs(self.l_mdl_dwnld_dir_str, exist_ok=True)

            # List all blobs in the target folder
            blobs_list = container_client.list_blobs(name_starts_with=f"model/{self.l_blob_str}")
            for blob in blobs_list:
                # Construct the full local filepath
                local_file_path = os.path.join(self.l_resources_pth_str, blob.name)

                # Ensure any nested directories are created
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                # Download the blob to a local file
                blob_client = container_client.get_blob_client(blob)
                with open(local_file_path, "wb") as download_file:
                    download_file.write(blob_client.download_blob().readall())

                print(f"Downloaded {blob.name} to {local_file_path}")

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__=="__main__":
    # Path to your image
    image_path = r'C:/EDS/Current_tasks/CV_solution_ecosystem/Anomaly_detection/resources/datasets/cubes/abnormal/input_20230210134059.jpg'

    l_blob_str = r"cubes"

    # Open the image file
    with Image.open(image_path) as img:
        # Convert the image to RGB (if it's not already in RGB, e.g., PNG with transparency)
        img = img.convert('RGB')

        # Convert the PIL image to a NumPy array
        image_array = np.array(img)

    # Now, image_array is a NumPy array representing the image
    print(image_array.shape)  # Prints the dimensions of the image (height, width, channels)

    image_array = np.array(image_array)

    l_file_path_str = Path(__file__).resolve()
    l_parent_root_path_str = l_file_path_str.parents[0]

    l_config_file_path_str = os.path.join(l_parent_root_path_str, "config.cfg")
    l_config_str = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    l_config_str.read(l_config_file_path_str)

    print(f"Inference started")
    l_inf_obj = CInference(l_blob_str,l_config_str,l_parent_root_path_str)
    l_inf_obj.inference_f(image_array)
    print(f"Inference completed")

