import os
import configparser
import shutil

from pathlib import Path

from anomalib.data.image.folder import Folder
from anomalib import TaskType
from azure.storage.blob import BlobServiceClient

class CDataSetup():
    def __init__(self,f_blob_str,f_cfg_dict,f_prnt_str):
        self.l_blob_str = f_blob_str
        self.l_cncn_dict = eval(f_cfg_dict['DEFAULT']['BLOB_INFO_DICT'])
        self.l_ds_dir_str = os.path.join(f_prnt_str,f_cfg_dict['DEFAULT']['DATASET_PATH'])

        if os.path.isdir(os.path.join(self.l_ds_dir_str,self.l_blob_str)):
            self.cleanup_folder()

        self.download_all_blobs_in_folder()

    def download_all_blobs_in_folder(self):
        """
        Download all blobs from a specific folder in an Azure Blob Storage container.

        :param None
        """
        try:
            # Create the BlobServiceClient object
            blob_service_client = BlobServiceClient.from_connection_string(self.l_cncn_dict["CONNECTION_STRING"])
            container_client = blob_service_client.get_container_client(container=self.l_cncn_dict["CONTAINER_NAME"])

            # Ensure the local directory exists
            os.makedirs(self.l_ds_dir_str, exist_ok=True)

            # List all blobs in the target folder
            blobs_list = container_client.list_blobs(name_starts_with=self.l_blob_str)
            for blob in blobs_list:
                # Construct the full local filepath
                local_file_path = os.path.join(self.l_ds_dir_str, blob.name)

                # Ensure any nested directories are created
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                # Download the blob to a local file
                blob_client = container_client.get_blob_client(blob)
                with open(local_file_path, "wb") as download_file:
                    download_file.write(blob_client.download_blob().readall())

                print(f"Downloaded {blob.name} to {local_file_path}")

        except Exception as e:
            print(f"An error occurred: {e}")

    def cleanup_folder(self):
        """
        Deletes a specified folder and all its contents.

        :param None
        """
        # Construct the full path to the folder
        folder_path = os.path.join(self.l_ds_dir_str, self.l_blob_str)

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

    def create_dataset_f(self):
        """
        Function to create a dataloader for training.

        :param None
        """
        l_dfct_ds_str = os.path.join(self.l_ds_dir_str,self.l_blob_str)
        l_abn_dir_list = [l_elt_str for l_elt_str in os.listdir(l_dfct_ds_str) if
                          l_elt_str != "normal" and l_elt_str != "mask" and l_elt_str != "good" and l_elt_str != "license.txt" and l_elt_str != "README.txt"]
        l_good_dir_list = [l_elt_str for l_elt_str in os.listdir(l_dfct_ds_str) if l_elt_str == "good" or l_elt_str == "normal"]

        if l_abn_dir_list ==[]:
            l_abn_dir_list=None

        l_fld_dm_obj = Folder(
            name=self.l_blob_str,
            root=l_dfct_ds_str,
            normal_dir=l_good_dir_list,
            abnormal_dir=l_abn_dir_list,
            image_size=(256, 256),
            train_batch_size=32,
            eval_batch_size=32,
            task=TaskType.CLASSIFICATION,
        )
        l_fld_dm_obj.setup()
        return l_fld_dm_obj

if __name__=="__main__":
    l_file_path_str = Path(__file__).resolve()
    l_parent_root_path_str = l_file_path_str.parents[0]

    l_config_file_path_str = os.path.join(l_parent_root_path_str, "config.cfg")
    l_config_str = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    l_config_str.read(l_config_file_path_str)

    l_data_cls_str = "cubes"

    l_data_obj = CDataSetup(l_data_cls_str,l_config_str,l_parent_root_path_str)
    l_fld_dm_obj = l_data_obj.create_dataset_f()
    i, data = next(enumerate(l_fld_dm_obj.val_dataloader()))
    print(data.keys())

