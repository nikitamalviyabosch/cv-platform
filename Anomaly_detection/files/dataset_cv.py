import os
import configparser
import shutil

from pathlib import Path

from anomalib.data.image.folder import Folder
from anomalib import TaskType
# from azure.storage.blob import BlobServiceClient

class CDataSetup():
    def __init__(self, f_prj_str, f_cfg_dict, f_prnt_str):
        self.f_prj_str = f_prj_str
        # self.l_cncn_dict = eval(f_cfg_dict['DEFAULT']['BLOB_INFO_DICT'])
        self.l_ds_dir_str = os.path.join(f_prnt_str,f_cfg_dict['DEFAULT']['DATASET_PATH'])

        self.l_ds_exist_bool =  False

        if os.path.isdir(os.path.join(self.l_ds_dir_str, f_prj_str)):
            self.l_ds_exist_bool =  True
            # self.cleanup_folder()
        else:
            self.l_ds_exist_bool = False

        # self.download_all_blobs_in_folder()

    def create_dataset_f(self):
        """
        Function to create a dataloader for training.

        :param None
        """
        l_dfct_ds_str = os.path.join(self.l_ds_dir_str,self.f_prj_str)
        l_abn_dir_list = [l_elt_str for l_elt_str in os.listdir(l_dfct_ds_str) if
                          l_elt_str != "normal" and l_elt_str != "mask" and l_elt_str != "good" and l_elt_str != "license.txt" and l_elt_str != "README.txt"]
        l_good_dir_list = [l_elt_str for l_elt_str in os.listdir(l_dfct_ds_str) if l_elt_str == "good" or l_elt_str == "normal"]

        if l_abn_dir_list ==[]:
            l_abn_dir_list=None

        l_fld_dm_obj = Folder(
            name=self.f_prj_str,
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

