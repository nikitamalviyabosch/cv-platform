import configparser
import os
from pathlib import Path

# from dataset_cv import CDataSetup
from Anomaly_detection.files.dataset_cv import CDataSetup
from Anomaly_detection.files.train_cv import CTrain

def train_f(f_prj_str):
    l_file_path_str = Path(__file__).resolve()
    l_parent_root_path_str = l_file_path_str.parents[0]

    l_config_file_path_str = os.path.join(l_parent_root_path_str, "config.cfg")
    l_config_str = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    l_config_str.read(l_config_file_path_str)

    l_data_obj = CDataSetup(f_prj_str, l_config_str, l_parent_root_path_str)
    l_fld_dm_obj = l_data_obj.create_dataset_f()
    l_train_obj = CTrain(l_fld_dm_obj, f_prj_str, l_config_str, l_parent_root_path_str)
    l_train_obj.train_f()
    l_train_obj.test_f()
    l_train_obj.save_model_f()

if __name__=="__main__":
    train_f(f"op")