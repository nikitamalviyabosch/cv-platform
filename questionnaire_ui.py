import os
import json
from pathlib import Path

import streamlit as st

class CProjectData:
    """Class to store the project configuration data"""
    def __init__(self):
        self.l_project_name_str = ""
        self.l_project_details_dict = {}
        self.l_path_details_dict = {}

    def write_data_to_json(self):
        """Function to write the project configurations to a json file"""
        # Getting the parent root path:- C:\CV_Platform
        l_file_path_str = Path(__file__).resolve()
        l_parent_root_path_str = l_file_path_str.parents[0]
        with open(os.path.join(l_parent_root_path_str,"ui_config_files", "proj_config_" + self.l_project_name_str + ".json"), 'w') as output_json_file:
            json.dump(self.__dict__, output_json_file)

    def save_data_to_json(self):
        """Function to save the project details in dictionary format for further processing"""

        self.l_project_name_str = st.session_state['project_name']

        # Getting the parent root path:- C:\CV_Platform
        l_file_path_str = Path(__file__).resolve()
        l_parent_root_path_str = l_file_path_str.parents[0]

        # Getting the integer to be used for OpenCV Videocapture
        l_camera_options_dict = {"Webcam":0,"Creative GestureCAM":3,"Intel Real Sense 3D camera(ZR300) RGB":4}
        
        # Project details from the form in the new prjects tab being stored in a dictionary
        self.l_project_details_dict["project_name"] = st.session_state['project_name']
        self.l_project_details_dict["use_case"] = st.session_state['use_case']
        self.l_project_details_dict["part_num"] = st.session_state['part_num']
        self.l_project_details_dict["camera_id"] = l_camera_options_dict[st.session_state['camera_id']]
        self.l_project_details_dict["comments"] = st.session_state['comments']

        # Based on the selected use case, relevant use-case specific resource folder path details will be filled in the variable:- self.l_path_details_dict   
        if st.session_state['use_case']=="VQI":
            l_ds_pth_str = os.path.join(l_parent_root_path_str,r"Anomaly_detection/files/resources/datasets/",st.session_state['project_name'])
            l_ok_ds_pth_str = os.path.join(l_ds_pth_str, "good")
            l_notok_ds_pth_str = os.path.join(l_ds_pth_str, "abnormal")
            l_mdl_pth_str = os.path.join(l_parent_root_path_str,r"Anomaly_detection/files/resources/model/")
            l_log_pth_str = os.path.join(l_parent_root_path_str,r"Anomaly_detection/files/resources/temp/")

            if not os.path.exists(l_ds_pth_str):
                os.mkdir(l_ds_pth_str)

            if not os.path.exists(l_notok_ds_pth_str):
                os.mkdir(l_notok_ds_pth_str)

            if not os.path.exists(l_ok_ds_pth_str):
                os.mkdir(l_ok_ds_pth_str)

            if not os.path.exists(l_mdl_pth_str):
                os.mkdir(l_mdl_pth_str)

            if not os.path.exists(l_log_pth_str):
                os.mkdir(l_log_pth_str)    
            
            self.l_path_details_dict= {"dataset_path":l_ds_pth_str,"ok_folder":l_ok_ds_pth_str,"notok_folder":l_notok_ds_pth_str,"model_path":l_mdl_pth_str,'log_path':l_log_pth_str}

        elif st.session_state['use_case']=="Intrusion detection":
            l_model_pth_str = os.path.join(l_parent_root_path_str,r"intrusion_detection/inference_model/yolov5s.pt")
            l_prj_pth_str = os.path.join(l_parent_root_path_str,r"intrusion_detection/resources",st.session_state['project_name'])
            l_scr_sht_pth_str = os.path.join(l_prj_pth_str,r"screenshot")
            l_op_csv_pth_str = os.path.join(l_prj_pth_str,r"output_csv")
            l_op_vid_pth_str = os.path.join(l_prj_pth_str,r"output_screenshots")
            l_coord_pth_str = os.path.join(l_prj_pth_str,r"coordinates")

            if not os.path.exists(l_prj_pth_str):
                os.mkdir(l_prj_pth_str)

            if not os.path.exists(l_scr_sht_pth_str):
                os.mkdir(l_scr_sht_pth_str)

            if not os.path.exists(l_op_csv_pth_str):
                os.mkdir(l_op_csv_pth_str)

            if not os.path.exists(l_op_vid_pth_str):
                os.mkdir(l_op_vid_pth_str)

            if not os.path.exists(l_coord_pth_str):
                os.mkdir(l_coord_pth_str)

            self.l_path_details_dict= {"project_path":l_prj_pth_str,"scrsht_path":l_scr_sht_pth_str,"op_csv_path":l_op_csv_pth_str,"l_op_vid_path":l_op_vid_pth_str,"l_coord_pth_str":l_coord_pth_str,"l_model_pth_str":l_model_pth_str}


        elif st.session_state['use_case']=="PPE":
            pass
        
        elif st.session_state['use_case']=="Assembly assistance":
            l_model_pth_str = os.path.join(l_parent_root_path_str,r"AIVA_new/inference_model/last_150.pt")
            l_prj_pth_str = os.path.join(l_parent_root_path_str,r"AIVA_new/resources",st.session_state['project_name'])
            l_op_csv_pth_str = os.path.join(l_prj_pth_str,r"output_csv")
            l_coord_pth_str = os.path.join(l_prj_pth_str,r"coordinates")

            if not os.path.exists(l_prj_pth_str):
                os.mkdir(l_prj_pth_str)

            if not os.path.exists(l_op_csv_pth_str):
                os.mkdir(l_op_csv_pth_str)

            if not os.path.exists(l_coord_pth_str):
                os.mkdir(l_coord_pth_str)

            self.l_path_details_dict= {"project_path":l_prj_pth_str,"op_csv_path":l_op_csv_pth_str,"l_coord_pth_str":l_coord_pth_str,"l_model_pth_str":l_model_pth_str}
        else:
            pass

if __name__=="__main__":
    l_file_path_str = Path(__file__).resolve()
    l_parent_root_path_str = l_file_path_str.parents[0]
    print(l_parent_root_path_str)