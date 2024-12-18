import os
import json
from pathlib import Path

import streamlit as st
import pandas as pd

from Anomaly_detection.files.streamlit_inference import main
from PPE_Detection_v3_Kamesh.PPE_Detection.Main_streamit import main_inf_ppe_f
from intrusion_detection.main import main_intrusion_det_f
from AIVA_new.video_tracking import video_tracking_page
from utility import show_logs_f

def start_inspection_f():
    # Getting the parent root path:- C:\CV_Platform
    l_file_path_str = Path(__file__).resolve()
    l_parent_root_path_str = l_file_path_str.parents[0]

    # Getting the ui config folder path:- C:\CV_Platform
    ui_config_path = os.path.join(l_parent_root_path_str,"ui_config_files")

    # Getting the list of config files present
    l_json_files_list = [pos_json for pos_json in os.listdir(ui_config_path) if pos_json.endswith('.json')]


    for files in l_json_files_list:
        if "proj_config" in files:
            json_file = open(os.path.join(ui_config_path, files), "r+")
            l_json_text = json.load(json_file)
            # read project configuration file which saved from new project
            l_project_name_str = l_json_text['l_project_name_str']
            l_project_details_dict = l_json_text['l_project_details_dict']
            l_path_details_dict = l_json_text['l_path_details_dict']
            l_use_case_str = l_project_details_dict['use_case']

            # list the projects with expander option
            with st.expander(f"{l_project_name_str}"):
                l_menu_list = ["","Live inference","Show logs"]

                if l_menu_list is not None:
                    l_choice_str = st.selectbox("Kindly select the Task", l_menu_list, key=f"{l_project_name_str}_{l_project_details_dict['part_num']}_start_inspection")

                    if l_choice_str == "Live inference":
                        # st.text("Triggered")
                        # if st.button("Start inference",key=f"{l_project_name_str}_{l_project_details_dict['part_num']}_start_prediction"):
                        if l_use_case_str == 'VQI':
                            main(l_project_details_dict['part_num'],l_project_name_str)
                        if l_use_case_str == 'PPE':
                            main_inf_ppe_f(f"{l_project_name_str}_{l_use_case_str}_inf_stream")

                        if l_use_case_str == "Intrusion detection":
                            l_op_csv_path_str = l_path_details_dict["op_csv_path"]
                            l_coord_pth_str = l_path_details_dict["l_coord_pth_str"]
                            l_mdl_pth_str = l_path_details_dict["l_model_pth_str"]
                            l_op_scr_path =  l_path_details_dict["l_op_vid_path"]
                            main_intrusion_det_f(l_coord_pth_str,l_op_csv_path_str,l_mdl_pth_str,l_op_scr_path)
                        
                        if l_use_case_str=="Assembly assistance":
                            l_json_pth_str = l_path_details_dict["l_coord_pth_str"]
                            l_op_log_csv_pth_str= l_path_details_dict["op_csv_path"]
                            video_tracking_page(l_project_details_dict['part_num'],l_json_pth_str,l_op_log_csv_pth_str)

                    if l_choice_str == "Show logs":
                        if l_use_case_str == 'VQI':
                            show_logs_f(l_path_details_dict['log_path'])
                        if l_use_case_str == "Intrusion detection":
                            show_logs_f(l_path_details_dict["l_op_vid_path"])

                        if l_use_case_str=="Assembly assistance":
                            l_logs_df = pd.read_csv(os.path.join(l_path_details_dict["op_csv_path"],"output_log.csv"))
                            st.dataframe(l_logs_df, use_container_width=True)
