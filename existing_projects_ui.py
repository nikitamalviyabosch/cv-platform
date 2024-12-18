import os
import json
from pathlib import Path

import streamlit as st

from utility import menu_list_f,data_acquisition_f,data_labeling_f,vqi_training_f,vqi_ds_acq_label_f
from AIVA_new.roi_capture import roi_capture_page

def existing_projects_f():

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
            
            l_usecase_str = l_project_details_dict["use_case"]
            # list the projects with expander option
            with st.expander(f"{l_project_name_str}"):
                l_menu_list = menu_list_f(l_project_details_dict["use_case"])

                if l_menu_list is not None:
                    l_choice_str = st.selectbox("Kindly select the Task", l_menu_list, key=l_project_name_str)

                    if l_choice_str == "Data acquisition & labeling":
                        if l_usecase_str == "VQI":
                            vqi_ds_acq_label_f(l_path_details_dict['notok_folder'],l_path_details_dict['ok_folder'],l_project_name_str)
                        elif l_usecase_str=="Assembly assistance":
                            roi_capture_page(l_path_details_dict["l_coord_pth_str"])

                    if l_choice_str == "Data acquisition":
                        if l_usecase_str == "Intrusion detection":
                            data_acquisition_f(l_path_details_dict['scrsht_path'],l_project_name_str)

                    if l_choice_str == "Data Labeling":
                        if l_usecase_str == "Intrusion detection":
                            data_labeling_f(l_path_details_dict['l_coord_pth_str'],l_path_details_dict['scrsht_path'],l_project_name_str)

                    if l_choice_str == "Training":
                        with st.spinner("Model training has started..."):
                            vqi_training_f(l_project_name_str)
                        st.success("Training has completed !")

                else:
                    st.text(f"Please proceed to the start inspection page !")

                
