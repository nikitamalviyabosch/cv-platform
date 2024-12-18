import os
import streamlit as st

from questionnaire_ui import CProjectData

def new_project_f():
    l_placeholder_obj = st.empty()
    l_camera_options_list = ["Webcam","Creative GestureCAM","Intel Real Sense 3D camera(ZR300) RGB"]
    l_proj_obj = CProjectData()
    
    with l_placeholder_obj.form(key="Form_1", clear_on_submit=True):
        st.text_input("Enter the project name:", "default", key='project_name')
        st.radio("Select the use case:", ("Intrusion detection", "VQI", "PPE", "Assembly assistance"), key="use_case")

        st.text_input("Enter the part number(Only for VQI and assembly assistance):", "default", key='part_num')

        st.selectbox("Select a Camera", l_camera_options_list,key='camera_id')

        # Text area for comments
        st.text_area("Comments",key='comments')

        if st.form_submit_button("Save"):
            # save the project configuration
            l_proj_obj.save_data_to_json()
            l_proj_obj.write_data_to_json()
            l_placeholder_obj.empty()
            with l_placeholder_obj:
                st.success("Data Saved! Please refresh")
        else:
            st.stop()