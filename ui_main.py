import os
import json
import numpy as np
import configparser
from pathlib import Path

import streamlit as st
from streamlit_option_menu import option_menu

# Questionnaire imports
from new_project_ui import new_project_f
from existing_projects_ui import existing_projects_f
from start_inspection_ui import start_inspection_f


def ui_main_f():
    l_page_title_str = "VisionSmart.ai"
    l_page_icon_str = ":mag:"
    l_layout_str = "wide"
    st.set_page_config(page_title=l_page_title_str, page_icon=l_page_icon_str, layout=l_layout_str,initial_sidebar_state="collapsed")
    st.sidebar.empty()
    with open(r"C:\CV_Platform\style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


    # st.title("VisionSmart.ai")
    st.markdown('''<h1 class="custom-h1">VisionSmart.ai</h1>''',unsafe_allow_html=True)

    # --- HIDE STREAMLIT STYLE ---
    hide_st_style = """
                                <style>
                                MainMenu {visibility: hidden;}
                                footer {visibility: hidden;}
                                header {visibility: hidden;}
                                </style>
                                """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # Logo Code
    l_logo_html=""" <div class="logo-container">
                    <img src="http://localhost:5000/image1" >
                    </div>
        """
    st.markdown(l_logo_html,unsafe_allow_html=True)
    #st.image(r"C:\Users\ESA3KOR\Pictures\Screenpresso\2024-08-29_15h30_57.png")

    # Ribbio code
    #st.image(r"C:\Users\ESA3KOR\Downloads\ribbon.png")
    ribbion_html = f"""<div class="ribbon-container">
    <img src="http://localhost:5000/image2">
    </div>"""
    st.markdown(ribbion_html, unsafe_allow_html=True)


    # --- NAVIGATION MENU ---
    selected = option_menu(menu_title=None, options=["New Project", "Existing Projects", "Start Inspection"],
                           icons=["file-plus", "gear", "play"], orientation="horizontal", default_index=0, styles={"nav-link-selected": {"background-color": "#133e77"}})
    
    if selected == "New Project":
        new_project_f()
    elif selected == "Existing Projects":
        existing_projects_f()
    elif selected == "Start Inspection":
        start_inspection_f()

if __name__=="__main__":
    ui_main_f()
