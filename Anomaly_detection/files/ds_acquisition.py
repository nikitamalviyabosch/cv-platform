import os
import cv2
import datetime
from pathlib import Path
import configparser
import json


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created folder: {path}")
    else:
        print(f"Folder already exists: {path}")


def main(base_folder, f_cfg_obj):
    l_part_num_str = input("Enter the part number: ")
    l_prj_metadata_pth_str = os.path.join(base_folder, f_cfg_obj['DEFAULT']['PRJ_METADATA_PATH'])
    l_full_ds_pth_str = os.path.join(base_folder, f_cfg_obj['DEFAULT']['DATASET_PATH'], l_part_num_str)

    # Create Project metadata path
    create_folder(l_prj_metadata_pth_str)
    # Create project dataset path
    create_folder(l_full_ds_pth_str)
    # Creating class folder inside the dataset folder
    abnormal_folder = os.path.join(l_full_ds_pth_str, "abnormal")
    good_folder = os.path.join(l_full_ds_pth_str, "good")
    create_folder(abnormal_folder)
    create_folder(good_folder)

    # Starting acquisition
    cap = cv2.VideoCapture(3)  # 3 for intel real sense
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

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
        cv2.putText(frame, l_part_num_str, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2, cv2.LINE_AA)

        cv2.imshow('Video Capture', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
            filename = os.path.join(abnormal_folder, f"{l_part_num_str}_abnormal_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Frame captured and saved to: {filename}")
        elif key == ord('s'):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
            filename = os.path.join(good_folder, f"{l_part_num_str}_good_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Frame captured and saved to: {filename}")
        elif key == ord('q'):
            print("Quitting...")

            # Create a dictionary with the required information
            prj_info = {
                "project_name": l_part_num_str,
                "prj_meta_data_pth": l_prj_metadata_pth_str,
                "prj_ds_pth": l_full_ds_pth_str,
                "prj_mdl_pth": ""
            }

            # Define the path for the JSON file
            json_file_path = os.path.join(l_prj_metadata_pth_str, f"{l_part_num_str}.json")

            # Save the dictionary to a JSON file
            with open(json_file_path, 'w') as json_file:
                json.dump(prj_info, json_file, indent=4)

            print(f"JSON file created at: {json_file_path}")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    l_file_path_str = Path(__file__).resolve()
    l_parent_root_path_str = l_file_path_str.parents[0]
    print(l_parent_root_path_str)
    l_config_file_path_str = os.path.join(l_parent_root_path_str, "config.cfg")
    print(f"l_config_file_path_str:-{l_config_file_path_str}")
    l_config_str = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    l_config_str.read(l_config_file_path_str)
    main(l_parent_root_path_str, l_config_str)
