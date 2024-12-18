import configparser
import os
import base64
from pathlib import Path
from io import BytesIO
from PIL import Image
import logging
from flask import Flask, jsonify,request
import numpy as np

from train import CTrain
from dataset import CDataSetup
from offline_inference import CInference

app = Flask(__name__)

FORMAT = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
# Use filename="file.log" as a param to logging to log to a file
logging.basicConfig(filename="file.log",format=FORMAT, level=logging.INFO)

@app.route('/v1/run-engine', methods=['POST'])
def run_engine():
    l_file_path_str = Path(__file__).resolve()
    l_parent_root_path_str = l_file_path_str.parents[0]

    l_config_file_path_str = os.path.join(l_parent_root_path_str, "config.cfg")
    l_config_str = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    l_config_str.read(l_config_file_path_str)

    l_data_cls_str = request.json.get('class')

    l_training_meta_data_dict = {}
    l_training_meta_data_dict["Data_class"] = l_data_cls_str

    l_data_obj = CDataSetup(l_data_cls_str, l_config_str, l_parent_root_path_str)
    l_fld_dm_obj = l_data_obj.create_dataset_f()
    l_train_obj = CTrain(l_fld_dm_obj, l_data_cls_str, l_config_str, l_parent_root_path_str)
    l_train_obj.train_f()
    l_training_meta_data_dict["results"] = l_train_obj.test_f()
    l_training_meta_data_dict["model_path"] = l_train_obj.save_model_f()

    # Instead of printing, return the dictionary as a JSON response
    return jsonify(l_training_meta_data_dict)

@app.route('/v1/anomaly-infer', methods=['POST'])
def anomaly_inference():
    logging.info('Received a request for inference.')
    image_file = request.json.get('image')
    l_blob_str = request.json.get('class')

    # Decode the base64 image
    image_data = base64.b64decode(image_file)

    try:
        # Open the image file using PIL
        logging.info('Image converted from base64 to PIL format.')
        image = Image.open(BytesIO(image_data)).convert('RGB')

        image_array = np.array(image)

        l_file_path_str = Path(__file__).resolve()
        l_parent_root_path_str = l_file_path_str.parents[0]

        l_config_file_path_str = os.path.join(l_parent_root_path_str, "config.cfg")
        l_config_str = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        l_config_str.read(l_config_file_path_str)

        print(f"Inference started")
        l_inf_obj = CInference(l_blob_str, l_config_str, l_parent_root_path_str)
        l_inf_results_dict = l_inf_obj.inference_f(image_array)
        print(f"Inference completed")
        print(f"Inference ended")
        # Return the inference result as a JSON response
        return jsonify(l_inf_results_dict)
    except Exception as e:
        logging.error(f'Error processing image: {str(e)}')
        response = {
            'message': 'Error occurred during inference.',
            'error': str(e)
        }
        logging.info(f'Error occurred during inference :-{str(e)}')
        return jsonify(response), 500

if __name__=="__main__":
    app.run(host="0.0.0.0",port=3002)
    # app.run()