import os
import shutil
from pathlib import Path
import configparser

from anomalib.models import Padim
from anomalib.engine import Engine
from anomalib.utils.normalization import NormalizationMethod
from anomalib import TaskType
from anomalib.deploy import ExportType

from Anomaly_detection.files.utility import upload_directory_to_container

class CTrain():
    def __init__(self,f_dl_obj,f_prj_str,f_cfg_obj,f_prnt_str):
        self.l_fld_dm_obj = f_dl_obj
        self.l_prj_str = f_prj_str
        # self.l_cncn_dict = eval(f_cfg_obj['DEFAULT']['BLOB_INFO_DICT'])
        self.l_mdl_dir_str = os.path.join(f_prnt_str,f_cfg_obj['DEFAULT']['MODEL_PATH'])
        self.model = None
        self.engine = None

        # if not os.path.exists(os.path.join(self.l_mdl_dir_str, self.l_prj_str)):
        #     os.makedirs(os.path.join(self.l_mdl_dir_str, self.l_prj_str))

        if os.path.isdir(os.path.join(self.l_mdl_dir_str, self.l_prj_str)):
            self.cleanup_folder()

        if not self.model:
            self.model_f()

        if not self.engine:
            self.create_engine_f()

    def model_f(self):
        """
        Create a PaDim anomaly model instance for training
        :param None
        """
        self.model = Padim(
            backbone="resnet18",
            layers=["layer1", "layer2", "layer3"],
        )

    def create_engine_f(self):
        """
        Create a anomalib engine class object with the necessary parameters
        :param None
        """
        self.engine = Engine(
            normalization=NormalizationMethod.MIN_MAX,
            threshold="F1AdaptiveThreshold",
            task=TaskType.CLASSIFICATION,
            image_metrics=["AUROC"],
            accelerator="auto",
            check_val_every_n_epoch=1,
            devices=1,
            max_epochs=1,
            num_sanity_val_steps=0,
            val_check_interval=1.0,
        )
    def train_f(self):
        """
        Create a anomalib engine class object with the necessary parameters
        :param None
        """
        self.engine.fit(model=self.model, datamodule=self.l_fld_dm_obj)

    def test_f(self):
        """
        Create a anomalib engine class object with the necessary parameters
        :param Dict: A dictionary containing the test results
        """

        test_results = self.engine.test(model=self.model, datamodule=self.l_fld_dm_obj)
        print(f"test_results :- {test_results}")

    def save_model_f(self):
        """
        Saving the trained model for the dataset in Azure in ONNX format
        :param None
        """
        try:
            l_path_str = os.path.join(self.l_mdl_dir_str, self.l_prj_str)
            # Ensure the local directory exists
            os.makedirs(l_path_str, exist_ok=True)
            l_mdl_save_dict = {}
            if self.engine:
                # Exporting model to OpenVINO
                openvino_model_path = self.engine.export(
                    model=self.model,
                    export_type=ExportType.OPENVINO,
                    export_root=str(l_path_str),
                )
                print(f"Model trained and saved successfully")
                # l_mdl_save_dict["parent_path"]=str(openvino_model_path.parent)
                # upload_directory_to_container(self.l_cncn_dict['CONNECTION_STRING'], self.l_cncn_dict['CONTAINER_NAME'], l_path_str,f"model/{self.l_prj_str}")
            else:
                print(f"Please train the model first !")
            # return l_mdl_save_dict
        except Exception as e:
            print(f"An error occurred: {e}")

    def cleanup_folder(self):
        """
        Deletes a specified folder and all its contents.

        :param None
        """
        # Construct the full path to the folder
        folder_path = os.path.join(self.l_mdl_dir_str, self.l_prj_str)

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


if __name__=="__main__":
    l_file_path_str = Path(__file__).resolve()
    l_parent_root_path_str = l_file_path_str.parents[0]

    l_config_file_path_str = os.path.join(l_parent_root_path_str, "config.cfg")
    l_config_str = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    l_config_str.read(l_config_file_path_str)


    l_data_obj = CDataSetup(l_data_cls_str, l_config_str, l_parent_root_path_str)
    l_fld_dm_obj = l_data_obj.create_dataset_f()
    l_train_obj = CTrain(l_fld_dm_obj, l_data_cls_str, l_config_str, l_parent_root_path_str)
    l_train_obj.train_f()


