# from azure.storage.blob import BlobServiceClient
import os

def upload_directory_to_container(connection_string, container_name, directory_path, target_blob_path=""):
    """
    Uploads the contents of a local directory to an Azure Blob Storage container.

    :param connection_string: Connection string to the Azure Storage account
    :param container_name: Name of the Azure Blob Storage container
    :param directory_path: Path to the local directory to upload
    :param target_blob_path: Path within the container where the directory should be uploaded
    """
    try:
        # Create the BlobServiceClient object
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)

        # Walk through the local directory
        for root, dirs, files in os.walk(directory_path):
            for filename in files:
                # Construct the full local path
                file_path = os.path.join(root, filename)

                # Construct the full path for the blob
                if target_blob_path:
                    blob_path = os.path.join(target_blob_path, os.path.relpath(file_path, directory_path)).replace("\\", "/")
                else:
                    blob_path = os.path.relpath(file_path, directory_path).replace("\\", "/")

                print(f"Uploading {file_path} to {blob_path}...")
                # Create a blob client and upload the file
                blob_client = container_client.get_blob_client(blob_path)
                with open(file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)

        print("Upload completed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")

def download_model_f():
    """
    Download all blobs from a specific folder in an Azure Blob Storage container.

    :param None
    """
    try:
        # Create the BlobServiceClient object
        blob_service_client = BlobServiceClient.from_connection_string(r"DefaultEndpointsProtocol=https;AccountName=cvsolutionplatform;AccountKey=p1sbaxx3rFzM9SPNrEvNblOiapgQlYiayKqudfqS1ePP6oOZgbum1QpIaGgPtQTnjMwq1IRp//lU+AStBucukw==;EndpointSuffix=core.windows.net")
        container_client = blob_service_client.get_container_client(container="anomalydetection")

        # Ensure the local directory exists
        os.makedirs(r"C:\EDS\Current_tasks\CV_solution_ecosystem\Anomaly_detection\resources\model\cubes", exist_ok=True)

        # List all blobs in the target folder
        blobs_list = container_client.list_blobs(name_starts_with="model/cubes")
        for blob in blobs_list:
            # Construct the full local filepath
            local_file_path = os.path.join(r"C:\EDS\Current_tasks\CV_solution_ecosystem\Anomaly_detection\resources", blob.name)

            # Ensure any nested directories are created
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Download the blob to a local file
            blob_client = container_client.get_blob_client(blob)
            with open(local_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())

            print(f"Downloaded {blob.name} to {local_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__=="__main__":
    # Usage example
    # connection_string = "DefaultEndpointsProtocol=https;AccountName=cvsolutionplatform;AccountKey=p1sbaxx3rFzM9SPNrEvNblOiapgQlYiayKqudfqS1ePP6oOZgbum1QpIaGgPtQTnjMwq1IRp//lU+AStBucukw==;EndpointSuffix=core.windows.net"
    # container_name = "anomalydetection"
    # local_directory_path = "C:/EDS/Current_tasks/CV_solution_ecosystem/dataset/cubes"
    # # Optional: specify a folder path within the container where files should be uploaded
    # # target_container_folder = "target_folder_within_container"
    # f_cls_str = "cubes"
    # upload_directory_to_container(connection_string, container_name, local_directory_path, f"models/{f_cls_str}")
    download_model_f()
