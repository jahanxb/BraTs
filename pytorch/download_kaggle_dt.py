import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi


def download_datasets(dataset_list, destination_path):
    api = KaggleApi()
    api.authenticate()  # Make sure the API is authenticated

    for dataset_slug in dataset_list:
        dataset_dir = os.path.join(destination_path, dataset_slug)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

            # Download the dataset files
            api.dataset_download_files(dataset_slug, path=dataset_dir, unzip=True)

            zip_file_path = os.path.join(dataset_dir, f"{dataset_slug}.zip")
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            print(f"Dataset '{dataset_slug}' downloaded successfully.")
        else:
            print(f"Dataset '{dataset_slug}' already exists.")

# Define the dataset list and destination path
dataset_list = [
    # 'awsaf49/brats2020-training-data',
    # 'awsaf49/brats20-dataset-training-validation',
    'polomarco/brats20logs'
]

destination_path = "/mydata/datasets/"

# Call the function to download the datasets
if __name__ == '__main__':
	download_datasets(dataset_list, destination_path)



