# text-based-denoising
This project implements diffusion based transformer model to denoise the images with text prompt guidance.

## Dataset Information
* **New Dataset Link:** [Download from Google Drive](https://drive.google.com/file/d/1fhG-FbfGXDncv_e1SX6Cepfl51f3sPxv/view?usp=drive_link)
* **Description:** The dataset is reconstructed in a better structure for improved performance.

## Features (DatasetManager)
The `DatasetManager` class includes 2 core functions:
1.  `scan_database`: Checks and summarizes the full dataset. Output saved to `dataset_metadata`.
2.  `split_and_save_json`: Splits dataset in defined ratio (train, valid, test) and saves to JSON files.

## How to Use
1.  Download the dataset, unzip it, and place it in the **root** directory.
2.  Run `DatasetManager.py` to generate documentation files.
3.  Check the `dataset_metadata/` folder for:
    * `dataset_info` and `dataset_structure`.
    * `train.json`, `valid.json`, and `test.json` for the `DataLoader`.