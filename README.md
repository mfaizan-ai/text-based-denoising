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

## Dataset Structure 
The image dataset is organzied as follows: 
```
🟦 dataset_full
├── 🟩 blur
│   └── 🟢 real
│       ├── ⚪ blended
│       └── ⚪ clean
├── 🟩 raindrop
│   ├── 🟢 real
│   │   ├── ⚪ blended
│   │   └── ⚪ clean
│   └── 🟢 syn_zka
│       ├── ⚪ blended
│       └── ⚪ clean
├── 🟩 rainstreak
│   ├── 🟢 real
│   │   ├── ⚪ blended
│   │   └── ⚪ clean
│   └── 🟢 syn_zka
│       ├── ⚪ blended
│       └── ⚪ clean
├── 🟩 rainstreak_raindrop
│   └── 🟢 real
│       ├── ⚪ blended
│       └── ⚪ clean
└── 🟩 reflection
    ├── 🟢 syn
    │   ├── ⚪ blended
    │   └── ⚪ clean
    └── 🟢 syn_zka
        ├── ⚪ blended
        └── ⚪ clean
```
These images are denoised by DIT, so which is text guided network that takes text embeddings as input, so we create text embeddings from the texttual data and store it, these embeddings are generated
for each task, for instance we have text prompts for blur taks that look like the following:
```
    "De-blur this image to restore sharp details.",
    "Remove the motion blur from the photo.",
    "Sharpen the blurry areas and recover clarity.",
    "Fix the out-of-focus blur in this picture.",
    "Clean the hazy and blurred regions.",
    "Make the blurry image crisp and clear.",
    "Recover the lost details from the blurred scene.",
    "Eliminate the camera shake blur.",
    "Restore the sharp edges of the blurred objects.",
    "Please deblur the foreground and background.",
    "Clear the motion artifacts from the image.",
    "Perform high-quality deblurring on this scene.",
    "Refocus this blurry photograph.",
    "Improve the resolution by removing blur.",
    "Sharp image restoration from blurred input.",
    "Remove defocus blur for a clearer view.",
    "Enhance the clarity by fixing the blur.",
    "Deblur the scene while maintaining natural textures.",
    "Get rid of the blurriness in this frame.",
    "Turn this blurry shot into a sharp one."
```
And the embeddings are stored before we run the training:

```text/text_embeddings/
├── blur
│   ├── 0.pt
│   ├── 1.pt
│   ├── 2.pt
│   └── ...
├── raindrop
│   ├── 0.pt
│   ├── 1.pt
│   ├── 2.pt
│   └── ...
├── rainstreak
│   ├── 0.pt
│   ├── 1.pt
│   ├── 2.pt
│   └── ...
├── rainstreak_raindrop
│   ├── 0.pt
│   ├── 1.pt
│   ├── 2.pt
│   └── ...
└── reflection
    ├── 0.pt
    ├── 1.pt
    ├── 2.pt
    └── ...
```

## Installation 
Anconda environment recommended.
```
git clone https://github.com/mfaizan-ai/text-based-denoising
cd text-based-denoising
```

create a virtual environment in Anaconda and activate it.
```
conda create -n windowseat python=3.12.11 -y 
conda activate windowseat
```
Now install all the dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```