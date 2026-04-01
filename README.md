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
## Usage
To train the model, subject a slurm job on the HPC or submit to GPU. Make sure to update paths to train the training successufly. 
```
sbatch run_training.sh
```

similarly the test can be run with slurm
```
sbatch run_test.sh
```

And also a test script can be used on terminal:
```
python test.py # default

python test.py \
      --checkpoint  runs/baseline/checkpoint_best.pt \
      --data-root   dataset \
      --meta-dir    dataset_metadata \
      --embed-dir   text/text_embeddings \
      --output-dir  runs/baseline
 
```
this will load the check point and performance inference on the test set, the model return the metrics on the test set 
and the inference over a few images from each category to see how model denoise those categories. 


## Running demo with Gradio
To launch gradio to run the demo for restoration
```
# Terminal 1 — on laptop, SSH to cluster then gpu01
ssh maguire01

cd /lustre/disk/home/users/mfaizan/windowseat-reflection-removal/text-based-denoising

source ../../bash_script/slurm_jobs_submission.sh

srun --jobid=$SLURM_JOBID --pty bash

source ../../bash_scripts/setup_everything.sh

cd /lustre/disk/home/users/mfaizan/windowseat-reflection-removal/text-based-denoising

conda activate windowseat

python app.py \
    --checkpoint runs/baseline_full/checkpoint_best.pt \
    --embed-dir  text/text_embeddings \
    --host 0.0.0.0 \
    --port 7862

# Terminal 2 — fresh terminal on laptop (not SSH'd)
ssh -L 7862:gpu01:7862 maguire01

# Browser
http://localhost:7862
```
![alt text](https://github.com/mfaizan-ai/text-based-denoising/blob/main/inference_images/gradio_demo.png)