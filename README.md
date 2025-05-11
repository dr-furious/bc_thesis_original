# Histological Image Data Processing Using Methods of Computer Vision and Deep Neural Networks

**Bachelor's Thesis** \
**Thesis ID:** FIIT-100241-116291 \
**Author:** Martin Sivák \
**Supervisor:** Prof. Ing. Vanda Benešová, PhD.

**Study program:** Informatics \
**Field of study:** Computer Science \
**Place of elaboration:** Institute of Computer Engineering and Applied Informatics, FIIT STU, Bratislava

## Table of contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project's folder structure
```text
.
├── configs/
│   ├── azure_connect_example.json
│   ├── azure_job.yaml
│   ├── azure_upload_data.yaml
│   ├── model_train_base.yaml
│   └── paths.yaml
├── data/
│   ├── processed_tiger/
│   ├── processed_tnbc/
│   ├── raw_tiger/
│   │   ├── images/
│   │   └── coco_annotations_placeholder.json
│   └── raw_tnbc/
│       ├── images/
│       └── masks/
├── example_data/
│   ├── processed_tiger/
│   ├── processed_tnbc/
│   ├── raw_tiger/
│   │   ├── images/
│   │   └── tiger-coco.json
│   └── raw_tnbc/
│       ├── images/
│       └── masks/
├── models/
├── src/
│   ├── azure/
│   │   ├── azure_conda.yaml
│   │   ├── azure_train.ipynb
│   │   └── azure_upload_data.ipynb
│   ├── models/
│   │   ├── inference.ipynb
│   │   ├── model_factory.py
│   │   └── til_dataset.py
│   ├── image_preprocessor.py
│   ├── image_stats.py
│   ├── mask_generator.py
│   ├── sample.py
│   └── utils.py
├── .amlignore
├── .gitignore
├── main.ipynb
├── main.py
├── README.md
└── requirements.txt
```

## Descriptions of folders and files
### `root` directory files
**.amlignore**\
Here is a list of folders and files that are ignored when a job is submitted to the Azure ML platform. When a job is submitted, Azure takes a snapshot of the directory it is give as the source directory. The files listed in `.amlignore` will be ignored by this operation.

**.gitgnore**\
Files to be ignored by the Git versioning system.

**main.ipynb**\
In this Jupyter notebook, the whole preprocessing, pseudo-mask generating, and pseudo-mask fusing pipeline can be run. It also provides the visualizations of preprocessed images and pseudo-masks. By default, this notebook is run, so you can also see the outputs of each cell. Open the [main.ipynb](./main.ipynb) to see it.

**main.py**\
This is the main training and evaluation script. It can be run both locally and on the Azure ML platform. See the [How to run]() section to see both possible options.

**README.md**\
This is the document you are currently reading.

**requirements.txt**\
Contains Python dependencies that need to be installed in order to run the project.

### `config` directory files

**azure_connect_example.json**\
Contains the information required to authenticate and connect to your Azure ML Workspace. You will need to fill out this config, otherwise the connection will not be successful. Keep the structure, just change the values to match your account.

**azure_job.yaml**\
Contains all configuration values that are used to submit the job run, which will train the model. These include the data asset information, the environment information (environment where the job will run), the job information (like source directory to push to Azure ML, compute target, etc.) and the arguments to be passed to the `main.py` function once it is executed.

**azure_upload_data.yaml**\
Here are the information about the folder you want to upload to the Azure storage, the destination folder on Azure ML and options to overwrite already existing files and see the progress of the whole process.

**model_train_base.yaml**\
Contains the hyperparameters that are used by the model during the training and evaluation. It also contains the option to load the pre-trained model.

**paths.yaml**\
This file contains all paths, or parts of paths where the images are being stored, created, modified and updated, and from which are loaded during the preprocessing. The whole folder structure for the preprocessing and pseudo-mask creation is created in the [main.ipynb](./main.ipynb) notebook, here also the full paths are build. Note that by default, all file manipulations are performed under the `./example_data/` directory (unless changed in this config). For the [Demo](), we recommend keeping this config file as is. For the real preprocessing, we advise to change the `root_data_dir` value to point to the `./data` directory.

### `data` and `example_data` directories
The `data` directory contains four main subdirectories. Here the images and annotations of the respective datasets reside ([TIGER]() and [TNBC]() datasets). The `./data/raw_*` folders contain the raw images and annotations (bounding box for TIGER - in the [COCO JSON]() format, PNG masks for TNBC). The `./data/preprocessed_*` directories contain more subdirectories that are created during the run of the [main.ipynb](./main.ipynb) notebook. The most important ones are:

- `./data/preprocessed_*/patches/images` which contains the 128x128 normalized image patches
- `./data/preprocessed_*/patches/masks` which contains the 128x128 mask (or pseudo-mask) patches
- `./data/preprocessed_tnbc/patches/folds` which contains TNBC image and mask patches but split into folds, where each fold directory has `./data/preprocessed_tnbc/patches/folds/fold_*/images` and  `./data/preprocessed_tnbc/patches/folds/fold_*/masks` folder

Note that this directory is meant to be used for real preprocessing, and you need to put here the correct images and annotations yourself. 

The `./example_data` directory follows the exact same structure, but already contains 10 example images from the TIGER dataset in the `./data/raw_tiger/images` subdirectory,  the `tiger-coco.json` file with the TIGER bounding box annotations in the  `./data/raw_tiger` subdirectory, and 4 images from the TNBC dataset in the `./data/raw_tnbc/images` subdirectory and their corresponding masks in the `./data/raw_tnbc/masks` subdirectory. This directory is by default listed as the `root_data_dir` in the `./configs/path.yaml` file, so in order to run the [Demo]() you do not need to change anything in the `./configs/path.yaml` file.

### `models` directory
Here the models that you wish to save and use for future fine-tuning or reference should be place. We do not include any pre-trained model here, since the `.ckpt` files are around 300MB in size.

### `src/azure` directory
**azure_conda.yaml**\
Defines the dependencies that will be installed within Azure ML environment.

**azure_train.ipynb**\
From this Jupyter notebook the training is managed. This involves authenticating, pulling the correct data asset path, creating the environment and submitting the job to Azure ML. Use the `./configs/azure_connect_example.json`, `./configs/azure_job.yaml` and `./configs/model_train_base.yaml` to manage the configuration of parameters.

**azure_upload_data.ipynb**\
This Jupyter notebook is used to upload locally stored folder to the remote Azure ML data storage. Use the `./configs/azure_upload_data.yaml` to manage the configuration of parameters.

### `src/models` directory
**inference.ipynb**\
In this Jupyter notebook, you run the inference of the model. A pretrained model is required for this stage. The predictions are visualized. The inference is run on the `tnbc_sample_img_patch` image from the `./configs/paths.yaml` config file.

**model_factory.py**\
Here we define the architecture of the model.

**til_dataset.py**\
This file defines a utility class that is used to output the image and mask pairs that are further used during the training and evaluation by the PyTorch `DataLoader`s.

### `src/*.py` files
**image_preprocessor.py**\
Contains the `ImageProcessor` class that groups all the preprocessing functionalities.

**image_stats.py**\
Contains the `ImageStats` class that prints the statistics of the folder that contains images, like average image height, width, area, etc.

**mask_generator.py**\
Contains the `MaskGenerator` class that groups all the mask-generating and mask-fusing functionalities.

**sample.py**\
Contains the `Sample` class that is used for visualization of the image and its mask.

**utils.py**\
Contains all other utility functionalities, for example for opening and loading `.json` and `.yaml` files.

## Installation guide

### Prerequisites
Below we list the necessary software requirements:
- Python version 3.12+
- Weights and Biases account for model logging
- Azure ML access (if you wish to train models there)

### 1. Clone the repository
Clone this repository and go into it:
```bash
git clone <repository-name>
cd <cloned-repository-name>
```
Alternatively, you can download the `.zip` file of this project and open terminal within it.

### 2. Set up the Python environment and install dependencies
Set up the virtual environment using `pip` (or create Conda environment, but we will be using `pip`).

Using MacOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

or Windows:

```bash

```


Dependencies are listed in the `requirements.txt` file. To install them all, use:
```bash
pip install -r requirements.txt
```

### 3. Configure the project

```bash
python3 main.py --data_path './example_data' --wandb '80d7df7ab330e7fe22301afbab951204c5c0b33a' --train_images_path 'processed_tiger/patches/images' --train_masks_path 'processed_tiger/patches/masks/fused_leave_1_out' --test_images_path 'processed_tnbc/patches/images' --test_masks_path 'processed_tnbc/patches/masks'
```
