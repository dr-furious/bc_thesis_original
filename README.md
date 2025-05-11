# Histological Image Data Processing Using Methods of Computer Vision and Deep Neural Networks

**Bachelor's Thesis** \
**Thesis ID:** FIIT-100241-116291 \
**Author:** Martin Sivák \
**Supervisor:** Prof. Ing. Vanda Benešová, PhD.

**Study program:** Informatics \
**Field of study:** Computer Science \
**Place of elaboration:** Institute of Computer Engineering and Applied Informatics, FIIT STU, Bratislava

## Table of contents
  * [Project's folder structure](#projects-folder-structure)
  * [Descriptions of folders and files](#descriptions-of-folders-and-files)
    + [`root` directory files](#root-directory-files)
    + [`config` directory files](#config-directory-files)
    + [`data` and `example_data` directories](#data-and-example_data-directories)
    + [`models` directory](#models-directory)
    + [`src/azure` directory](#srcazure-directory)
    + [`src/models` directory](#srcmodels-directory)
    + [`src/*.py` files](#srcpy-files)
  * [Installation guide](#installation-guide)
    + [Prerequisites](#prerequisites)
    + [1. Clone the repository](#1-clone-the-repository)
    + [2. Set up the Python environment](#2-set-up-the-python-environment)
    + [3. Install dependencies](#3-install-dependencies)
  * [How to run the Demo](#how-to-run-the-demo)
    + [1. Preprocessing and pseudo-mask creation](#1-preprocessing-and-pseudo-mask-creation)
    + [2.A Training on Azure](#2a-training-on-azure)
    + [2.B Training locally](#2b-training-locally)
    + [Inference](#inference)


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

**.gitignore**\
Files to be ignored by the Git versioning system.

**main.ipynb**\
In this Jupyter notebook, the whole preprocessing, pseudo-mask generating, and pseudo-mask fusing pipeline can be run. It also provides the visualizations of preprocessed images and pseudo-masks. By default, this notebook is run, so you can also see the outputs of each cell. Open the [main.ipynb](./main.ipynb) to see it.

**main.py**\
This is the main training and evaluation script. It can be run both locally and on the Azure ML platform. See the [Demo](#how-to-run-the-demo) section to see both possible options.

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
This file contains all paths, or parts of paths where the images are being stored, created, modified and updated, and from which are loaded during the preprocessing. The whole folder structure for the preprocessing and pseudo-mask creation is created in the [main.ipynb](./main.ipynb) notebook, here also the full paths are build. Note that by default, all file manipulations are performed under the `/example_data/` directory (unless changed in this config). For the [Demo](#how-to-run-the-demo), we recommend keeping this config file as is. For the real preprocessing, we advise to change the `root_data_dir` value to point to the `/data` directory.

### `data` and `example_data` directories
The `data` directory contains four main subdirectories. Here the images and annotations of the respective datasets reside ([TIGER](https://tiger.grand-challenge.org/Data/) and [TNBC](https://zenodo.org/records/3552674) datasets). The `/data/raw_*` folders contain the raw images and annotations (bounding box for TIGER - in the [COCO JSON](https://roboflow.com/formats/coco-json) format, PNG masks for TNBC). The `/data/preprocessed_*` directories contain more subdirectories that are created during the run of the [main.ipynb](./main.ipynb) notebook. The most important ones are:

- `/data/preprocessed_*/patches/images` which contains the 128x128 normalized image patches
- `/data/preprocessed_*/patches/masks` which contains the 128x128 mask (or pseudo-mask) patches
- `/data/preprocessed_tnbc/patches/folds` which contains TNBC image and mask patches but split into folds, where each fold directory has `/data/preprocessed_tnbc/patches/folds/fold_*/images` and  `/data/preprocessed_tnbc/patches/folds/fold_*/masks` folder

Note that this directory is meant to be used for real preprocessing, and you need to put here the correct images and annotations yourself. 

The `/example_data` directory follows the exact same structure, but already contains 10 example images from the TIGER dataset in the `/data/raw_tiger/images` subdirectory,  the `tiger-coco.json` file with the TIGER bounding box annotations in the  `/data/raw_tiger` subdirectory, and 4 images from the TNBC dataset in the `/data/raw_tnbc/images` subdirectory and their corresponding masks in the `/data/raw_tnbc/masks` subdirectory. This directory is by default listed as the `root_data_dir` in the `/configs/path.yaml` file, so in order to run the [Demo](#how-to-run-the-demo) you do not need to change anything in the `/configs/path.yaml` file.

### `models` directory
Here the models that you wish to save and use for future fine-tuning or reference should be place. We do not include any pre-trained model here, since the `.ckpt` files are around 300MB in size.

### `src/azure` directory
**azure_conda.yaml**\
Defines the dependencies that will be installed within Azure ML environment.

**azure_train.ipynb**\
From this Jupyter notebook the training is managed. This involves authenticating, pulling the correct data asset path, creating the environment and submitting the job to Azure ML. Use the `/configs/azure_connect_example.json`, `/configs/azure_job.yaml` and `/configs/model_train_base.yaml` to manage the configuration of parameters.

**azure_upload_data.ipynb**\
This Jupyter notebook is used to upload locally stored folder to the remote Azure ML data storage. Use the `/configs/azure_upload_data.yaml` to manage the configuration of parameters.

### `src/models` directory
**inference.ipynb**\
In this Jupyter notebook, you run the inference of the model. A pretrained model is required for this stage. The predictions are visualized. The inference is run on the `tnbc_sample_img_patch` image from the `/configs/paths.yaml` config file.

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
- `pip` version 23.2+
- Internet access to download packages
- Weights and Biases account for model logging
- Azure ML access (if you wish to train models there)

### 1. Clone the repository
Clone this repository and navigate into it:
```bash
git clone <repository-name>
cd <cloned-repository-name>
```
Alternatively, you can download the `.zip` file of this project, unpack it, and open terminal within it.


### 2. Set up the Python environment 
Set up the virtual environment using `pip` (or create Conda environment, but we will be using `pip`).

Using MacOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

or Windows (from PowerShell):

```shell
python -m venv .venv
.\.venv\Scripts\Activate.ps1 
```

### 3. Install dependencies
Dependencies are listed in the `requirements.txt` file. To install them all, use:
```bash
pip install -r requirements.txt
```

## How to run the Demo
Here we present a way how to run the demo version (using the demo data placed in the `/example_data` folder). Be aware of the fact, that since we only have 10 training images and 4 testing images in this demo, the model performance will be poor. This is just to showcase how the project works. To download full datasets, visit the [Grand Challenge - TIGER](https://tiger.grand-challenge.org/Data/) challenge for the TIGER dataset and the [Zenodo - TNBC](https://zenodo.org/records/3552674) for the TNBC datasets. Also note that our project works with the PNG images only.

### 1. Preprocessing and pseudo-mask creation
1. Navigate into the [main.ipynb](./main.ipynb) Jupyter notebook. You will notice that the notebook is already run (for the demonstration). Feel free to examine it before trying to run anything. 
2. Next, make sure that you clear all outputs (to avoid any confusion) and start running it cell after cell (or all at once). You will notice that under the `/example_data/processed_*` directories, different subdirectories will appear. Those will be populated with different images or versions of images and masks during the preprocessing and pseudo-mask creation. During the execution of the cells, you will also see the textual and visual output responses. 
3. After the whole notebook is run, feel free to examine the different subdirectories that were created - but be careful not to delete, move, or rename any of them or their contents.
4. The data is now prepared for the training. 

### 2.A Training on Azure
Here we describe the necessary steps that are required in order to be able to train the model on Azure ML platform.

1. Ensure you have access to an Azure ML workspace and all the required information. Fill them into the `/configs/azure_connect_example.json` configuration file.
2. Ensure that the information in the `/configs/azure_upload_data.yaml` configuration file is correct. **You will need** to input the correct `target_path` as this is **not** provided by us!
3. Then navigate into the [azure_upload_data.ipynb](./src/azure/azure_upload_data.ipynb) and run it cell by cell. Be especially careful with the local and remote directory paths. The contents of the local directory will be copied into the remote directory.
4. After the data has been uploaded, you will need to create the Azure Data Asset. See the [official Azure documentation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-data-assets) how to do it.
5. Then you will need to create an Azure compute instance. See this [official Azure documentation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-compute-instance) for precise instructions.
6. Next, you will need to modify `/configs/azure_job.yaml` file, as we cannot provide defaults for certain variables:
   - See the `dataset` top-level key. You need to input the `name` of the Data Asset and its `version` you created in Step 4.
   - See the `job` top-level key. You need to change the `job.compute` to have the name of the compute instance target you created in Step 5.
   - See the `jobs.args.wandb` key. You will need to input your Weights and Biases key, so the training and evaluation process can be monitored. See the [official guide](https://docs.wandb.ai/support/find_api_key/) how to get the key.
7. _(Optional)_ If you wish, you can try to change the model parameters, you can do so in the `/configs/model_train_base.yaml` file, but this step is optional.
8. Now navigate into the [azure_train.ipynb](./src/azure/azure_train.ipynb) and follow the instructions within it to submit the training and evaluation job to the Azure ML platform.
9. During the training you can see and monitor the whole process in your Weights and Biases account.
10. After the training and evaluation is done, look for the `outputs` folder in the job details on the Azure ML platform. It should be in the _Outputs + logs_ tab, but the Azure ML platform UI changes constantly.
11. You can download the trained model from the `outputs/checkpoints/best.ckpt`. Be aware that the checkpoint file has around 300MB in size. 

### 2.B Training locally
This options presents a way how to run the training and evaluation locally. Note that the Demo will work just fine, since there is only a fraction of the size of real dataset, but when training with a large dataset, the time to train the model locally can be significantly longer.

Follow these steps:
1. _(Optional)_ If you wish, you can try to change the model parameters, you can do so in the `/configs/model_train_base.yaml` file, but this step is optional.
2. Run the `main.py` script. Be sure to input your correct Weights and Biases key. See the [official guide](https://docs.wandb.ai/support/find_api_key/) how to get the key.
    ```bash
    python3 main.py --data_path './example_data' --wandb '<your-wandb-key>' --train_images_path 'processed_tiger/patches/images' --train_masks_path 'processed_tiger/patches/masks/fused_leave_1_out' --test_images_path 'processed_tnbc/patches/images' --test_masks_path 'processed_tnbc/patches/masks'
    ```
3. During the training you can see and monitor the whole process in your Weights and Biases account.
4. Once the training finished, you will notice that a new `/outputs` directory was created. This contains both the trained model in the `/outputs/checkpoints/best.ckpt` file and the raw Weights and Biases logs in the `outputs/wandb` folder. Furthermore, it contains a `outputs/test_results.json` with the evaluation metrics from the evaluation phase.
   
### 3. Inference
In case you wish to see how the model during the inference, navigate into the [inference.ipynb](./src/models/inference.ipynb). Notice that this notebook is run as well, feel free to examine it and then clear the outputs (to avoid any confusion). You **will need to** input the path to the trained model `.ckpt` file, as we do **not** provide a trained model in the demo. Run the notebook and see the results!
