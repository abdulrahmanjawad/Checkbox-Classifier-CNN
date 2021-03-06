# Checkbox-Classifier-CNN
A deep learning model that classifies checkboxes in an image.

# Directory Structure
Below is the struture for how to setup the directory and dataset paths etc. for easier walkthrough.
```bash
Checkbox-Classifier-CNN/
|-- data/
|   |-- checked/
|   |-- other/
|   |-- unchecked/
|-- save/
|   |-- checkbox_classifier_16_20_002.pt
|-- data exploration.ipynb
|-- inference.py
|-- train.py
|-- evaluate.py
|-- requirements.txt
```
Model Weights naming: checkbox_classifier_batchsize_epochs_learningrate.pt

# Getting Started
Make sure [Python 3](https://www.python.org/downloads/) is already installed.

### Setting up the environment
 1. Clone or download the repository on your local machine.
 2. Use `cd` to command in the terminal to move into the directory.
 3. Within the Checkbox-Classifier-CNN directory create a Virtual Python Environment with command:
      ```bash
      python -m venv venv
      ```
    where `venv` is the name of the environment.
 4. Activate the enviroment using the command:
      ```bash
      source venv/bin/activate
      ```
 5. Install the required packages using:
      ```bash
      pip install -r requirements.txt
      ```
 
 ### Download pretrained weights
 1. Create a folder names `save` inside the repo. Refer to the directory structure for understanding
 2. Download weights from [Google Drive Link](https://drive.google.com/drive/folders/1y56-0YsTCSJlH7Xvl0r-KPp4Rt9LAl19?usp=sharing)
 3. Place the weights in `save` folder.
 
 ### Dataset
 Place the dataset inside the repo as shown in the directory structure above.

# Inference
Inference script can be used to make predictions on images using trained models. Below is the command-line method to use the inference script.

```bash
python inference.py -m path/to/model -i path/to/image
```
e.g `python inference.py -m save/checkbox_classifier_16_20_002.pt -i data/unchecked/00c823c283b9f48037db70f39229dc16.png`

# Model
Below are details regarding the data preprocessing, architecture, hyperparameters and evaluation.

## Data Preprocessing
- finding the min-max height and width of the dataset images to get better understanding of optimal image size.
- ideal image size selected (H, W): (256, 512) 
- calculating the mean and standard deviation of the dataset for normalization

## Architecture
Resnet18 architecture used for training. Model trained from scratch.

## Hyperparameters
Default hyperparameters used:
- Batch Size: 16
- Epochs: 20
- Optimizer: Adam
- Learning Rate: 0.002

## Evaluation
Model was evaluated in every epoch after training loop. Dataset was split 75-25 train-val. Validation set was used to assess model performance.
Model can also be evaluated for confusion matrix and precision/recall. Use the command-line below:
```bash
python evaluate.py -m path/to/model
```
e.g `python evaluate.py -m save/checkbox_classifier_16_20_01.pt`

# Training
To train the model use the command-line below:
```bash
python train.py -d data_dir -b batch_size -e epochs -l learning_rate -s save_path -t split
```
e.g `python train.py -d data/ -b 16 -e 20 -l 0.002 -s save/`

# Notebooks
Visit the notebooks to see data exploration etc.
- [Data Exploration](data_exploration.ipynb)

# Tools
- [Python 3](https://www.python.org/downloads/)
- [PyTorch](pytorch.org)

