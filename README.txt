# Image Classification on CityScapes Dataset

This project implements an image classification pipeline using the **CityScapes dataset**.
It features a custom training loop with early stopping, model checkpointing, and an inference script for evaluating the trained weights.


## Project Overview

The core of this project is built on the **ResNet** (Residual Network) architecture.
This choice was made after comparative testing against other classic architectures like AlexNet and VGG16,
as ResNet consistently provided the highest accuracy for this specific classification task.

### Why ResNet?
ResNet’s primary advantage lies in its use of **residual connections (skip connections)**.
These allow gradients to flow more easily through the network during backpropagation, effectively mitigating the **vanishing gradient problem**.
This architectural design enables the training of much deeper networks without a loss in performance.


---
## Getting Started

### 1. Data Setup
Download the CityScapes dataset (e.g., from [Kaggle](https://www.kaggle.com/datasets/shuvoalok/cityscapes)) and store the files in the `data/` directory.

### 2. Training
To begin training the model, run:
```bash
python3 training.py ./data
```
The training process utilizes a "classic fit" method with early stopping to ensure the best performing model is saved.

### 3. Inference
```bash
python3 inference.py ./data model.pt
```


## Project Stucture
The codebase is modularized into four main components:

`network.py`: Defines the ResNet architecture.
`dataset.py`: Handles data loading and preprocessing.
`training.py`: Contains the training logic and optimization.
`inference.py`: Used for model evaluation and testing.


## Project mandatory parts:

### STUDENT's UCO: 525265

############################################################################################################
### Write short answers to the questions. Please do not exceed a total of 250 words for all of your answers. #
############################################################################################################

1. Project code:
Code is written in python and can be found in files: dataset.py, inference.py, network.py, training.py. 
For training I used classic fit method with early stopping and picking the best model.


2. What model architecture did you use? Why did you choose it?
I used Resnet architecture, which is good at training very deep neural networks effectively.
Resnet contains residual connections and skip connections,
which help to mitigate the vanishing gradient problem by allowing gradients to flow directly through the network.
Resnet also achieved very good results in image classification tasks. 
I tried to use another architectures as AlexNet and VGG16, but Resnet has achieved best results so I use this architecture.
