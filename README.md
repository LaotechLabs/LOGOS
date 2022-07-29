# LOGOS: A Brand Independent logo detection model

This is an implementation of LOGOS on Python 3.7.13, IceVision, and Fastai. The model generates bounding boxes for each instance of a logo in the image. It's based on retinanet and a ResNet50 backbone.

![](./images/cover.JPG)

The Repository includes:
  - Jupyter notebook to train the model
  - Jupyter notebook to download and annotate the model
  - Jupyter notebook to perform inference 
  - Pretrained weights for the model
  - Annotations for the datasets
  
 
## Getting Started

| File Name                   	| Description                                                                                                                          	|                                                                                                                                                                                	|
|-----------------------------	|--------------------------------------------------------------------------------------------------------------------------------------	|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| inference.ipynb             	| It is the easiest way to start. It shows an example of how to perform inference on an image with the help of the pre trained weights. 	| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LaotechLabs/LOGOS/blob/main/inference.ipynb)             	|
| download_and_annotate.ipynb 	| It shows how to download and generate the annotations for LogoDet3K and Visually29K datasets                                         	| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LaotechLabs/LOGOS/blob/main/download_and_annotate.ipynb) 	|
| Training.ipynb              	| Shows how to train the model using LogoDet-3K and Visually29K datasets                                                               	| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LaotechLabs/LOGOS/blob/main/Training.ipynb)              	|

## Datasets

| Dataset     	| Link                                             	|
|-------------	|--------------------------------------------------	|
| LogoDet-3K   	| https://github.com/LaotechLabs/LOGOS 	            |
| Visuallydata 	| https://github.com/diviz-mit/visuallydata        	|

## Contributing

Contributions to this repository are welcome.

## Requirements

Python 3.7, torch==1.10.0, torchvision==0.11.1, icevision==0.12.0

## Installation

If you are using colab all installation instructions are given in the first cell of the notebooks
