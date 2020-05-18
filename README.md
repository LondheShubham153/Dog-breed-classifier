# Dog Breeds Classification with CNN Transfer Learning

### Table of Contents

1. [Installation](#installation)
2. [Project Overview](#overview)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

Beyond the Anaconda distribution of Python, the following packages need to be installed:
* opencv-python==3.2.0.6
* h5py==2.6.0
* matplotlib==2.0.0
* numpy==1.12.0
* scipy==0.18.1
* tqdm==4.11.2
* scikit-learn==0.18.1
* keras==2.0.2
* tensorflow==1.0.0   `   

## Project Overview<a name="overview"></a>

In this project, I built and trained a neural network model with CNN (Convolutional Neural Networks) transfer learning, using 8351 dog images of 133 breeds. CNN is a type of deep neural networks, which is commonly used to analyze image data. Typically, a CNN architecture consists of convolutional layers, activation function, pooling layers, fully connected layers and normalization layers. Transfer learning is a technique that allows a model developed for a task to be reused as the starting point for another task.
The trained model can be used by a web or mobile application to process real-world, user-supplied images.  Given an image of a dog, the algorithm will predict the breed of the dog.  If an image of a human is supplied, the code will identify the most resembling dog breed.

## File Descriptions <a name="files"></a>

Below are main foleders/files for this project:
1. haarcascades
    - haarcascade_frontalface_alt.xml:  a pre-trained face detector provided by OpenCV
2. bottleneck_features
    - DogVGG19Data.npz: pre-computed the bottleneck features for VGG-19 using dog image data including training, validation, and test
3. saved_models
    - VGG19_model.json: model architecture saved in a json file
    - weights.best.VGG19.hdf5: saved model weights with best validation loss
4. dog_app.ipynb: a notebook used to build and train the dog breeds classification model 
5. extract_bottleneck_features.py: functions to compute bottleneck features given a tensor converted from an image
6. images: a few images to test the model manually

Note: 
The dog image dataset used by this project can be downloaded here: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
The human image dataset can be downloaded here: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip

## Results<a name="results"></a>

1. The model was able to reach an accuracy of 72.97% on test data.
2. If a dog image is supplied, the model gives a prediction of the dog breed.
3. The model is also able to identify the most resembling dog breed of a person.

Project files can be found in this github repo: https://github.com/swang13/dog-breeds-classification
More discussions can be found in this blog: https://medium.com/@wangshuocugb2005/dog-breeds-classification-with-cnn-transfer-learning-92217cba3129

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credits must be given to Udacity for the starter codes and data images used by this project. 

