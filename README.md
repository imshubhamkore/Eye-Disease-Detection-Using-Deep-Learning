# Eye Disease Detection Using EfficientNet B0

This project aims to detect eye diseases using the EfficientNet B0 model. The model is trained on a dataset of eye images and can classify different eye diseases with high accuracy.

## Table of Contents
-Introduction
-Installation
-Dataset
-Model Architecture
-Results

## Introduction
Eye diseases are a significant health concern worldwide. Early detection and diagnosis are crucial for effective treatment. This project focuses on automating the process of eye disease detection using deep learning techniques, specifically the EfficientNet B0 model.

## Installation
We are using Google Colab for creating our model.Google Colab offers free access to GPUs, seamless integration with Google Drive, and pre-installed libraries. It simplifies environment setup and sharing.

## Dataset
The dataset consists of Normal, Diabetic Retinopathy, Cataract and Glaucoma retinal images where each class have approximately 1000 images. These images are collected from various sources like IDRiD, Ocular recognition, HRF etc.

## Model Architecture
The architecture of EfficientNet B0 is based on a compound scaling method that uses a combination of depth, width, and resolution scaling to balance the trade-off between model size and accuracy. It starts with a base convolutional neural network (CNN) architecture and scales it up or down by adjusting the number of layers, width of the layers, and resolution of the input image. EfficientNet B0 has 7.8 million parameters and achieves state-of-the-art accuracy on several image classification tasks, while being up to 8.4x smaller and up to 6.1x faster than previous state-of-the-art models like ResNet-50 and ResNeXt-101. The model has been pre-trained on the ImageNet dataset and can be fine-tuned for various computer vision tasks such as object detection, segmentation, and image captioning.

## Results
-Test Accuracy- 0.9087 
-Train Accuracy: 0.9370   Loss: 0.1360 
-Val Accuracy: 0.9037     Loss: 0.2467

We have done this model by using four pretrained models, and we got the 90% accuracy from EfficientNetB0. We trained and deployed our model with that accuracy.

In this branch of the main file, I can add other pretrained models as well as the basic CNN model. I am also adding the Flask deployment Python file.
