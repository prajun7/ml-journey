## Group 13 - Final Project

The dataset link: https://paperswithcode.com/dataset/cifar-100

The proposal:

Name of the project:
CIFAR-100

Objective:
The goal of this project is to develop a convolutional neural network (CNN) to classify images
from the CIFAR-100 dataset into 100 distinct categories. This project will focus on applying
machine learning techniques to achieve accurate multi-class image classification, providing
hands-on experience with CNNs.

Data and data preprocessing:
The CIFAR-100 dataset consists of 60,000 32x32 color images, with 50,000 for training and
10,000 for testing, distributed across 100 classes (500 training and 100 testing images per class).
Before feeding the data into a model, preprocessing steps are applied to improve learning
efficiency. This includes converting images into tensors, which allows them to be processed
efficiently by deep learning frameworks. Additionally, the pixel values are normalized to bring
them into a standard range, helping stabilize and accelerate the training process.

Architecture of the neural network:
This architecture starts with a low number of filters to capture basic features and increases to
capture more complex patterns, culminating in a softmax layer for class probability output. The
use of ReLU activation ensures non-linearity, while max pooling reduces spatial dimensions,
controlling computational complexity.

Layer Type : Details
Input : 32x32x3 color image
⬇️
Convolutional 1 : 32 filters, kernel size 3x3, stride 1, ReLU
⬇️
Max Pooling : 2x2, stride 2
⬇️
Convolutional 2 : 64 filters, kernel size 3x3, stride 1, ReLU
⬇️
Max Pooling : 2x2, stride 2
⬇️
Convolutional 3 : 128 filters, kernel size 3x3, stride 1, ReLU
⬇️
Max Pooling : 2x2, stride 2
⬇️
Flatten : Convert 3D feature maps to 1D vector
⬇️
Fully Connected 1 : 512 units, ReLU activation
⬇️
Fully Connected 2 : 100 units, softmax activation
