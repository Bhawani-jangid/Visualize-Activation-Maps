
---

# Emotion Detection and Visualization Project

## This is repo of creation of model - [link](https://github.com/Bhawani-jangid/Emotion-Detection-Using-Image)


## Overview
This project implements an emotion detection model using deep learning and provides a graphical user interface (GUI) for users to upload images and detect emotions. Additionally, it includes functionality to visualize the activation maps of specific layers in the trained model, helping to understand which features contribute to the model's predictions.

## Components
1. **GUI for Emotion Detection**
   - The GUI allows users to upload images and detect the emotion displayed in the image.
   - Utilizes a pre-trained convolutional neural network (CNN) to classify emotions into seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

2. **Model Training Script**
   - This script is responsible for training the CNN on a dataset of facial expressions.
   - The model architecture consists of several convolutional, pooling, and dense layers optimized for emotion classification.

3. **Activation Map Visualization**
   - A script to visualize the activation maps of a specified layer in the trained model.
   - This allows users to see which features of the input image are being activated and how they influence the model's predictions.

## Requirements
To run this project, you will need the following libraries:

- Python 3.x
- TensorFlow (>= 2.0)
- OpenCV
- NumPy
- Matplotlib
- Keras

You can install the required libraries using pip:
```bash
pip install tensorflow opencv-python numpy matplotlib keras
```

## Usage
1. **Run the GUI:**
   - Launch the GUI script.
   - Click the "Upload Image" button to select an image file.
   - After uploading, click "Detect Emotion" to see the predicted emotion displayed.

2. **Train the Model:**
   - Execute the model training script to train the emotion detection model. Ensure you have the training dataset structured properly.
   - The model architecture and training process are defined in the training script.

3. **Visualize Activation Maps:**
   - Use the visualization script to load a specific image and visualize the activation maps of a selected layer in the trained model.
   - Adjust the `layer_name` variable to specify which layer's activation maps you wish to visualize.

## Image Input Requirements
- The input images for emotion detection should be grayscale images of faces resized to 48x48 pixels.

## Notes
- Ensure that the paths in the scripts (for model files and images) are correctly set to your local environment.
- The GUI and visualization features are developed using Tkinter for the GUI and Matplotlib for visualizations.

## DATASET Link - 
[Kaggle](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer) 
---
