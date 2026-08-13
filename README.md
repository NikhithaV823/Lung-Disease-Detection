# Lung-Disease-Detection
Lung Disease Detection Using Acoustic Handcrafted Features Extracted from Respiratory Audio Sounds

## Overview

This project presents a machine learning and deep learning based approach for detecting lung diseases using respiratory audio sounds. Acoustic features are extracted from respiratory recordings and used to classify the recordings into different lung disease categories.

The project compares an Artificial Neural Network (ANN) with a One-Dimensional Convolutional Neural Network (1D-CNN) to determine an effective model for respiratory sound classification.

## Objectives

- To detect lung diseases using respiratory audio recordings.
- To extract meaningful acoustic features from respiratory sounds.
- To address class imbalance in the dataset.
- To compare the performance of ANN and 1D-CNN models.
- To develop a simple application for respiratory sound-based disease prediction.

## Dataset

The project uses the **ICBHI 2017 Respiratory Sound Database**, a publicly available respiratory sound dataset containing recordings from patients with different respiratory conditions.

The dataset was obtained from the official ICBHI Challenge website:

**Dataset Source:** [ICBHI 2017 Respiratory Sound Database](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)

After preprocessing and removal of invalid audio files, the dataset used in this project contains **915 audio samples** belonging to five classes:

- COPD
- Healthy
- URTI
- Pneumonia
- Bronchial Disease

The Bronchial Disease category combines Bronchiectasis and Bronchiolitis.

## Feature Extraction

Acoustic features are extracted from the respiratory audio recordings using the `librosa` library.

The extracted features include:

- **MFCC:** 40 features
- **Chroma STFT:** 12 features
- **Mel-spectrogram:** 128 features
- **Spectral Contrast:** 7 features
- **Tonnetz:** 6 features

A total of **193 acoustic features** are obtained for each audio sample.

## Data Preprocessing

The preprocessing pipeline includes:

1. Loading respiratory audio recordings.
2. Removing invalid or zero-byte audio files.
3. Extracting acoustic features from the recordings.
4. Combining the extracted features into a feature matrix.
5. Addressing class imbalance using SMOTE.
6. Performing a stratified train-test split.
7. Preparing the extracted features for ANN and 1D-CNN models.

## Models

### Artificial Neural Network (ANN)

The ANN model consists of multiple fully connected layers with batch normalization and dropout for improved training and regularization.

The network uses a softmax output layer to classify the respiratory recordings into the five disease categories.

### One-Dimensional Convolutional Neural Network (1D-CNN)

The 1D-CNN model uses convolutional layers to learn patterns from the extracted acoustic feature sequence.

The architecture includes:

- Conv1D layers
- Batch Normalization
- Max Pooling
- Dropout
- Global Average Pooling
- Dense layers
- Softmax output layer

## Model Comparison

Both ANN and 1D-CNN models are trained and evaluated using the same classification task. Their performance is compared using metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The model with the better overall performance is selected as the final model for the application.

## Application

A Flask-based web application is developed to provide respiratory sound-based disease prediction.

The application allows a user to upload a respiratory audio recording. The trained model processes the audio and predicts the corresponding disease category along with a confidence score.

## Technologies Used

- Python
- NumPy
- Pandas
- Librosa
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Flask
- Jupyter Notebook

***** This project is developed for academic and research purposes.*******
