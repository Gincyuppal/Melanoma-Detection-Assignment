# Skin Disease Classification using Custom CNN

## Overview

This project aims to build a custom Convolutional Neural Network (CNN) model to accurately detect various skin diseases, including melanoma. The model will be trained on a dataset containing images of different skin diseases, and the goal is to create a robust classification system for early disease detection.

## Dataset

The dataset used in this project is sourced from the International Skin Imaging Collaboration (ISIC) and contains a total of 2,357 images. The dataset includes the following diseases:
- Actinic keratosis
- Basal cell carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented benign keratosis
- Seborrheic keratosis
- Squamous cell carcinoma
- Vascular lesion

### Data Preprocessing

1. **Data Reading/Data Understanding:** The dataset was downloaded and analyzed to understand its structure and class distribution.

2. **Dataset Creation:** We created training and validation datasets from the provided images, with a batch size of 32. All images were resized to 180x180 pixels for consistency.

3. **Dataset Visualization:** We developed code to visualize a sample image from each of the nine classes to gain insights into the data.

## Model Building & Training

### Initial Model
1. **Model Architecture:** We built a custom CNN model from scratch to classify the nine classes. The model includes convolutional layers, pooling layers, and fully connected layers.

2. **Normalization:** Images were rescaled to have pixel values between 0 and 1.

3. **Optimizer and Loss Function:** We selected an appropriate optimizer and loss function for training the model.

4. **Training:** The model was trained for approximately 20 epochs.

5. **Findings:** We analyzed the training results to check for evidence of overfitting or underfitting.

### Data Augmentation
1. **Addressing Overfitting:** To address overfitting, we implemented data augmentation techniques to artificially increase the dataset size.

2. **Training on Augmented Data:** The model was trained on the augmented data for another 20 epochs.

3. **Findings:** We assessed whether the overfitting issue was resolved after training on augmented data.

### Class Distribution Analysis
1. **Class Imbalance:** We examined the distribution of classes in the training dataset to identify which class had the fewest samples.

2. **Dominant Classes:** We also identified which classes dominated the dataset in terms of the proportionate number of samples.

### Handling Class Imbalances
1. **Rectifying Imbalances:** We used the Augmentor library to address class imbalances in the training dataset.

2. **Model Training:** The model was trained on the rectified, balanced dataset for approximately 30 epochs.

3. **Findings:** We assessed whether the issues related to class imbalances were resolved after training on the rectified data.

## Conclusion

This project demonstrates the process of building a custom CNN model for multiclass classification of skin diseases. It includes steps to handle class imbalances and address overfitting using data augmentation. The trained model can be used for early detection of skin diseases, including melanoma.

## Instructions for Running the Code

1. Clone this GitHub repository.
2. Download the dataset from the provided link and organize it as specified in the code.
3. Open the Jupyter notebook provided in the repository.
4. Follow the code in the notebook to train the model and analyze the results.

## Dependencies

- Python 3.x
- TensorFlow
- Augmentor
- Jupyter Notebook

## Author

GINCY UPPAL

## Acknowledgments

- The dataset used in this project is sourced from the International Skin Imaging Collaboration (ISIC).
- Inspiration and guidance from the [course or instructor name, if applicable].
