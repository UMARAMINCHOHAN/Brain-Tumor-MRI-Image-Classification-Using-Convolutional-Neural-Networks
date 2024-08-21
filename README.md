# Brain Tumor MRI Image Classification Using Deep Learning
## Project Overview
### The Project is a part of the MSc Data Science course at the University of Hertfordshire and aims at classifying brain tumors using MRI images. Use the data to train deep learning models that can automatically classify MRI images into one of four categories: glioma, meningioma, pituitary tumors, or no tumor. Deep Learning Project: Convolutional Neural Networks (CNNs) for faster and better tumor detection.
## Motivation
### Brain tumors are also one of the most challenging and dangerous diseases that ever existed, thus making an early diagnosis for appropriate treatment so crucial. Manual interpretation of MRI scans by radiologists is slow and error-prone. This project aims to mitigate these problems by constructing deep learning automated classification models, thus improving accuracy and providing access for diagnosis in low-resource environments.
## Dataset
### We will implement this project using a dataset of 7,023 MRI images that fall under these four categories:
### •	Glioma 
### •	Meningioma
### •	Pituitary tumors
### •	No tumor
### The image libraries are from three publicly available datasets: Figshare, SARTAJ, and Br35H. Pre-processing steps were also performed to maintain consistency, including resizing, normalization, and data augmentation during training.
## Models Developed
### This project created and evaluated three models.
### 1.	Model 1: A simple CNN with basic convolutional layers and dropout designed for image classification. Test Accuracy: 81.39%.
### 2.	Model 2: CNN for Image Diagnosis/Caption Complexity Model to capture more complex anatomical features. This one, on the other hand, got only 69.49% test accuracy which is not great and signals overfitting or bad hyperparameters.
### 3.	VGG16 with Transfer Learning: A pre-trained VGG16 model trained over the MRI data. With 94.05% test accuracy, our model was able to generalize well and give the best performance among all models on the test dataset.
## Results
### •	Model 1: 81.39%, facing an accuracy challenge for glioma, and pituitary tumors.
### •	Model 2: Test accuracy: 69.49%, struggled with overfitting and suboptimal performance across all classes.
### •	VGG16 with Transfer Learning: Test accuracy: 94.05%, but still faced challenges with class-specific performance, particularly in precision and recall metrics.
## Future Work
### •	Data Augmentation and Expansion: Explore advanced data augmentation techniques and expand the dataset to improve model generalizability.
### •	Advanced Architectures: Experiment with more sophisticated architectures like ResNet or DenseNet to enhance feature extraction and classification performance.
### •	Hyperparameter Optimization: Utilize techniques such as Bayesian optimization or grid search to fine-tune model parameters for better performance.
### •	Transfer Learning Improvements: Investigate other pre-trained models and fine-tuning strategies to further improve classification accuracy.

