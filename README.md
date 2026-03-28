# 😷 Face Mask Detection using Deep Learning

A deep learning-based computer vision project that detects whether a person is wearing a face mask or not using Convolutional Neural Networks (CNN).

---

## 📌 Project Overview
This project aims to classify images into two categories:
- ✅ With Mask  
- ❌ Without Mask  

The model is trained on a dataset from Kaggle and achieves high accuracy on unseen data.

---

## 📂 Dataset
The dataset is sourced from Kaggle:

🔗 https://www.kaggle.com/datasets/omkargurav/face-mask-dataset

- Total Images: **7553**
- With Mask: **3725**
- Without Mask: **3828**

---

## ⚙️ Technologies Used
- Python 🐍  
- TensorFlow / Keras 🤖  
- NumPy  
- OpenCV  
- Matplotlib  

---

## 🧠 Model Architecture
The model is built using a Convolutional Neural Network (CNN):

- Conv2D (32 filters) + ReLU  
- MaxPooling  
- Conv2D (64 filters) + ReLU  
- MaxPooling  
- Flatten  
- Dense (128) + Dropout  
- Dense (64) + Dropout  
- Output Layer (2 classes)

---

## 🚀 Training Details
- Image Size: **128x128**
- Train/Test Split: **80/20**
- Epochs: **5**
- Optimizer: **Adam**
- Loss Function: **Sparse Categorical Crossentropy**

---

## 📊 Results
- ✅ Training Accuracy: ~93%  
- ✅ Validation Accuracy: ~91%  
- ✅ Test Accuracy: **~91.7%**

---

## 📈 Performance Visualization
The model performance was evaluated using:
- Loss Curve 📉  
- Accuracy Curve 📈  

---

