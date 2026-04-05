<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:ff6f00,50:d00000,100:1a0a2e&height=220&section=header&text=Face%20Mask%20Detection%20AI&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=End-to-End%20Deep%20Learning%20Computer%20Vision%20Pipeline&descAlignY=58&descSize=18&descColor=dddddd"/>

<img src="https://img.shields.io/badge/Model-Custom%20CNN-ff6f00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
<img src="https://img.shields.io/badge/Accuracy-91.79%25-7ee787?style=for-the-badge&logo=checkmarx&logoColor=white"/>
<img src="https://img.shields.io/badge/Dataset-7.5K%2B%20Images-58a6ff?style=for-the-badge&logo=kaggle&logoColor=white"/>
<img src="https://img.shields.io/badge/Framework-Keras%20/%20TF2-d00000?style=for-the-badge&logo=keras&logoColor=white"/>

</div>

---

## 📌 Executive Summary

Public safety automation requires real-time, low-latency facial analysis. This repository features a **hand-architected Convolutional Neural Network (CNN)** designed to identify mask-wearing compliance. Unlike heavy pre-trained models, this pipeline focuses on efficiency—achieving high precision while remaining lightweight enough for edge-device deployment.

---

## 🏗️ Architectural Blueprint



The model follows a rigorous "Feature-to-Decision" flow:

```text
INPUT [128x128x3]
  │
  ▼
┌──────────────────────────────────────┐
│  FEATURE EXTRACTION (Spatial)        │
│  Conv2D (32) → MaxPool → Conv2D (64) │
│  (Captures textures and geometries)  │
└──────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────┐
│  REASONING (Dense)                   │
│  Dense (128) → ReLU → Dropout (0.5)  │
│  Dense (64)  → ReLU → Dropout (0.5)  │
└──────────────────────────────────────┘
  │
  ▼
OUTPUT [Probability Score] (Sigmoid)
```

---

## 📊 Dataset & Preprocessing

The model was trained using a high-quality, balanced dataset to ensure the CNN could learn distinct facial features for both classes without bias.

| Metric | Detail |
|:--- |:--- |
| **Source** | [Kaggle Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) |
| **Total Samples** | 7,553 images |
| **Class Balance** | ~50% With Mask / ~50% Without Mask |
| **Input Shape** | 128 x 128 (Normalized RGB) |
---
### 🛠️ Preprocessing Pipeline

Every image in the dataset undergoes a standardized transformation to ensure optimal model convergence:

-  **Image Normalization:** Pixel values are scaled from `[0-255]` to the `[0, 1]` range to stabilize gradient descent.
-  **Standardization:** All images are resized to a fixed `128x128` resolution and converted to `RGB`.
-  **Regularization:** Integrated **Dual Dropout (50%)** layers after the dense blocks to penalize complexity and neutralize overfitting.
- **Validation Strategy:** A 10% holdout during the training phase was used to monitor real-time validation accuracy and loss.

---

## 📈 Training Performance

The model architecture was optimized for fast convergence while maintaining high generalization capabilities.

| Phase | Metric | Value | Result |
|:--- |:--- |:--- |:--- |
| **Training** | Final Accuracy | **93.21%** | Strong convergence |
| **Testing** | **Test Accuracy** | **91.79%** | **Excellent Generalization** |
| **Inference** | Test Loss | **0.2373** | High-confidence predictions |

> **💡 Key Insight:** The convergence was reached in just **5 epochs**, demonstrating the effectiveness of the custom feature extraction filters designed specifically for this facial recognition domain.

---

## 🔮 Production Inference

The following function demonstrates how to deploy the model for real-time predictions. It handles the image resizing and tensor reshaping automatically to match the training environment.

```python
import cv2
import numpy as np

def predict_mask(path, model):
    # Load and resize the image to 128x128
    img = cv2.imread(path)
    img_resized = cv2.resize(img, (128, 128))
    
    # Rescale pixels and add the batch dimension
    img_scaled = img_resized / 255.0
    img_tensor = np.reshape(img_scaled, [1, 128, 128, 3])
    
    # Generate probability prediction
    prediction = model.predict(img_tensor)
    label = np.argmax(prediction)
    
    return "✅ MASK DETECTED" if label == 1 else "❌ NO MASK"

# Example:
# result = predict_mask("sample_face.jpg", my_model)
# print(result)
```


---

## 🛠️ Tech Stack

The following industry-standard tools and libraries were used to build, train, and deploy this computer vision pipeline:

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white" alt="Keras"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557c?style=flat-square&logo=python&logoColor=white" alt="Matplotlib"/>
  <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"/>
</div>

| Tool | Role |
| :--- | :--- |
| **TensorFlow / Keras** | Core deep learning framework for model architecture and training. |
| **OpenCV** | Image processing and real-time inference handling. |
| **NumPy** | High-performance multidimensional array processing. |
| **Matplotlib** | Visualizing training history and model convergence. |
| **Scikit-Learn** | Data splitting and evaluation metrics. |

---

## 👤 Author

**Mostafa Shazly** *Aspiring AI Engineer*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/mostafa-shazly-148945314)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mostafashazly7)

<div align="center">
  <br>
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:ff6f00,50:d00000,100:1a0a2e&height=100&section=footer"/>
  <br>
  <i>⭐ If you found this repository helpful, please consider leaving a star! ⭐</i>
</div>
