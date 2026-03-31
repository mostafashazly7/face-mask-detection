# 😷 Face Mask Detection using Deep Learning (CNN)

## 📌 Overview
This project uses a Convolutional Neural Network (CNN) to detect whether a person is wearing a face mask or not from images.

The model is trained on a labeled dataset and achieves high accuracy in binary classification:
- ✅ With Mask
- ❌ Without Mask

---

## 📂 Dataset
Dataset sourced from Kaggle:

- 📎 Face Mask Dataset  
- Total images: **7553**
  - With Mask: **3725**
  - Without Mask: **3828**

### Classes:
| Label | Description |
|------|------------|
| 1 | With Mask |
| 0 | Without Mask |

---

## 🧹 Data Preprocessing
- Loaded images from directories
- Resized all images to **128x128**
- Converted images to RGB format
- Normalized pixel values (0 → 1)
- Converted data into NumPy arrays
- Train/Test split: **80% / 20%**

---

## 🧠 Model Architecture (CNN)

```
Input (128x128x3)
↓
Conv2D (32 filters) + ReLU
↓
MaxPooling
↓
Conv2D (64 filters) + ReLU
↓
MaxPooling
↓
Flatten
↓
Dense (128) + ReLU
↓
Dropout (0.5)
↓
Dense (64) + ReLU
↓
Dropout (0.5)
↓
Output Layer (2 neurons, Sigmoid)
```

---

## ⚙️ Training Details
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Epochs: 5
- Validation Split: 10%

---

## 📈 Model Performance

- ✅ **Training Accuracy:** ~93%
- ✅ **Validation Accuracy:** ~91%
- ✅ **Test Accuracy:** **~91.7%**

---

## 📊 Visualizations
- Training vs Validation Loss
- Training vs Validation Accuracy

---

## 🔮 Prediction
You can test the model on any image:

```python
input_image_path = 'path_to_image.jpg'
```

### Output:
- `The person is wearing a mask`
- `The person is not wearing a mask`

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/mostafashazly7/face-mask-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the notebook or script

---

## 🛠️ Technologies Used
- Python 
- NumPy
- OpenCV
- Matplotlib
- TensorFlow / Keras
- PIL (Python Imaging Library)

---

## 📌 Future Improvements
- Use Transfer Learning (ResNet, MobileNet)
- Improve accuracy with more epochs
- Real-time detection using webcam 🎥
- Deploy as a web app (Streamlit / Flask)

---

## 👤 Author
**Mostafa Shazly**

- GitHub: https://github.com/mostafashazly7

---

## ⭐ Support
If you like this project, please ⭐ the repository!
