<div align="center">

<img src="https://img.shields.io/badge/accuracy-97.07%25-brightgreen?style=for-the-badge&logo=tensorflow" />
<img src="https://img.shields.io/badge/model-EfficientNetB0-blue?style=for-the-badge" />
<img src="https://img.shields.io/badge/framework-TensorFlow%202.x-orange?style=for-the-badge&logo=tensorflow" />
<img src="https://img.shields.io/badge/UI-Streamlit-red?style=for-the-badge&logo=streamlit" />
<img src="https://img.shields.io/badge/language-Arabic%20%7C%20English-purple?style=for-the-badge" />

<br/><br/>

```
███╗   ██╗ █████╗ ██╗██╗     ███████╗██████╗     ██╗████████╗
████╗  ██║██╔══██╗██║██║     ██╔════╝██╔══██╗    ██║╚══██╔══╝
██╔██╗ ██║███████║██║██║     █████╗  ██║  ██║    ██║   ██║   
██║╚██╗██║██╔══██║██║██║     ██╔══╝  ██║  ██║    ██║   ██║   
██║ ╚████║██║  ██║██║███████╗███████╗██████╔╝    ██║   ██║   
╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝╚═════╝     ╚═╝   ╚═╝  
```

# Nailed It 

*Detecting nail conditions with 97% accuracy using transfer learning*

---

</div>

## 🔬 What Is This?

**Nailed It** is a deep learning application that classifies nail images into 4 categories using a fine-tuned **EfficientNetB0** model. Upload a photo of a nail — the model tells you if it looks healthy or flags a potential condition, along with a confidence score and clinical advice.

---

## 🎯 Classified Conditions

| Label | Condition | Arabic |
|---|---|---|
| `healthy` | Normal, healthy nail | طبيعي |
| `psoriasis` | Nail psoriasis | صدفية الأظافر |
| `clubbing` | Nail clubbing (possible systemic disease) | تضخم الأظافر |
| `Acral_Lentiginous_Melanoma` | Acral lentiginous melanoma (skin cancer) | ورم ميلانيني |

---

## 📊 Model Performance

```
Final Test Accuracy:  97.07%
Final Test Loss:      0.0959
Total Test Images:    410 (across 4 classes)
```

### Per-Class Classification Report

```
                            precision    recall  f1-score   support

Acral_Lentiginous_Melanoma     0.97      0.95      0.96       128
                  clubbing     0.95      0.97      0.96       134
                   healthy     1.00      0.98      0.99        62
                 psoriasis     0.99      1.00      0.99        86

                  accuracy                         0.97       410
               macro avg       0.98      0.97      0.98       410
            weighted avg       0.97      0.97      0.97       410
```

---

## 🏗️ Architecture & Training Pipeline

### Two-Phase Transfer Learning

```
┌─────────────────────────────────────────────────────────┐
│                    PHASE 1 — Feature Extraction          │
│                                                          │
│  EfficientNetB0 (frozen, ImageNet weights)               │
│       ↓                                                  │
│  GlobalAveragePooling2D                                  │
│       ↓                                                  │
│  BatchNormalization                                      │
│       ↓                                                  │
│  Dense(256, relu)  →  Dropout(0.5)                       │
│       ↓                                                  │
│  Dense(4, softmax)  ← output                             │
│                                                          │
│  lr=1e-3 · epochs=15 · EarlyStopping(patience=4)        │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                    PHASE 2 — Fine-Tuning                 │
│                                                          │
│  Unfreeze top 20 layers of EfficientNetB0                │
│  lr=1e-5 · epochs=20 · EarlyStopping(patience=5)        │
│  ReduceLROnPlateau(factor=0.3, patience=2)               │
└─────────────────────────────────────────────────────────┘
```

### Data Preprocessing

- **Augmentation:** rotation ±20°, zoom 20%, horizontal flip  
- **Preprocessing:** EfficientNet `preprocess_input` (scales to [-1, 1])  
- **Image size:** 224 × 224 × 3  
- **Class balancing:** computed inverse-frequency class weights  
- **Dataset split:** stratified 80/20 train/test via `train_test_split`

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/your-username/nailed-it.git
cd nailed-it
```

### 2. Install dependencies

```bash
pip install tensorflow streamlit pillow numpy scikit-learn matplotlib
```

### 3. Prepare your dataset

Place your images in the following structure:

```
dataset/
├── train/
│   ├── Acral_Lentiginous_Melanoma/
│   ├── clubbing/
│   ├── healthy/
│   └── psoriasis/
└── test/
    ├── Acral_Lentiginous_Melanoma/
    ├── clubbing/
    ├── healthy/
    └── psoriasis/
```

### 4. Train the model

```bash
python train.py
# Model saved to: model/nail_model.keras
```

### 5. Launch the app

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
nailed-it/
├── train.py              # Full training pipeline (2-phase)
├── app.py                # Streamlit web app
├── model/
│   └── nail_model.keras  # Saved best model
├── dataset/              # Original dataset (train/test)
├── dataset_fixed/        # Re-split dataset (80/20)
│   ├── train/
│   └── test/
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Deep Learning | TensorFlow / Keras |
| Base Model | EfficientNetB0 (ImageNet) |
| Data Pipeline | `ImageDataGenerator` |
| Class Balancing | Inverse-frequency weights |
| Web UI | Streamlit |
| Image Processing | Pillow, NumPy |
| Evaluation | scikit-learn |

---

## 📄 License

This project is for educational and research purposes.  

---


</div>
