# 🫁 PneumoScan AI — Pneumonia Detection Using CNN

An end-to-end deep learning web application that classifies chest X-ray images as **PNEUMONIA** or **NORMAL** using a hybrid DenseNet121 model with a custom attention mechanism.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Model Training Pipeline](#model-training-pipeline)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Disclaimer](#disclaimer)

---

## Overview

**PneumoScan AI** is a full-stack web application that leverages a convolutional neural network (CNN) to detect pneumonia from chest X-ray images. The project combines a **DenseNet121** backbone with a **custom attention mechanism** to improve classification accuracy, backed by a responsive web interface built with vanilla HTML/CSS/JS and served through an Express.js backend.

The model was trained on the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle, achieving **~88% test accuracy** and a **0.95 AUC score**.

---

## Key Features

- **Hybrid CNN Architecture** — DenseNet121 backbone + custom attention mechanism for enhanced feature extraction
- **Modern Web Interface** — Responsive, premium UI branded as "PneumoScan AI" with drag-and-drop uploads
- **Real-Time Predictions** — Upload an X-ray and receive classification results with confidence scores in seconds
- **Confidence Visualization** — Animated confidence meter with color-coded indicators (red for pneumonia, green for normal)
- **Two-Phase Training** — Transfer learning followed by fine-tuning for optimal performance
- **Responsive Design** — Fully responsive layout across desktop, tablet, and mobile devices
- **Drag & Drop Upload** — Intuitive image upload via click or drag-and-drop

---

## Project Structure

```
Pneumonia_Detection-main/
│
├── model_weight/                  # Model weights directory (git-ignored)
│   └── vgg19_model_final.h5       # Final trained model (~30 MB)
│
├── public/                        # Static frontend assets
│   └── index.html                 # Single-page application (HTML + CSS + JS)
│
├── uploads/                       # Uploaded images stored here by Multer
│
├── hybrid.ipynb                   # Jupyter notebook — full training pipeline
├── predict.py                     # CLI / backend prediction script
├── server.js                      # Express.js API server
├── package.json                   # Node.js dependencies
├── package-lock.json              # Dependency lock file
├── .gitignore                     # Ignores model_weight/
└── README.md                      # This file
```

---

## Model Architecture

The model uses a **hybrid architecture** combining transfer learning with a custom attention mechanism:

**Layer-by-Layer Breakdown:**

1. **Input** — 224 × 224 × 3 (RGB image)
2. **DenseNet121** — Pre-trained on ImageNet, used as the feature extractor (frozen during initial training)
3. **GlobalAveragePooling2D** — Reduces spatial dimensions to a 1024-dim feature vector
4. **Attention Branch:**
   - Dense(256, ReLU) → BatchNorm → Dropout(0.5) → Dense(1, Sigmoid)
   - Produces a scalar attention weight
5. **Multiply** — Applies attention weight to the feature vector (element-wise)
6. **Classification Head:**
   - Dense(512, ReLU, L2 regularization) → BatchNorm → Dropout(0.6)
   - Dense(256, ReLU, L2 regularization) → BatchNorm → Dropout(0.5)
7. **Output** — Dense(1, Sigmoid) → probability in [0, 1]
   - `< 0.5` → PNEUMONIA
   - `≥ 0.5` → NORMAL

**Model Summary:**

- **Total Parameters:** 7,960,642 (~30.37 MB)
- **Trainable (initial phase):** 921,090 (~3.51 MB)
- **Non-trainable (frozen DenseNet121):** 7,039,552 (~26.85 MB)

---

## System Architecture

The application follows a three-tier architecture:

**1. Frontend (Browser) — `public/index.html`**
- Responsive SPA built with Poppins font, Font Awesome icons, and vanilla CSS/JS
- Drag-and-drop or click-to-upload image area
- Processing spinner overlay while model runs
- Animated confidence meter with color-coded results
- Sends X-ray image as `multipart/form-data` via `POST /predict`

**2. Backend (Node.js) — `server.js`**
- Express.js server running on port 3000
- Multer middleware handles file uploads (filters for JPG/JPEG/PNG only)
- Saves uploaded images to the `uploads/` directory
- Spawns `predict.py` as a child process, passing the image path
- Parses JSON output from stdout and returns `{ label, confidence }` to the client
- CORS enabled, SPA fallback routing for all GET requests

**3. ML Engine (Python) — `predict.py`**
- Loads the trained model from `model_weight/vgg19_model_final.h5`
- Preprocesses the image: resize to 224×224, BGR→RGB conversion, normalize to [0, 1]
- Runs inference with `model.predict()`
- Handles both sigmoid (1 output) and softmax (2 outputs) model formats
- Outputs JSON to stdout: `{ "label": "PNEUMONIA", "confidence": 97.5 }`
- Debug logging goes to stderr

---

## Dataset

The project uses the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle:

- **Train:** ~3,875 PNEUMONIA + ~1,341 NORMAL = ~5,216 images
- **Validation:** 8 PNEUMONIA + 8 NORMAL = 16 images
- **Test:** 390 PNEUMONIA + 234 NORMAL = 624 images

> **Note:** The dataset is imbalanced (PNEUMONIA ≈ 74.3% of training data). This is handled using **computed class weights** via `sklearn.utils.class_weight`.

---

## Model Training Pipeline

The complete training pipeline is documented in `hybrid.ipynb`:

### 1. Data Preprocessing

- Load images as **grayscale** using OpenCV
- Apply **histogram equalization** to enhance contrast
- Resize to **224 × 224** pixels
- Convert to **3-channel** format (required by DenseNet121)
- Normalize pixel values to **[0, 1]**

### 2. Data Augmentation

Applied only to training data via Keras `ImageDataGenerator`:

- Rotation: ±10°
- Width/Height shift: ±10%
- Shear: 0.1
- Zoom: ±10%
- Horizontal flip: enabled
- Fill mode: nearest

### 3. Training Phase (50 Epochs)

- **Optimizer:** Adam (lr=1e-4)
- **Loss:** Binary Cross-Entropy
- **Batch Size:** 32
- **Metrics:** Accuracy, Precision, Recall, AUC
- **Class Weights:** Auto-computed for imbalanced data
- **Callbacks:**
  - `EarlyStopping` (patience=8, restore best weights)
  - `ReduceLROnPlateau` (factor=0.2, patience=4, min_lr=1e-7)
  - `ModelCheckpoint` (save best model)
  - `TensorBoard` (training logs)

### 4. Fine-Tuning Phase (10 Epochs)

After the initial training, the **last 20 layers of DenseNet121** are unfrozen:

- **Learning Rate:** 1e-5 (10× lower than initial phase)
- **Loss:** Binary Cross-Entropy
- This allows the pre-trained feature extractor to adapt to the medical imaging domain

---

## Model Performance

### Test Set Results (624 images)

- **Accuracy:** 87.98%
- **Precision:** 88.41%
- **Recall:** 78.21%
- **AUC:** 95.31%

### Per-Class Classification Report

```
              precision    recall  f1-score   support

   PNEUMONIA       0.88      0.94      0.91       390
      NORMAL       0.88      0.78      0.83       234

    accuracy                           0.88       624
   macro avg       0.88      0.86      0.87       624
weighted avg       0.88      0.88      0.88       624
```

### Training Progression Highlights

- **Epoch 1:** Train Acc 62.6%, Val AUC 89.1%
- **Epoch 10:** Train Acc 91.1%, Val AUC 96.9%
- **Epoch 30:** Train Acc 94.5%, Val AUC 100%
- **Epoch 50:** Train Acc 94.1%, Val AUC 100%
- **Fine-tune Epoch 10:** Train Acc 99.6%

---

## Tech Stack

**Backend:**
- Node.js + Express 5 — Web server and API routing
- Multer 2 — Multipart file upload handling
- CORS — Cross-origin resource sharing

**Machine Learning:**
- Python 3.12 — ML scripting
- TensorFlow / Keras — Deep learning framework
- OpenCV — Image loading and preprocessing
- NumPy — Numerical computations
- scikit-learn — Class weight computation and metrics
- Matplotlib / Seaborn — Training visualization

**Frontend:**
- HTML5 + CSS3 + Vanilla JavaScript
- Google Fonts (Poppins) — Typography
- Font Awesome 6 — UI icons

---

## Installation and Setup

### Prerequisites

- **Node.js** (v18+ recommended)
- **Python** (3.10+ recommended)
- **pip** (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Md-Shamir-raza/Pneumonia-Detection-using-CNN.git
cd Pneumonia-Detection-using-CNN
```

### Step 2: Install Node.js Dependencies

```bash
npm install
```

### Step 3: Set Up Python Environment

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Install Python packages:

```bash
pip install tensorflow opencv-python numpy scikit-learn
```

### Step 4: Place Model Weights

Ensure the trained model file exists at:

```
model_weight/vgg19_model_final.h5
```

To train the model yourself, run the `hybrid.ipynb` notebook with the [Chest X-Ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) placed as `archive.zip` in the project root.

### Step 5: Configure Python Path (if needed)

The server uses the system `python` command by default. If your Python is at a different path, update line 35 in `server.js`:

```javascript
const command = `python ${pythonScript} "${imagePath}"`;
// Change 'python' to your specific path, e.g.:
// const command = `.venv/bin/python ${pythonScript} "${imagePath}"`;
```

### Step 6: Start the Server

```bash
node server.js
```

Open your browser at: **http://localhost:3000**

---

## Usage

### Web Interface

1. Open **http://localhost:3000** in your browser
2. Scroll to the **"Upload Chest X-Ray"** section
3. **Click** the upload area or **drag & drop** an X-ray image (JPG/JPEG/PNG)
4. Wait for the AI model to process the image
5. View the result — **PNEUMONIA Detected** (red) or **Normal** (green) with a confidence percentage
6. Click **"Analyze Another"** to test a new image

### Command Line Prediction

Run predictions directly from the terminal:

```bash
python predict.py path/to/chest-xray.jpg
```

Output (stdout):

```json
{"label": "PNEUMONIA", "confidence": 97.5}
```

Debug information is printed to stderr for troubleshooting.

---

## API Reference

### POST /predict

Upload a chest X-ray image for classification.

**Request:**
- Content-Type: `multipart/form-data`
- Field name: `image`
- Accepted formats: JPG, JPEG, PNG

**Success Response (200):**

```json
{
  "label": "PNEUMONIA",
  "confidence": 97.5
}
```

**Error Response (400):**

```json
{
  "error": "No image uploaded."
}
```

**Example with cURL:**

```bash
curl -X POST -F "image=@chest_xray.jpg" http://localhost:3000/predict
```

---

## Disclaimer

> **This project is intended for educational and experimental purposes only.**

- The prediction results should **NOT** be used as a final medical diagnosis.
- Always consult a qualified healthcare professional for medical advice.
- Uploaded images are stored in the `uploads/` directory on the server.
- The model was trained on a specific dataset and may not generalize to all clinical scenarios.

---

**Built with ❤️ using TensorFlow, Express.js, and a passion for AI in healthcare.**
