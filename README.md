# Cat vs Dog Image Classification

A deep learning project for binary image classification to distinguish between cats and dogs using Convolutional Neural Networks (CNN). The project includes a training pipeline, prediction module, and an interactive web interface powered by Streamlit.

## Table of Contents
- [Project Overview](#project-overview)
- [File Structure](#file-structure)
- [Data Flow](#data-flow)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Validation Pipeline](#validation-pipeline)
- [Inference Pipeline](#inference-pipeline)
- [Hyperparameters](#hyperparameters)
- [Performance Metrics](#performance-metrics)
- [Hardware & GPU Requirements](#hardware--gpu-requirements)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Expected Outputs](#expected-outputs)
- [Sample Predictions](#sample-predictions)
- [Deployment Architecture](#deployment-architecture)
- [Future Scalability Options](#future-scalability-options)

---

## Project Overview

This project implements a binary image classifier trained on 20,000 training images (10,000 cats + 10,000 dogs) with validation on 5,000 test images (2,500 cats + 2,500 dogs). The model uses a CNN architecture with three convolutional blocks, batch normalization, and dropout regularization to prevent overfitting.

**Key Features:**
- Binary classification (Cat vs Dog)
- Web-based inference interface with Streamlit
- Support for clipboard image pasting
- Real-time confidence score visualization
- Lightweight model (~38 MB)
- Normalized 128×128 RGB image input

---

## File Structure

```
cat_vs_dog_image_classification_project/
├── app.py                              # Streamlit web application for inference
├── train.py                            # Training script with model definition
├── predict.py                          # Standalone prediction script for testing
├── requirement.txt                     # Python dependencies with versions
├── models/
│   └── cat_dog_classifier.keras       # Trained Keras model (38 MB)
└── dataset/
    ├── train/                         # Training dataset
    │   ├── cats/                      # 10,000 cat images
    │   └── dogs/                      # 10,000 dog images
    └── test/                          # Validation/test dataset
        ├── cats/                      # 2,500 cat images
        └── dogs/                      # 2,500 dog images
```

### Detailed File Descriptions

#### **app.py** (Streamlit Web Application)
**Purpose:** Interactive web interface for real-time predictions
**Key Components:**
- **Model Loading:** Loads the pre-trained Keras model from `models/cat_dog_classifier.keras`
- **Input Methods:** 
  - File upload (JPG, JPEG, PNG)
  - Clipboard paste via `streamlit_paste_button`
- **Image Processing:**
  - Converts to RGB format
  - Resizes to 128×128 pixels
  - Normalizes pixel values (divides by 255.0)
  - Adds batch dimension for model input
- **Output:**
  - Displays uploaded/pasted image
  - Shows classification result with emoji indicators
  - Displays confidence scores as percentages and progress bars
  - Decision threshold: 0.5 (>0.5 = Dog, <0.5 = Cat)
- **Dependencies:** Streamlit, TensorFlow, OpenCV, PIL, NumPy

#### **train.py** (Training Script)
**Purpose:** Trains the CNN model on the dataset
**Key Components:**
- **Data Loading:**
  - Uses `keras.utils.image_dataset_from_directory()` for automatic directory-based loading
  - Infers labels from subdirectory names (cats/dogs)
  - Batch size: 16 images per batch
  - Image size: 128×128 pixels (RGB)
- **Data Preprocessing:**
  - Normalization: Divides pixel values by 255.0
  - Casts to float32 dtype
  - Applied via `dataset.map(process)` for efficient data pipeline
- **Model Architecture:** Sequential CNN (see Model Architecture section)
- **Compilation:**
  - Optimizer: Adam (adaptive learning rate)
  - Loss: Binary Crossentropy (for binary classification)
  - Metrics: Accuracy
- **Training:**
  - Epochs: 10
  - Validation: Runs validation on test dataset after each epoch
  - Plots accuracy and loss graphs
- **Output:**
  - Saves trained model to `models/cat_dog_classifier.keras`
  - Generates training history plots

#### **predict.py** (Standalone Prediction Script)
**Purpose:** Command-line prediction utility for testing
**Key Components:**
- **Model Loading:** Loads the trained Keras model
- **Test Image:** Reads a specific test image (`dataset/test/cats/cat.26.jpg`)
- **Preprocessing:**
  - Reads image using OpenCV
  - Resizes to 128×128
  - Normalizes by dividing by 255.0
  - Adds batch dimension
- **Prediction:**
  - Outputs raw prediction value (probability of dog class)
  - Prints "CAT" if prediction < 0.5, "DOG" if ≥ 0.5
- **Use Case:** Quick verification of model predictions without UI

#### **requirement.txt** (Dependencies)
Specifies Python package versions:
```
tensorflow>=2.18.0      # Deep learning framework
numpy>=1.26.0          # Numerical computing
matplotlib>=3.9.0      # Visualization
opencv-python>=4.10.0  # Image processing
pillow>=11.0.0         # Image library
scikit-learn>=1.6.0    # Machine learning utilities
```

**Note:** `streamlit>=1.0.0` and `streamlit-paste-button` should be added for app.py to work.

#### **cat_dog_classifier.keras** (Trained Model)
- **Format:** Keras native format (.keras)
- **Size:** ~38 MB
- **Model Type:** Sequential CNN
- **Input Shape:** (None, 128, 128, 3)
- **Output Shape:** (None, 1) - single sigmoid probability
- **Training Time:** Varies by hardware (see GPU Requirements)
- **Purpose:** Binary classification of cat vs dog images

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA FLOW DIAGRAM                        │
└─────────────────────────────────────────────────────────────┘

TRAINING PIPELINE:
─────────────────
dataset/train/ ──> image_dataset_from_directory() ──> (128,128,3) images
                        ↓
              BatchSize: 16, Labels: inferred
                        ↓
              Normalization: image/255.0
                        ↓
        ┌──────────────────────────┐
        │   CNN Model Training     │
        │  (10 epochs)             │
        └──────────────────────────┘
                        ↓
              Models/cat_dog_classifier.keras
                        ↓
            History: Accuracy & Loss Plots


VALIDATION PIPELINE:
───────────────────
dataset/test/ ──> image_dataset_from_directory() ──> (128,128,3) images
                        ↓
              BatchSize: 16, Labels: inferred
                        ↓
              Normalization: image/255.0
                        ↓
        ┌──────────────────────────┐
        │   Model Validation       │
        │  (Each epoch)            │
        └──────────────────────────┘
                        ↓
            Accuracy & Loss Metrics


INFERENCE PIPELINE (Streamlit App):
──────────────────────────────────
User Input (Upload/Paste) ──> Load as PIL Image
                        ↓
                Convert to RGB
                        ↓
            Resize to 128×128 pixels
                        ↓
           Normalize: image/255.0
                        ↓
         Add batch dimension (1, 128, 128, 3)
                        ↓
           ┌─────────────────────────┐
           │  Model.predict(image)   │
           │  Output: sigmoid prob   │
           └─────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  Threshold at 0.5:            │
        │  > 0.5 = Dog                  │
        │  < 0.5 = Cat                  │
        └───────────────────────────────┘
                        ↓
         Display result + confidence scores


INFERENCE PIPELINE (Standalone Script):
──────────────────────────────────────
Image file ──> cv2.imread()
            ↓
        Resize to 128×128
            ↓
    Normalize: image/255.0
            ↓
  Add batch dimension (1, 128, 128, 3)
            ↓
    Model.predict(image)
            ↓
  Print result (CAT/DOG)
```

---

## Model Architecture

### Architecture Overview
The model is a **Sequential Convolutional Neural Network** with 3 convolutional blocks followed by a dense classifier.

```
Input Layer
│
├─ Conv2D(32, 3×3, ReLU)
├─ BatchNormalization
├─ MaxPooling2D(2×2)
│
├─ Conv2D(64, 3×3, ReLU)
├─ BatchNormalization
├─ MaxPooling2D(2×2)
│
├─ Conv2D(128, 3×3, ReLU)
├─ BatchNormalization
├─ MaxPooling2D(2×2)
│
├─ Flatten
│
├─ Dense(128, ReLU)
├─ Dropout(0.1)
│
├─ Dense(64, ReLU)
├─ Dropout(0.1)
│
└─ Dense(1, Sigmoid) ──> Output [0, 1]
```

### Layer-by-Layer Details

| Layer Type | Filters/Units | Kernel | Activation | Output Shape (after) | Parameters |
|-----------|--------------|--------|-----------|----------------------|-----------|
| Conv2D | 32 | 3×3 | ReLU | 126×126×32 | 896 |
| BatchNormalization | - | - | - | 126×126×32 | 128 |
| MaxPooling2D | - | 2×2 | - | 63×63×32 | 0 |
| Conv2D | 64 | 3×3 | ReLU | 61×61×64 | 18,496 |
| BatchNormalization | - | - | - | 61×61×64 | 256 |
| MaxPooling2D | - | 2×2 | - | 30×30×64 | 0 |
| Conv2D | 128 | 3×3 | ReLU | 28×28×128 | 73,856 |
| BatchNormalization | - | - | - | 28×28×128 | 512 |
| MaxPooling2D | - | 2×2 | - | 14×14×128 | 0 |
| Flatten | - | - | - | 25,088 | 0 |
| Dense | 128 | - | ReLU | 128 | 3,211,392 |
| Dropout | - | - | - | 128 | 0 (rate: 0.1) |
| Dense | 64 | - | ReLU | 64 | 8,256 |
| Dropout | - | - | - | 64 | 0 (rate: 0.1) |
| Dense | 1 | - | Sigmoid | 1 | 65 |

**Total Parameters:** ~3,313,809
**Trainable Parameters:** ~3,313,809

### Architecture Design Rationale
- **Convolutional Blocks:** 3 levels of feature extraction with progressively increasing filters (32 → 64 → 128)
- **Batch Normalization:** Stabilizes training, allows higher learning rates
- **MaxPooling:** Reduces spatial dimensions, extracts dominant features
- **Dropout:** Regularization with 0.1 rate (10% neuron dropout) prevents overfitting
- **Sigmoid Output:** Produces probability [0,1] for binary classification
- **Dense Layers:** Classifier head with 128 and 64 units for final decision

---

## Training Pipeline

### Dataset Preparation
**Training Dataset:**
- **Source:** `dataset/train/` directory
- **Structure:** Subdirectories for each class (cats, dogs)
- **Total Images:** 20,000
  - Cats: 10,000 images
  - Dogs: 10,000 images
- **Image Format:** JPEG, PNG, or other OpenCV-compatible formats
- **Class Distribution:** Perfectly balanced (50% cats, 50% dogs)

**Validation Dataset:**
- **Source:** `dataset/test/` directory
- **Total Images:** 5,000
  - Cats: 2,500 images
  - Dogs: 2,500 images

### Data Loading & Preprocessing
```python
# Step 1: Load from directory
train_ds = keras.utils.image_dataset_from_directory(
    directory='dataset/train',
    labels='inferred',           # Infers from subdirectory names
    label_mode='int',            # 0 for cats, 1 for dogs (alphabetical)
    batch_size=16,               # Process 16 images at a time
    image_size=(128,128)         # Resize all images to 128×128
)

# Step 2: Normalize
def process(image, label):
    image = tf.cast(image/255.0, tf.float32)  # Scale to [0, 1]
    return image, label

train_ds = train_ds.map(process)
```

### Training Execution
```
Input: 128×128×3 RGB images
       ↓
[Train Batch 1 (16 images)] → Model Forward Pass → Loss Calculation
       ↓
[Adam Optimizer] → Gradient Computation → Backpropagation
       ↓
[Weight Updates] → Repeat for all training batches
       ↓
[Validation] → Validate on test set
       ↓
[History Recording] → Accuracy and Loss metrics
       ↓
[Repeat] → Next epoch (total 10 epochs)
```

### Training Configuration
| Parameter | Value | Notes |
|-----------|-------|-------|
| **Batch Size** | 16 | Mini-batch gradient descent |
| **Epochs** | 10 | Number of passes through entire dataset |
| **Optimizer** | Adam | Adaptive learning rate, momentum-based |
| **Loss Function** | Binary Crossentropy | Standard for binary classification |
| **Learning Rate** | 0.001 | Default Adam learning rate |
| **Metrics** | Accuracy | Monitored during training |

### Training Output
- **Model Summary:** Displays all layers with parameter counts
- **Per-Epoch Metrics:**
  - Training accuracy
  - Training loss
  - Validation accuracy
  - Validation loss
- **Visualization:** Matplotlib plots showing:
  - Accuracy trends (training vs validation)
  - Loss trends (training vs validation)
- **Saved Artifact:** `models/cat_dog_classifier.keras`

---

## Validation Pipeline

### Validation Dataset
- **Source:** `dataset/test/` directory
- **Images:** 5,000 (2,500 cats + 2,500 dogs)
- **Preprocessing:** Same as training (resize to 128×128, normalize by dividing by 255.0)
- **Batch Size:** 16

### Validation Process
```
For each epoch:
  ├─ Run training on train_ds
  ├─ After training updates complete:
  │  ├─ Load validation_ds batches
  │  ├─ Forward pass (no gradient computation)
  │  ├─ Calculate loss and accuracy
  │  └─ Aggregate metrics across all validation batches
  └─ Record validation_accuracy and validation_loss
```

### Validation Metrics
- **Validation Accuracy:** Percentage of correctly classified validation images
- **Validation Loss:** Binary crossentropy loss on validation set
- **Overfitting Indicator:** Comparison of training vs validation curves
  - If val_loss diverges upward while train_loss decreases → overfitting
  - Dropout (0.1 rate) and BatchNormalization help mitigate this

### Validation Output
The training script generates two plots:
1. **Accuracy Plot:** Overlays training and validation accuracy over 10 epochs
2. **Loss Plot:** Overlays training and validation loss over 10 epochs

---

## Inference Pipeline

### Streamlit Web Application (`app.py`)

#### Input Methods
1. **File Upload:**
   - Accepts JPG, JPEG, PNG formats
   - Maximum file size: Default Streamlit limit (typically 200 MB)
   
2. **Clipboard Paste:**
   - Direct image paste via `streamlit_paste_button`
   - Uses keyboard shortcut (Ctrl+V or Cmd+V)

#### Image Processing Steps
```python
1. Load Image:
   image = Image.open(file).convert("RGB")  # or from clipboard
   
2. Convert to Array:
   img = np.array(image)
   
3. Handle Grayscale (if needed):
   if len(img.shape) == 2:
       img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
   
4. Resize:
   img = cv2.resize(img, (128, 128))
   
5. Normalize:
   img = img / 255.0
   
6. Add Batch Dimension:
   img = np.expand_dims(img, axis=0)  # Shape: (1, 128, 128, 3)
   
7. Inference:
   prediction = model.predict(img, verbose=0)  # Output: [[probability]]
   
8. Extract Probability:
   dog_prob = float(prediction[0][0])  # Range: [0.0, 1.0]
   cat_prob = 1 - dog_prob
```

#### Decision Logic
```
if dog_prob > 0.5:
    Classification: DOG
    Confidence: dog_prob × 100 %
else:
    Classification: CAT
    Confidence: cat_prob × 100 %
```

#### Output Display
- **Image Display:** Original uploaded/pasted image (350px width)
- **Result Badge:** ` Dog (X.XX%)` or ` Cat (X.XX%)`
- **Confidence Bars:** Progress bars for cat and dog percentages
- **Detailed Scores:** Numerical percentages (e.g., "Cat: 75.43%")

### Standalone Prediction Script (`predict.py`)

#### Workflow
```python
1. Load Model:
   model = tf.keras.models.load_model("models/cat_dog_classifier.keras")
   
2. Read Image:
   img = cv2.imread("dataset/test/cats/cat.26.jpg")
   
3. Preprocess:
   img = cv2.resize(img, (128, 128))
   img = img / 255.0
   img = np.expand_dims(img, axis=0)
   
4. Predict:
   prediction = model.predict(img)
   
5. Output:
   Print raw prediction value
   Print classification: "CAT" or "DOG"
```

#### Use Cases
- Batch processing images
- Integration with external systems
- Programmatic predictions without UI

---

## Hyperparameters

### Model Architecture Hyperparameters
| Hyperparameter | Value | Impact |
|---|---|---|
| **Input Image Size** | 128×128 | Balance between detail and computation |
| **Conv Filter Sizes** | 32, 64, 128 | Progressive feature extraction depth |
| **Kernel Size** | 3×3 | Small receptive field, computational efficiency |
| **Activation Function** | ReLU (hidden), Sigmoid (output) | Non-linearity, binary output probability |
| **Pooling Size** | 2×2 | Reduce spatial dimensions by 50% |
| **Dense Layer Units** | 128, 64 | Classifier capacity |
| **Dropout Rate** | 0.1 | 10% neuron dropout for regularization |

### Training Hyperparameters
| Hyperparameter | Value | Impact |
|---|---|---|
| **Batch Size** | 16 | Memory efficiency, gradient estimation |
| **Epochs** | 10 | Number of full dataset passes |
| **Optimizer** | Adam | Adaptive learning rate optimization |
| **Learning Rate** | 0.001 (default) | Step size for weight updates |
| **Loss Function** | Binary Crossentropy | Appropriate for binary classification |
| **Momentum (Adam β1)** | 0.9 | Exponential decay rate for 1st moment |
| **Momentum (Adam β2)** | 0.999 | Exponential decay rate for 2nd moment |

### Data Preprocessing Hyperparameters
| Hyperparameter | Value | Rationale |
|---|---|---|
| **Normalization Range** | [0, 1] | Standard practice (dividing by 255) |
| **Image Size** | 128×128 | Sufficient for cat/dog features, reasonable computation |
| **Batch Size (loading)** | 16 | Balance between memory and convergence |
| **Color Mode** | RGB | Standard for color images (not grayscale) |

---

## Performance Metrics

### Expected Training Metrics

Based on the architecture and 20,000 training images with 5,000 validation images:

| Epoch | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss |
|---|---|---|---|---|
| 1 | ~70-75% | ~70-72% | ~0.55-0.65 | ~0.58-0.68 |
| 2 | ~80-85% | ~80-82% | ~0.40-0.50 | ~0.42-0.52 |
| 3 | ~85-88% | ~84-86% | ~0.35-0.42 | ~0.38-0.48 |
| 4 | ~87-89% | ~85-87% | ~0.30-0.38 | ~0.35-0.45 |
| 5 | ~88-90% | ~86-88% | ~0.27-0.35 | ~0.32-0.42 |
| 6 | ~89-91% | ~87-89% | ~0.25-0.32 | ~0.30-0.40 |
| 7 | ~90-92% | ~87-89% | ~0.23-0.30 | ~0.29-0.39 |
| 8 | ~90-92% | ~87-89% | ~0.22-0.28 | ~0.28-0.38 |
| 9 | ~91-92% | ~88-90% | ~0.21-0.27 | ~0.28-0.38 |
| 10 | ~91-93% | ~88-90% | ~0.20-0.26 | ~0.27-0.37 |

**Note:** Exact values depend on random initialization and GPU/CPU variations.

### Expected Inference Performance

| Metric | Value |
|---|---|
| **Inference Time (Single Image)** | ~50-200 ms (CPU), ~10-50 ms (GPU) |
| **Model Size** | ~38 MB |
| **Memory Usage (Inference)** | ~100-200 MB (loading model + inference) |
| **Expected Test Accuracy** | 88-90% |

### Overfitting Analysis

The 10% dropout rate and batch normalization are designed to control overfitting:
- **Good Sign:** Validation accuracy stays within 2-3% of training accuracy
- **Warning Sign:** Validation accuracy plateaus while training continues to improve
- **Risk:** With only 10 epochs, the model may be underfitting rather than overfitting

---

## Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip or conda package manager
- Virtual environment (recommended)

### Step 1: Clone/Download Project
```bash
cd c:\Projects\cat_vs_dog_image_classification_project
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirement.txt
pip install streamlit streamlit-paste-button  # Missing from requirement.txt
```

### Step 5: Verify Installation
```bash
python -c "import tensorflow; print(tensorflow.__version__)"
python -c "import streamlit; print(streamlit.__version__)"
```

---

## Usage

### Option 1: Training the Model
```bash
# Activate virtual environment (if not already activated)
venv\Scripts\activate

# Run training script
python train.py
```

**Expected Output:**
- Model summary printed to console
- Training progress for 10 epochs
- Validation metrics per epoch
- Accuracy and loss plots displayed
- Model saved to `models/cat_dog_classifier.keras`

**Duration:** 5-120 minutes depending on hardware (GPU recommended)

### Option 2: Batch Predictions
```bash
# Run standalone prediction script
python predict.py
```

**Expected Output:**
```
Raw Prediction: [[0.123456]]
CAT
```

**Note:** Modify the image path in `predict.py` to test different images.

### Option 3: Interactive Web Interface (Recommended)
```bash
# Activate virtual environment (if not already activated)
venv\Scripts\activate

# Run Streamlit app
streamlit run app.py
```

**Access:** Opens browser at `http://localhost:8501`

**Features:**
- Upload images via file uploader
- Paste images directly from clipboard
- Real-time predictions with confidence scores
- Visual confidence bars
- Instantaneous results

---

## Expected Outputs

### Training Script Output
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 126, 126, 32)     896
 batch_normalization         (None, 126, 126, 32)     128
 max_pooling2d (MaxPooling2  (None, 63, 63, 32)       0
 conv2d_1 (Conv2D)           (None, 61, 61, 64)       18496
 batch_normalization_1       (None, 61, 61, 64)       256
 max_pooling2d_1 (MaxPooling (None, 30, 30, 64)       0
 conv2d_2 (Conv2D)           (None, 28, 28, 128)      73856
 batch_normalization_2       (None, 28, 28, 128)      512
 max_pooling2d_2 (MaxPooling (None, 14, 14, 128)      0
 flatten (Flatten)           (None, 25088)            0
 dense (Dense)               (None, 128)              3211392
 dropout (Dropout)           (None, 128)              0
 dense_1 (Dense)             (None, 64)               8256
 dropout_1 (Dropout)         (None, 64)               0
 dense_2 (Dense)             (None, 1)                65
=================================================================
Total params: 3,313,809
Trainable params: 3,313,809
Non-trainable params: 0

Epoch 1/10
625/625 [==============================] - 45s 72ms/step - loss: 0.6523 - accuracy: 0.6845 - val_loss: 0.5421 - val_accuracy: 0.7256
Epoch 2/10
625/625 [==============================] - 42s 67ms/step - loss: 0.4821 - accuracy: 0.7834 - val_loss: 0.4283 - val_accuracy: 0.8012
...
Epoch 10/10
625/625 [==============================] - 41s 65ms/step - loss: 0.2087 - accuracy: 0.9189 - val_loss: 0.2953 - val_accuracy: 0.8876

Model Saved!
```

### Streamlit Web App Interface
```
┌─────────────────────────────────────────────┐
│       Cat vs Dog Classifier                 │
├─────────────────────────────────────────────┤
│ Upload an image or paste one directly       │
│ from your clipboard.                        │
├─────────────────────────────────────────────┤
│    Upload Image  [Browse Files...]          │
├─────────────────────────────────────────────┤
│    Paste Image (Ctrl + V)  [Paste Button]   │
├─────────────────────────────────────────────┤
│ [Selected Image Display]                    │
│ ┌───────────────────────────┐               │
│ │  [350px × 350px image]    │               │
│ │  Selected Image           │               │
│ └───────────────────────────┘               │
├─────────────────────────────────────────────┤
│       Dog (73.45%)                          │
├─────────────────────────────────────────────┤
│ ### Confidence Scores                       │
│ ▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░  Cat: 26.55%    │
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░  Dog: 73.45%    │
└─────────────────────────────────────────────┘
```

### Prediction Script Output
```
Raw Prediction: [[0.2345]]
CAT
```

---

## Sample Predictions

### Sample Input 1: Clear Cat Image
```
Input: Well lit photo of a tabby cat
Output:
Raw Prediction: [[0.0892]]
CAT
Confidence: 91.08%
```

### Sample Input 2: Clear Dog Image
```
Input: Clear photo of a golden retriever
Output:
Raw Prediction: [[0.9234]]
DOG
Confidence: 92.34%
```

### Sample Input 3: Ambiguous Image
```
Input: Dark/blurry animal photo
Output:
Raw Prediction: [[0.5123]]
DOG
Confidence: 51.23%
Notes: Close to decision boundary, less confident
```

### Sample Input 4: Edge Case
```
Input: Image with both cat and dog
Output:
Raw Prediction: [[0.6789]]
DOG
Confidence: 67.89%
Notes: Model classifies based on dominant animal
```

### Typical Confidence Distribution
- **High Confidence (>90%):** Clean, clear images of single animals
- **Medium Confidence (70-90%):** Good images with minor occlusions
- **Low Confidence (50-70%):** Blurry, partially visible, or mixed images
- **Uncertain (<60%):** Near decision boundary, ambiguous images

---


## Summary of Key Metrics

- **Dataset:** 25,000 images (20K training, 5K validation)
- **Model Size:** ~38 MB (trainable)
- **Parameters:** 3.3M trainable parameters
- **Expected Accuracy:** 88-90% on validation set
- **Training Time:** 5-120 minutes (depending on hardware)
- **Inference Time:** 10-200 ms (per image)
- **GPU Memory:** 1-2 GB (batch size 16)
- **Architecture:** Sequential CNN with 3 conv blocks
- **Input:** 128×128 RGB images
- **Output:** Binary probability [0, 1]

---

## License & Attribution

This project uses publicly available cat vs dog datasets and TensorFlow/Keras frameworks.

## Support & Contributing

For issues or improvements:
1. Refer to [TensorFlow Documentation](https://www.tensorflow.org/)
2. Check [Streamlit Documentation](https://docs.streamlit.io/)
3. Review training/inference logs for debugging

---#
