# Speech Emotion Recognition using MFCC, CNN & BiLSTM

## Project Overview

This project builds a **Speech Emotion Recognition (SER)** system to classify human emotions from spoken audio. It processes audio files using **MFCC feature extraction** and feeds them into a **hybrid deep learning model** combining **Convolutional Neural Networks (CNN)** and **Bidirectional LSTM (BiLSTM)** layers.

The system is trained on the **RAVDESS dataset**, which includes emotional speech clips labeled with 8 emotion categories like happy, sad, angry, fearful, and more.

---

## Key Technologies

### MFCC (Mel-Frequency Cepstral Coefficients)
- **Why**: Human hearing is more sensitive to certain frequencies. MFCC mimics this by mapping speech signals onto the Mel scale.
- **What it does**: Extracts compact and informative frequency features from raw audio.
- **How it works**:
  1. Convert audio to spectrogram
  2. Apply Mel filter banks
  3. Take the logarithm
  4. Apply Discrete Cosine Transform (DCT)
- **Output**: 2D time-frequency representation used as CNN input

- they capture the spectral properties of speech that are most relevant to how humans hear. In our project, MFCCs help the model understand the tone, pitch, and rhythm, which are crucial for detecting emotions in speech.

---

## CNN (Convolutional Neural Network)
- **Why**: CNNs are great at capturing local spatial patterns — perfect for analyzing MFCC “images”.
- **What it does**: Learns short-term temporal and spectral features in the MFCC map.
- **How it works**:
  - Multiple `Conv2D` + `BatchNorm` + `ReLU` + `MaxPooling` layers
  - Final flattening and feature vector output
- **Role**: Acts as a feature extractor for the BiLSTM

---

## BiLSTM (Bidirectional Long Short-Term Memory)
- **Why**: Emotions in speech depend on the sequence of sounds, not just isolated frames.
- **What it does**: Understands temporal dependencies by reading audio patterns forward and backward.
- **How it works**:
  - Two LSTM layers (forward and backward)
  - Combined hidden states to learn both past and future context
- **Role**: Decodes the time sequence of features from the CNN

---

## 🧱 Model Architecture Summary
nput: MFCC (shape: time_steps x n_mfcc)

→ CNN Layer 1 (Conv2D + ReLU + MaxPool)
→ CNN Layer 2 (Conv2D + ReLU + MaxPool)
→ Flatten
→ BiLSTM (128 units)
→ Dense Layer (Softmax for emotion classification)

Output: Emotion class (e.g., Happy, Sad, Angry, etc.)














| Layer                              | Output Shape   | Description                                                 |
| ---------------------------------- | -------------- | ----------------------------------------------------------- |
| **Input Layer**                    | (100, 40, 1)   | MFCC input features                                         |
| **Conv2D (64 filters)**            | (100, 40, 64)  | Extracts low-level spatial features                         |
| **MaxPooling2D**                   | (50, 20, 64)   | Reduces dimensionality, keeping the most important features |
| **Dropout (0.3)**                  | (50, 20, 64)   | Prevents overfitting                                        |
| **Conv2D (128 filters)**           | (50, 20, 128)  | Extracts higher-level spatial features                      |
| **MaxPooling2D**                   | (25, 10, 128)  | Further reduces dimensionality                              |
| **Dropout (0.3)**                  | (25, 10, 128)  | Regularization                                              |
| **Reshape**                        | (25, -1)       | Prepares data for LSTM                                      |
| **Bidirectional LSTM (128 units)** | (25, 256)      | Learns time dependencies (both forward and backward)        |
| **Bidirectional LSTM (128 units)** | (256)          | Aggregates temporal context                                 |
| **Dense (256 units)**              | (256)          | Fully connected layer with ReLU activation                  |
| **Dropout (0.5)**                  | (256)          | Prevents overfitting                                        |
| **Dense (softmax)**                | (num\_classes) | Output layer with probabilities for each emotion class      |


