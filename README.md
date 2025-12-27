# Atrial-Fibrillation-detection-using-Neuro-Symbolic-AI

## 📌 Project Overview

This project focuses on the **detection of Atrial Fibrillation (AFib)** from ECG signals using a **hybrid Artificial Intelligence approach** that combines **Deep Learning (CNN)** with **Symbolic / Rule-based reasoning**.
The objective is to improve **accuracy, interpretability, and clinical relevance** compared to using a neural network alone.

---

## 🎯 Problem Statement

Atrial Fibrillation is one of the most common cardiac arrhythmias and can lead to severe complications such as stroke and heart failure if not detected early. Traditional methods either rely purely on machine learning (black-box models) or rule-based medical criteria. This project bridges both approaches.

---

## 💡 Proposed Solution

The system uses:

* **Convolutional Neural Networks (CNN)** to learn complex ECG patterns automatically.
* **Symbolic features** such as:

  * Heart Rate Variability (HRV)
  * RR interval variation
  * Absence of P-wave
* A **decision fusion mechanism** where neural predictions are combined with symbolic indicators to improve reliability.

---

## 🧠 Key Features

* ECG signal preprocessing and normalization
* CNN-based binary classification (AFib / Normal)
* Extraction of clinically relevant ECG features
* Confusion matrix and classification report analysis
* Improved explainability compared to pure deep learning models

---

## 🛠️ Technologies & Libraries Used

* **Python**
* **NumPy & Pandas** – data handling and preprocessing
* **TensorFlow / Keras** – CNN model implementation
* **Scikit-learn** – train-test split and evaluation metrics
* **Matplotlib & Seaborn** – visualization
* **scikit-fuzzy (skfuzzy)** – symbolic reasoning support

---

## 📂 Dataset

* ECG dataset in CSV format
* Contains:

  * ECG-derived features
  * HRV values
  * RR interval variation
  * P-wave presence/absence
  * AFib labels (0: Normal, 1: AFib)

> Dataset is split into training and testing sets using an 80–20 ratio.

---

## ⚙️ Model Workflow

1. Load and preprocess ECG data
2. Normalize input features
3. Train CNN model on ECG features
4. Extract symbolic features
5. Combine CNN output with symbolic indicators
6. Evaluate model using accuracy, precision, recall, and confusion matrix

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## 🚀 Results

The hybrid model demonstrates:

* Better robustness than CNN alone
* Improved interpretability due to symbolic rules
* Strong potential for clinical decision support systems

---

## 🔮 Future Enhancements

* Integration with real-time ECG acquisition devices
* API-based deployment for healthcare systems
* Extension to mobile or wearable platforms
* Inclusion of fuzzy logic–based decision fusion
* Larger and more diverse ECG datasets

---

## 🧑‍💻 How to Run the Project

1. Clone the repository
2. Install required Python libraries
3. Place ECG dataset in the project directory
4. Run the main Python script
5. View evaluation results and plots

