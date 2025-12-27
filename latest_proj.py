
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# 📌 Load Dataset
file_path = "ecg.csv"
df = pd.read_csv(file_path)

# 📌 Extract Features & Labels (Assuming last column is label, others are features)
X = df.iloc[:, :-3].values  # ECG signal features
y = df.iloc[:, -1].values   # Labels (0 = Normal, 1 = AFib)
hrv_values = df.iloc[:, -3].values  # HRV
rr_variation_values = df.iloc[:, -2].values  # RR Variation
p_wave_absence_values = df.iloc[:, -1].values  # P-Wave Absence

# 📌 Normalize Data
X = (X - np.mean(X)) / np.std(X)

# 📌 Reshape for CNN
X_cnn = X.reshape(X.shape[0], X.shape[1], 1)

# 📌 Split Data
X_train, X_test, y_train, y_test, hrv_train, hrv_test, rr_train, rr_test, p_wave_train, p_wave_test = train_test_split(
    X_cnn, y, hrv_values, rr_variation_values, p_wave_absence_values, test_size=0.2, random_state=42
)

# ----------------------------------------
# 🔹 Step 1: CNN Model for AFib Detection
# ----------------------------------------
def create_cnn():
    model = models.Sequential([
        layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_cnn.shape[1], 1)),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

cnn_model = create_cnn()

# 📌 Train CNN
history = cnn_model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.2)

# 📌 Evaluate CNN
cnn_acc = cnn_model.evaluate(X_test, y_test)[1]
print(f"CNN Accuracy: {cnn_acc:.2f}")

# 📌 Plot Training History
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('CNN Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('CNN Loss')
plt.legend()
plt.show()

# ----------------------------------------
# 🔹 Step 4: Confusion Matrix & Evaluation
# ----------------------------------------

# 📌 Predict on Test Set
y_pred = (cnn_model.predict(X_test) > 0.5).astype("int32")

# 📌 Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# 📌 Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=['Normal', 'AFib'], yticklabels=['Normal', 'AFib'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - CNN AFib Detection")
plt.show()

# 📌 Classification Report
print(classification_report(y_test, y_pred, target_names=['Normal', 'AFib']))

# ----------------------------------------
# 🔹 Step 2: Symbolic AI (Fuzzy Logic for AFib)
# ----------------------------------------

# 📌 Define Fuzzy Variables
hrv = ctrl.Antecedent(np.arange(0, 200, 1), 'HRV')
rr_variation = ctrl.Antecedent(np.arange(0, 0.3, 0.01), 'RR_Variation')
p_wave = ctrl.Antecedent(np.arange(0, 1, 0.1), 'P_Wave_Absence')
afib_risk = ctrl.Consequent(np.arange(0, 1, 0.1), 'AFib_Risk')

# 📌 Membership Functions
hrv.automf(3)
rr_variation.automf(3)
p_wave.automf(3)
afib_risk['low'] = fuzz.trimf(afib_risk.universe, [0, 0.2, 0.4])
afib_risk['medium'] = fuzz.trimf(afib_risk.universe, [0.3, 0.5, 0.7])
afib_risk['high'] = fuzz.trimf(afib_risk.universe, [0.6, 0.8, 1.0])

# 📌 Define Fuzzy Rules
rule1 = ctrl.Rule(hrv['poor'] & rr_variation['good'], afib_risk['high'])
rule2 = ctrl.Rule(hrv['average'] & p_wave['good'], afib_risk['medium'])
rule3 = ctrl.Rule(rr_variation['poor'] & p_wave['average'], afib_risk['high'])

# 📌 Create Control System
afib_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
afib_decision = ctrl.ControlSystemSimulation(afib_ctrl)

def check_afib(hrv_value, rr_value, p_wave_absence):
    afib_decision.input['HRV'] = hrv_value
    afib_decision.input['RR_Variation'] = rr_value
    afib_decision.input['P_Wave_Absence'] = p_wave_absence
    afib_decision.compute()
    return afib_decision.output.get('AFib_Risk', 0)

# ----------------------------------------
# 🔹 Step 3: Hybrid AI - CNN + Fuzzy Logic
# ----------------------------------------
def hybrid_afib_detection(ecg_signal, hrv_value, rr_value, p_wave_absence):
    cnn_pred = cnn_model.predict(ecg_signal.reshape(1, X_cnn.shape[1], 1))[0, 0]
    symbolic_pred = check_afib(hrv_value, rr_value, p_wave_absence)
    
    if cnn_pred > 0.5 and symbolic_pred > 0.7:
        return "High AFib Risk 🚨"
    elif cnn_pred > 0.5 or symbolic_pred > 0.5:
        return "Moderate AFib Risk ⚠️"
    else:
        return "Low AFib Risk ✅"

risk_levels = []
for i in range(len(X_test)):
    risk_level = hybrid_afib_detection(X_test[i], hrv_test[i], rr_test[i], p_wave_test[i])
    risk_levels.append(risk_level)

plt.figure(figsize=(10, 5))
sns.countplot(y=risk_levels, order=['Low AFib Risk ✅', 'Moderate AFib Risk ⚠️', 'High AFib Risk 🚨'], palette='coolwarm')
plt.title("AFib Risk Level Distribution")
plt.xlabel("Count")
plt.ylabel("Risk Level")
plt.show()

plt.figure(figsize=(6, 6))
plt.pie([risk_levels.count("Low AFib Risk ✅"), risk_levels.count("Moderate AFib Risk ⚠️"), risk_levels.count("High AFib Risk 🚨")], 
        labels=['Low', 'Moderate', 'High'], autopct='%1.1f%%', colors=['green', 'orange', 'red'])
plt.title("AFib Risk Proportion")
plt.show()

print("Patient Test Results:")
for i, risk in enumerate(risk_levels):
    print(f"Patient {i+1}: {risk}")







