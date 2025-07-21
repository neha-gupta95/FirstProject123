import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load Dataset
data = pd.read_csv('creadit_card_sample.csv')  # your uploaded file

# Step 2: Check and understand data
print(data.head())
print(data['Class'].value_counts())  # 0 = non-fraud, 1 = fraud

# Step 3: Split features and labels
X = data.drop('Class', axis=1)
y = data['Class']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ------------------------------
# ✅ Model 1: Isolation Forest (Unsupervised)
# ------------------------------
print("\n--- Isolation Forest ---")
iso_forest = IsolationForest(contamination=0.001, random_state=42)
y_pred_iso = iso_forest.fit_predict(X_test)

# Isolation Forest returns -1 for anomaly, 1 for normal
# Convert: -1 → 1 (fraud), 1 → 0 (non-fraud)
y_pred_iso = [1 if x == -1 else 0 for x in y_pred_iso]

print(confusion_matrix(y_test, y_pred_iso))
print(classification_report(y_test, y_pred_iso))

# ------------------------------
# ✅ Model 2: Logistic Regression (Supervised)
# ------------------------------
print("\n--- Logistic Regression ---")
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))