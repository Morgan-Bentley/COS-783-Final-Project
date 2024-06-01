import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('datasets/creditcard.csv')

# Display basic information about the dataset
print(data.info())
print(data.describe())

# Preprocess data
data = data.dropna()  # Drop missing values
features = data.drop('Class', axis=1)  # Features are all columns except the target
target = data['Class']  # Target is the 'Class' column

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

model = IsolationForest(contamination=0.001, random_state=42)
model.fit(X_train)

# Predict anomalies
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Convert predictions from {-1, 1} to {0, 1}
y_pred_train = [0 if x == 1 else 1 for x in y_pred_train]
y_pred_test = [0 if x == 1 else 1 for x in y_pred_test]

print("Training Data Classification Report")
print(classification_report(y_train, y_pred_train))

print("Testing Data Classification Report")
print(classification_report(y_test, y_pred_test))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix")
print(conf_matrix)
