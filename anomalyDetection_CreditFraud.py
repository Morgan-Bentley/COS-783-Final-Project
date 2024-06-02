import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# Create results directory
results_dir = 'creditResults'
os.makedirs(results_dir, exist_ok=True)

# Load dataset
data = pd.read_csv('creditcard.csv')

# Display basic information about the dataset
print(data.info())
print(data.describe())

# Visualize the distribution of the target variable
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=data)
plt.title('Class Distribution')
plt.savefig(os.path.join(results_dir, 'class_distribution.png'))
plt.close()

# Visualize data distribution
data.hist(figsize=(20, 20), bins=50)
plt.savefig(os.path.join(results_dir, 'data_distribution.png'))
plt.close()

# Correlation matrix to understand relationships
corr_matrix = data.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig(os.path.join(results_dir, 'correlation_matrix.png'))
plt.close()

# Preprocess data
data = data.dropna()  # Drop missing values
features = data.drop('Class', axis=1)  # Features are all columns except the target
target = data['Class']  # Target is the 'Class' column

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Train Isolation Forest model
model = IsolationForest(contamination=0.001, random_state=42)
model.fit(X_train)

# Predict anomalies
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Convert predictions from {-1, 1} to {0, 1}
y_pred_train = [0 if x == 1 else 1 for x in y_pred_train]
y_pred_test = [0 if x == 1 else 1 for x in y_pred_test]

# Evaluate the model
print("Training Data Classification Report")
print(classification_report(y_train, y_pred_train))

print("Testing Data Classification Report")
print(classification_report(y_test, y_pred_test))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix")
print(conf_matrix)

# Plot Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
plt.close()

# Additional Evaluation Metrics
roc_auc = roc_auc_score(y_test, y_pred_test)
print(f"ROC-AUC Score: {roc_auc}")

precision, recall, _ = precision_recall_curve(y_test, y_pred_test)
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig(os.path.join(results_dir, 'precision_recall_curve.png'))
plt.close()

# Feature importance using permutation importance
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring='accuracy')
sorted_idx = perm_importance.importances_mean.argsort()

plt.figure(figsize=(12,6))
plt.barh(np.array(features.columns)[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.title('Feature Importance (Permutation Importance)')
plt.savefig(os.path.join(results_dir, 'feature_importance.png'))
plt.close()
