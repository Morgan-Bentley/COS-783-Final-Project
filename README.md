# COS 783 Final Project
Topic 2:<br>
 Metadata Analysis: AI techniques can assist in analyzing large volumes of metadata by automating pattern recognition, anomaly detection, and correlation analysis. AI algorithms can identify suspicious patterns or outliers that may indicate important evidence.

## Edwin Sen-Hong Chang 20424575
### Anomaly Detection Models Implemented:

#### 1. Credit Card Fraud Detection
- **Dataset:** The dataset consists of credit card transactions made by European cardholders in September 2007. Due to confidentiality issues, features are anonymized.
- **Methodology:** This is an unsupervised learning problem where we utilized the Isolation Forest algorithm from the Scikit-Learn library in Python to detect anomalies (fraudulent transactions).
- **Dataset Source:** [Credit Card Fraud Dataset](https://www.kaggle.com/code/samkirkiles/credit-card-fraud/data) (Kaggle)

#### 2. Phishing URLs Detection
- **Dataset:** The dataset contains URLs that may or may not be classified as phishing sites. Features include `having_IP_Address`, `URL_Length`, `HTTPS`, `SSLfinal_State,Domain_registeration_length`, etc.
- **Methodology:** This is a supervised learning problem where we utilized the Random Forest Classifier from the Scikit-Learn library in Python to detect phishing URLs.
- **Dataset Source:** [Phishing URLs Dataset](https://archive.ics.uci.edu/dataset/327/phishing+websites)

### Implementation Details:

1. **Credit Card Fraud Detection:**
   - **Preprocessing:** Data cleaning, handling missing values, and feature scaling.
   - **Model Training:** Isolation Forest for anomaly detection.
   - **Evaluation:** Confusion matrix, classification report, ROC-AUC score.

2. **Phishing URLs Detection:**
   - **Preprocessing:** Data cleaning, handling missing values, and feature scaling.
   - **Model Training:** Random Forest Classifier for anomaly detection.
   - **Evaluation:** Confusion matrix, classification report, ROC-AUC score, feature importance.

### Results and Evaluation:

- **Credit Card Fraud Detection:**
  - Precision, Recall, F1-Score
  - ROC-AUC Score
  - Confusion Matrix

- **Phishing URLs Detection:**
  - Precision, Recall, F1-Score
  - ROC-AUC Score
  - Feature Importance

### Other Contributions:

- **arffToCSV.py**
  - As the Phishing dataset was in `.arff` format, a solution was found online to translate the `.arff` file to a `.csv` file format for easier implementation. [ARFF to CSV Help Site](https://stackoverflow.com/questions/55653131/converting-arff-file-to-csv-using-python) 

## Morgan Bentley 18103007