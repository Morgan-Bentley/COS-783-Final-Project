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
### Anomaly Detection Models Implemented:

#### 1. Cyber Threat Detection and Diagnosis from network traffic
- **Dataset:** The dataset contains a collection of data for detecting, diagnosing and mitigating cyber threats using network traffic data
- **Methodology:** This is an unsupervised learning problem where we utilized the Isolation Forest algorithm from the Scikit-Learn library in Python to detect anomalies (fraudulent transactions). For tis problem we used Long Short-Term Memory  networks, a type of Recurrent Neural Network (RNN) architecture that is designed to capture long-term dependencies in sequences of data. Libraries utilized are tensoflow and scikit learn.
- **Dataset Source:** [Cyber Threat Dataset: Network, Text & Relation](https://www.kaggle.com/datasets/ramoliyafenil/text-based-cyber-threat-detection/data) (Kaggle)

### Implementation Details:

1. **Cyber Threat Detection and Diagnosis from network traffic:**
   - **Preprocessing:** Data cleaning, all unlabeled elements were labeled benign, all NaN values replaced with 0.
   - **Model Training:** The model is compiled using the categorical cross-entropy loss function, the Adam optimizer, and accuracy as the metric. It is trained for 10 epochs (iterations) with a batch size of 64, and 20% of the training data is set aside for validation. An EarlyStopping callback stops training if the validation loss doesn't improve for 3 epochs (iterations), helping to prevent overfitting and ensuring the model generalizes well.
   - **Evaluation:** To evaluate the model We used an accuracy function and a loss function over each iteration. A confusion matrix to quanttify correct/incorrect classifications

### Results and Evaluation:

- **Cyber Threat Detection and Diagnosis from network traffic:**
  - Accuracy
  - Loss function
  - Confusion Matrix