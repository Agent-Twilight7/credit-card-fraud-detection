# credit-card-fraud-detection
Anomaly Detection in Credit Card Transactions using Isolation Forest
# Credit Card Fraud Detection using Machine Learning

## Overview
This project focuses on detecting fraudulent credit card transactions using anomaly detection techniques, specifically the **Isolation Forest** algorithm. The primary goal is to accurately identify fraudulent transactions in an imbalanced dataset where only a small fraction of transactions are fraudulent. 

By leveraging unsupervised learning methods, this project demonstrates how anomaly detection can be applied to identify outliers without relying on labeled data.

## Dataset
- **Source**: [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Description**: The dataset contains 284,807 transactions made by European cardholders over two days in September 2013. Out of these, only 492 transactions are fraudulent (~0.17%).
- **Features**:
  - `V1, V2, ..., V28`: Anonymized numerical features obtained via PCA transformation.
  - `Time`: Seconds elapsed between the transaction and the first transaction in the dataset.
  - `Amount`: Transaction amount.
  - `Class`: Binary label (0 for legitimate, 1 for fraud).

## Project Highlights
- Implemented an **Isolation Forest** model for anomaly detection.
- Preprocessed the data by standardizing features for optimal model performance.
- Achieved **74.18% accuracy** on detecting fraudulent transactions.
- Evaluated model performance using metrics like **Precision**, **Recall**, **F1-Score**, and **ROC-AUC**.

## Files in this Repository
- **credit-card-fraud-detection.ipynb**: The Jupyter Notebook containing the complete code and analysis.
- **isolation_forest_model.pkl**: Serialized model file saved using `pickle` for reuse.
- **requirements.txt**: List of required Python packages for replicating the environment.
- **data_preprocessing.py**: Python script for loading and preprocessing the dataset.
- **model_training.py**: Python script for training the Isolation Forest model.
- **evaluation.py**: Python script for evaluating the model's performance.
- **README.md**: Documentation of the project.

## How to Run
### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```
### 2. Install required libraries
`pip install -r requirements.txt`
### 3. Run the Jupyter notebook
`jupyter notebook credit-card-fraud-detection.ipynb`
### 4. (Optional) Run the scripts
```
# Data preprocessing
python data_preprocessing.py

# Model training
python model_training.py

# Model evaluation
python evaluation.py
```

