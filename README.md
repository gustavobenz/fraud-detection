# 💳 Credit Card Fraud Detection with Machine Learning

## 📌 Overview

This project focuses on building machine learning models to detect fraudulent credit card transactions in a highly imbalanced dataset.

Fraud detection is a critical problem in financial systems, where the challenge lies in identifying rare fraudulent events without negatively impacting legitimate users.

---

## 🎯 Objective

Develop and evaluate predictive models capable of detecting fraudulent transactions while handling class imbalance and optimizing performance using appropriate evaluation metrics.

---

## 📊 Dataset

* Source: Kaggle - Credit Card Fraud Detection
* Transactions: 284,807
* Fraud cases: 492 (~0.17%)

The dataset is highly imbalanced, making traditional accuracy metrics unreliable.

---

## ⚙️ Approach

### 1. Data Preparation

* Train/test split with stratification
* Feature scaling using StandardScaler
* Handling class imbalance using SMOTE

### 2. Models Trained

* Logistic Regression
* Random Forest
* XGBoost

### 3. Evaluation Metrics

Due to class imbalance, the following metrics were used:

* Precision
* Recall
* F1-score
* ROC AUC
* PR AUC (most relevant)

---

## 📈 Results

| Model               | Precision (Fraud) | Recall (Fraud) | PR AUC   | Notes                |
| ------------------- | ----------------- | -------------- | -------- | -------------------- |
| Logistic Regression | Low               | High           | Moderate | High false positives |
| Random Forest       | Medium            | High           | Good     | Balanced performance |
| XGBoost             | **0.73**          | **0.85**       | **0.86** | Best trade-off       |

### 🔍 Key Insight

The XGBoost model achieved the best balance between detecting fraud and minimizing false positives:

* False Positives: 31
* False Negatives: 15

---

## 💡 Business Interpretation

Fraud detection requires balancing:

* **Recall** → detect as many frauds as possible
* **Precision** → avoid blocking legitimate transactions

A model with high recall but low precision can harm user experience, while a model with high precision but low recall can miss fraud.

The final model achieves a strong trade-off between these two objectives.

---

## ⚙️ Technical Notes

* SMOTE significantly improved fraud detection (recall)
* Dataset size increased after resampling, impacting training time
* Random Forest required optimization (reduced trees and depth) for faster iteration

---

## 🔄 Threshold Tuning

Model performance can be adjusted by changing the classification threshold:

* Lower threshold → higher recall (more fraud detected)
* Higher threshold → higher precision (fewer false positives)

This allows alignment with business priorities.

---

## 🚀 Future Improvements

* Hyperparameter tuning (Grid Search / Bayesian Optimization)
* Real-time fraud detection pipeline
* Model monitoring and drift detection
* Feature engineering and anomaly detection approaches

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* XGBoost
* Matplotlib

---

## 📂 Project Structure

```
fraud-detection/
│
├── fraud_detection.ipynb
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

1. Clone the repository:

```
git clone https://github.com/your-username/fraud-detection.git
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the notebook:

```
jupyter notebook fraud_detection.ipynb
```

---

## 👨‍💻 Author

Developed as part of a Data Science portfolio project focused on fraud detection and machine learning in imbalanced datasets.


