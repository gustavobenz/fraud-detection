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
* Handling class imbalance using SMOTE (applied only on train set to avoid data leakage)

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

### 4. Cross-Validation

5-fold stratified cross-validation using an SMOTE + XGBoost pipeline to ensure reliable performance estimates across different data splits.

### 5. Model Export

Trained model and scaler serialized with `joblib` for deployment in a real-time scoring pipeline.

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

## 💼 Business Impact

Fraud detection involves a direct financial trade-off:

| Event | Estimated Cost |
|---|---|
| Undetected fraud (False Negative) | Full transaction value lost + chargeback fee (~$25–$50) |
| Legitimate transaction blocked (False Positive) | Customer friction, potential churn (~$5–$15 per incident) |

**With XGBoost at threshold 0.5 (per 56,962 test transactions):**

* 15 frauds missed → estimated loss: ~15 × avg fraud amount
* 31 legitimate transactions blocked → estimated friction cost: ~31 × $10 = $310

Lowering the threshold to 0.3 catches more fraud but increases blocked legitimate transactions — the right threshold depends on the cost structure of each business.

This model enables data-driven threshold selection rather than arbitrary rule-based blocking.

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


