# 📊 Flipkart Customer Support CSAT Prediction

This machine learning project aims to **predict customer satisfaction (CSAT) scores** based on customer interaction data collected by Flipkart's support channels. The goal is to build a robust, production-ready ML pipeline that can help improve support quality and proactively flag dissatisfied customers.

---

## 🚀 Project Overview

- **Domain:** E-Commerce / Customer Service
- **Objective:** Predict whether a customer is satisfied (CSAT ≥ 4) or not using historical support data.
- **Tech Stack:** Python, Pandas, Scikit-learn, SMOTE, Matplotlib, Seaborn, GridSearchCV, Joblib

---

## 🧠 Problem Statement

Given a dataset of customer interactions, predict the **binary CSAT score**:
- 1 → Satisfied (CSAT ≥ 4)
- 0 → Not Satisfied (CSAT < 4)

This can help:
- Identify agents or cities causing dissatisfaction
- Optimize support processes
- Increase retention by responding to dissatisfaction early

---

## 🧱 Project Pipeline

1. **Data Cleaning & Preprocessing**
   - Handling null values, duplicates, and inconsistent formats
   - Text cleaning (NLP) for "Customer Remarks"

2. **Feature Engineering**
   - Extracting derived features like `price_per_minute`
   - Label encoding and one-hot encoding for categorical variables

3. **Balancing Data**
   - Used **SMOTE** (Synthetic Minority Over-sampling) to handle class imbalance

4. **Model Building**
   - ✅ Logistic Regression
   - ✅ Random Forest Classifier
   - ✅ Gradient Boosting Classifier

5. **Evaluation Metrics**
   - Accuracy, Precision, Recall, F1 Score, ROC AUC
   - Model Explainability via Feature Importance

6. **Hyperparameter Tuning**
   - Applied **GridSearchCV** for Logistic, RF, and GB models

7. **Model Deployment Readiness**
   - Saved final model (`best_random_forest_model.joblib`)
   - Reloaded and tested on unseen data ✅

---

## 📈 Final Results

| Model                | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---------------------|----------|-----------|--------|----------|----------|
| Logistic Regression | ~0.84    | ~0.83     | ~0.85  | ~0.84    | ~0.88    |
| Random Forest (final) | ✅ **0.90+** | ✅ **0.91+** | ✅ **0.89+** | ✅ **0.90+** | ✅ **0.93+** |
| Gradient Boosting   | ~0.88    | ~0.89     | ~0.86  | ~0.87    | ~0.91    |

---

## 📁 Folder Structure

