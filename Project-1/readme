# 👥 Employee Attrition Analyzer

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)]()

An end-to-end Machine Learning pipeline to predict employee turnover risk using organizational, financial, and workplace sentiment data from the **IBM HR Analytics** dataset.

---

## 📌 Executive Summary

Employee turnover disrupts team dynamics, slows down delivery, and introduces substantial replacement costs. This project builds a complete predictive analytics pipeline designed to help People Ops and HR leaders proactively flag flight-risk employees before they leave.

### Key Objectives:
- Clean and prepare mixed tabular data (handling high cardinality, constant values, and scaling).
- Tackle real-world target imbalance (~16% attrition rate).
- Train and compare **Logistic Regression**, **Decision Tree**, and **Random Forest** classifiers.
- Evaluate models using multi-metric diagnostics (**Confusion Matrix**, **F1 Score**, **Recall**, and **ROC-AUC**).
- Identify and rank top organizational churn drivers to provide actionable retention strategies.

---

## 📊 Dataset Profile

- **Dataset:** IBM HR Analytics Employee Attrition & Performance (`WA_Fn-UseC_-HR-Employee-Attrition.csv`)
- **Total Records:** 1,470 employees
- **Total Attributes:** 35 features
- **Target Label:** `Attrition` (`Yes` = 1, `No` = 0)
- **Class Breakdown:**
  - **Retained (No):** 1,233 (~83.9%)
  - **Departed (Yes):** 237 (~16.1%)

---

## 🛠️ Architecture & Pipeline

```text
  Raw CSV Data
       │
       ▼
 [Data Cleaning] ──► Drop zero-variance & ID fields (StandardHours, Over18, etc.)
       │
       ▼
 [Preprocessing] ──► One-Hot Encoding (pd.get_dummies) + StandardScaler
       │
       ▼
  [Train/Test]   ──► 80/20 Stratified Partitioning (random_state=42)
       │
       ├──────────────────────┬──────────────────────┐
       ▼                      ▼                      ▼
Logistic Regression     Decision Tree          Random Forest
       │                      │                      │
       └──────────────────────┼──────────────────────┘
                              ▼
                 [Model Evaluation & Plots]
                 ├── Confusion Matrices
                 ├── F1 / Recall / Accuracy Metrics
                 └── Feature Importance Analysis
