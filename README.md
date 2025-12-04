# Breast Cancer Diagnosis with Machine Learning

This project uses the Wisconsin Diagnostic Breast Cancer (WDBC) dataset to build
machine learning models that classify tumors as benign or malignant based on
features extracted from digitized images of cell nuclei.

The project implements a full machine learning pipeline:
- Data loading and exploration
- Basic visual analysis (class distribution, histograms, boxplots, correlation heatmap)
- Preprocessing (dropping ID, label encoding, train–test split, feature scaling)
- Training and evaluating two models:
  - Logistic Regression
  - Random Forest Classifier

## Dataset

Breast Cancer Wisconsin (Diagnostic) Data Set from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).


```text
data/wdbc.data
