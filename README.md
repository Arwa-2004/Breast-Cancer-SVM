# Breast Cancer Detection with Support Vector Machine (SVM)
<img width="1084" height="568" alt="image" src="https://github.com/user-attachments/assets/b6d8ec3a-10a5-4487-9ef6-28fcf451adb4" />

In this project I used the Breast Cancer Wisconsin dataset from scikit-learn to build a machine learning model that classifies tumors as malignant or benign using a Support Vector Machine (SVM).

## Dataset 📊 
- `load_breast_cancer()` from `sklearn.datasets`
- 569 samples, 30 features
- Binary classification: 0 = benign, 1 = malignant

## What This Notebook Covers
- Exploratory Data Analysis (EDA) with seaborn/matplotlib
- Data preprocessing with pandas
- SVM model training with scikit-learn
- Hyperparameter tuning using GridSearchCV
- Visualization of results (confusion matrix, PCA, boxplots, etc.)

## Model Performance
- Accuracy: 95%
- High precision and recall on both classes
- Very low false negative rate (important in medical diagnosis)

## Libraries Used

![NumPy](https://img.shields.io/badge/NumPy-1.24.3-blue)
![pandas](https://img.shields.io/badge/pandas-2.0.3-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6.1-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12.2-red)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.1-green)
![SVM](https://img.shields.io/badge/SVM-Imbalance%20Handling-blue)

## 💻 How to Run
```bash
pip install -r requirements.txt
jupyter notebook breast_cancer_svm.ipynb
