# MEPDA: Machine Ensemble Learning for Postpartum Depression Assessment (Applied to Bangladeshi Mothers)

## Overview

Postpartum depression (PPD) is a severe yet underdiagnosed mental health condition affecting women after childbirth — particularly in Bangladesh, where cultural stigma and limited healthcare infrastructure make early detection difficult.

This project develops an interpretable, ensemble machine learning pipeline to predict PPD risk in Bangladeshi mothers using sociodemographic, familial, personal health, and neonatal data. It evaluates multiple classifiers, applies SMOTE to handle class imbalance, builds ensemble models from the top-performing classifiers, and uses LIME to make predictions explainable.

---

## Dataset

The dataset was sourced from a published study on PPD risk factors among Bangladeshi women (n = 800) who gave birth within the past 24 months. Data was collected from hospitals, clinics, health complexes, and rural/residential zones across major cities in Bangladesh. The dataset contains 50+ features across multiple domains:

Sociodemographic factors
- Family & household characteristics
- Pregnancy-related variables
- Neonatal health indicators
- Mental health screening (PHQ-2, PHQ-9, EPDS)

Target Variable: EPDS Result

- 0 → Low Risk
- 1 → Mild Risk
- 2 → High Risk

**Dataset Source:**

Raisa, J. F., & Kaiser, M. S. (2025). *Data for postpartum depression prediction in Bangladesh*. Mendeley Data. https://doi.org/10.17632/4nznnrk8cg.2


---

## Pipeline Summary

Data Cleaning & Encoding  
↓  
Outlier Detection & Removal (IQR Analysis)  
↓  
Balanced Dataset Splitting (80:20 Stratified Split)  
↓  
SMOTE  
↓  
Feature Scaling (StandardScaler)  
↓  
Model Training & Hyperparameter Tuning (GridSearchCV + 5-Fold Stratified Cross-Validation)  
↓  
Ensemble Learning (Hard Voting & Soft Voting using Random Forest + Extra Trees)  
↓  
Model Explainability (LIME)

---
## Models Used

The following machine learning models were evaluated:

- Random Forest
- Extra Trees
- XGBoost
- CatBoost
- Support Vector Machine (SVM)
- Gradient Boosting
- K-Nearest Neighbors (KNN)
- AdaBoost
- Logistic Regression
- Decision Tree


## Results

**Best Final Model:** Hard Voting Ensemble (Random Forest + Extra Trees)  
Achieved a mean cross-validation accuracy of **76.43%**.

---

## Explainable AI (LIME)

Since ensemble models are often considered black-box models, LIME (Local Interpretable Model-Agnostic Explanations) was used to provide local explanations for individual predictions. This improves transparency and supports the potential use of the model in healthcare and clinical decision-making environments.
