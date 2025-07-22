## Introduction

This project employs machine learning models to forecast heart disease risk by analyzing the comprehensive "Indicators of Heart Disease" dataset. The primary objective is to develop and validate predictive models that integrate key risk factors to classify individuals based on their likelihood of developing heart disease. These models aim to identify high-risk individuals for early intervention, establish a user-friendly risk assessment tool, and explore the root causes of heart disease risk to inform data-driven healthcare strategies. 

## Dataset

The dataset is taken from Kaggle: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

Our dataset sources its information from a 2022 CDC (Centers for Disease Control and Prevention) questionnaire. It was collected through telephone surveys interviewing over 400,000 U.S. residents annually about their health status. This dataset consists of 445,132 rows and 40 columns, of which 34 are categorical and 6 are numerical. The dataset is robust and well-distributed across the key parameters, making it suitable for thorough analysis. It includes demographic information (e.g. state, height, weight, age, and BMI) along with details on physical imparity and medical history, such as vaccinations, asthma status, average hours of sleep, and indicators of heart disease, including angina and stroke. The dataset also includes details such as the number of days respondents dedicate each month to maintaining their mental and physical health. The target variable for our models is the column “HadHeartAttack.” Note that there is a significant imbalance in the dataset (which will be addressed) as we calculated that ~94.54% of respondents never had a heart attack, while ~5.46% experienced one.

## Methodology
- Data Cleaning
- One-Hot Encoding for categorical variables
- Min-Max Scaling for numerical variables
- SMOTE for data imbalances in our target variable
- PCA and feature importance for dimentionality reduction
- GridCV for hyperparameter tuning of our ML models
- Model Validation using accuracy and ROC curve metrics

## Important Results of the different ML models

|  | Accuracy Score | ROC AUC Score |
|-----------------|-----------------|-----------------|
| Decision Tree     | 0.94     | 0.73     |
| Random Forest    | 0.71     | 0.76     |
| Logistic Regression    | 0.83     | 0.79     |
| SVM    | 0.93     | 0.72     |
| Deep Learning    | 0.92     | 0.85     |
| LightGBM    | 0.94     | 0.87     |

## Structure of the code

- EDA.ipynb: This file contains some exploratory data analysis on the dataset that helped us to determine feature importance for the ML models.
- ML_Models.ipynb: This file contains all the different ML models we tested.
- LightGBM_Model.ipynb: This file contains the code for our best gradient boosting LightGBM model.
- Webscraping_and_prediction_tool.ipynb: This file contains a ipynb implemented widget based prediction application along with a web scraping tool. 
