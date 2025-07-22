# Heart Disease Risk Prediction and Resource Recommendation

## 1. Introduction and Key Objectives

Heart disease remains a leading global cause of mortality, influenced by a complex interplay of demographic, health, lifestyle, comorbid, and behavioral factors. This project employs machine learning models to forecast heart disease risk by analyzing the comprehensive [Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease) dataset.

**Objectives:**

- Develop and validate predictive models integrating key risk factors to classify individuals based on their likelihood of developing heart disease.
- Identify high-risk individuals for early intervention.
- Establish a user-friendly heart attack risk assessment tool.
- Explore root causes of heart disease risk to inform data-driven healthcare strategies.
- Utilize text analysis, web scraping, and geographical mapping to examine interactions between heart attacks and contributing factors.
- Recommend trusted, related health resources.

---

## 2. Dataset Description

The dataset originates from the 2022 CDC questionnaire, collected via telephone surveys of over 400,000 U.S. residents annually. It contains:

- **Rows:** 445,132
- **Columns:** 40 (34 categorical, 6 numerical)
- Includes demographic info (state, height, weight, age, BMI), physical impairments, medical history (e.g., vaccinations, asthma), sleep habits, and heart disease indicators (e.g., angina, stroke).
- Target variable: `HadHeartAttack` (imbalanced: ~5.46% positive cases).

[Dataset on Kaggle](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)

---

## 3. Exploratory Data Analysis (EDA) and Key Findings

- **Geographical Impact:** States like Arkansas and West Virginia show high physical activity days and heart attack rates. West Virginia has one of the highest average BMIs.
- **Demographics:** Heart attack rates increase with age and are higher in males. Over 70% of heart attack patients had BMI above the healthy CDC range (18.5-24.9).
- **Disease Correlations:** Angina has the highest correlation with heart attacks (0.45), followed by stroke (0.18) and diabetes (0.15).
- **Health Correlations:** Poor general health correlates with more physical and mental health issue days.
- **West Virginia Deep Dive:** Text analysis of a 120+ page CDC report confirmed angina, obesity, age, physical inactivity, and alcohol use as key factors influencing heart attacks in West Virginia.

---

## 4. Data Cleaning and Preprocessing

- Dropped duplicates and NaN values.
- Binary mapping for categorical columns with two categories (e.g., Yes/No).
- Target encoding for high-cardinality categorical variables.
- Standard scaling to normalize features.
- Addressed class imbalance with SMOTE oversampling.
- Dimensionality reduction with PCA applied for logistic regression to reduce overfitting.

---

## 5. Methods and Machine Learning Models

- **Hyperparameter Tuning:** GridSearchCV used for optimizing models.
- **Models Tested:**
  - Decision Tree Classifier (94% accuracy; prone to overfitting)
  - Random Forest Classifier (71% accuracy; ensemble method)
  - Logistic Regression (83% accuracy; interpretable)
  - Support Vector Machine (SVM) (93% accuracy)
  - Deep Learning Neural Network (91% accuracy)
  - LightGBM (93.7% accuracy after tuning with 14 key features)

- **Key Notes on LightGBM:**
  - Fast, memory-efficient gradient boosting.
  - Hyperparameters such as `learning_rate`, `num_leaves`, and regularization (`lambda_l1`, `lambda_l2`) were tuned.
  
- **Model Explainability:**
  - LIME used to explain predictions of the deep learning model.
  - Angina and age identified as the most important predictors.

---

## 6. Application: Personalized Heart Attack Risk Prediction Tool

- Utilizes LightGBM model with 14 curated health attributes.
- Users input health info (age, sex, BMI, lifestyle, medical history).
- Outputs binary risk prediction (high risk if probability > 0.5).
- Designed for quick and user-friendly individual risk assessment.

---

## 7. Application: Heart Health Information and Resources Retrieval Tool

- Uses Selenium and BeautifulSoup to scrape trusted CDC website.
- Automatically fetches relevant articles and guidelines on heart health.
- Allows personalized searches for up to five health factors.
- Provides direct links to reliable CDC resources for user education.

---

## 8. Important Results of the different ML models

|  | Accuracy Score | ROC AUC Score |
|-----------------|-----------------|-----------------|
| Decision Tree     | 0.94     | 0.73     |
| Random Forest    | 0.71     | 0.76     |
| Logistic Regression    | 0.83     | 0.79     |
| SVM    | 0.93     | 0.72     |
| Deep Learning    | 0.92     | 0.85     |
| LightGBM    | 0.94     | 0.87     |

## 9. Structure of the code

- EDA.ipynb: This file contains some exploratory data analysis on the dataset that helped us to determine feature importance for the ML models.
- ML_Models.ipynb: This file contains all the different ML models we tested.
- LightGBM_Model.ipynb: This file contains the code for our best gradient boosting LightGBM model.
- Webscraping_and_prediction_tool.ipynb: This file contains a ipynb implemented widget based prediction application along with a web scraping tool. 

- Required files:  
  - `heart_2022_with_nans.csv`  
  - `us-states.json`  
  - `cvh_burden_2010.pdf`  
  - Code notebook: **Prediction Tool and Web Scraping Recommendations (Use Colab)**

- **Note:** Run the notebook on [Google Colab](https://colab.research.google.com/) for best compatibility.

