# Predicting Hospital Readmission Among Diabetes Patients

## Project Overview

Diabetes is a chronic illness that requires continuous medical care, and hospital readmissions are a common concern for diabetes patients. This project aims to predict the likelihood of hospital readmission among diabetes patients using various machine-learning models, it uses predictive models to explore the factors that influence the likelihood of readmission within 30 days. Given the imbalanced nature of the dataset, where the number of patients not readmitted significantly outweighs the number of patients who are, the focus was on using models that could effectively handle class imbalance. Techniques such as class weighting were applied to improve model performance for the minority class (readmitted patients).

## Data
The dataset used in this project contains the following information:

- Demographics (age, gender, race)
- Admission type and discharge disposition
- Diagnoses (ICD-9 codes, grouped into broader disease categories)
- Medications (insulin, metformin, etc.)
- Number of prior hospital visits (emergency, inpatient, outpatient)
- Outcome (whether the patient was readmitted within 30 days)
- The target variable is binary: whether or not a patient was readmitted within 30 days of their hospital visit.

## Data Preprocessing

Before modeling, the dataset underwent preprocessing to address the following:

- Handling missing values and imbalanced classes.
- Re-coding and collapsing some variables.
- One-hot encoding for categorical variables.


## Models Used

Five different models were evaluated for their ability to predict hospital readmission:

- Logistic Regression
- Random Forest
- XGBoost
- CatBoost
- Bagging Classifier

## Model Evaluation
The models were evaluated based on the following:

- Accuracy
- Precision, Recall, F1-score
- AUC-ROC Curve for binary classification
- Precision-Recall Curve for imbalanced data

## Model Performance

The performance of each model was assessed using the AUC (Area Under the Curve) and the F1 Score for Class 1 (readmitted patients). The following table summarizes the key metrics:

| Model                | AUC   | F1 Score (Class 1) |
|----------------------|-------|--------------------|
| Logistic Regression   | 0.63  | 0.20               |
| Random Forest         | 0.63  | 0.22               |
| XGBoost               | 0.63  | 0.21               |
| CatBoost              | 0.63  | 0.20               |
| Bagging Classifier    | 0.63  | 0.20               |

## Feature Importances

For the Logistic Regression model, feature importance analysis revealed that the most critical factors for predicting readmission were:

1. **Discharge Disposition: Home**: Patients discharged to their homes were less likely to be readmitted, which could be attributed to the likelihood of fewer complications post-discharge or better follow-up care.
2. **Number of Inpatient Visits**: A higher number of inpatient visits increased the likelihood of readmission, suggesting that patients with more complex or recurring health issues had a higher risk.
3. **Primary Diagnosis: Respiratory**: Patients with respiratory diagnoses were more likely to be readmitted, possibly due to complications or chronic respiratory conditions.
4. **Age Groups [50-60] and [0-40]**: Younger patients in these age brackets were at a higher risk of readmission.
5. **Insulin Use**: Patients not using insulin had a higher chance of being readmitted compared to those with steady insulin levels.

## Challenges Faced

One of the key challenges was the high-class imbalance, where the majority of patients were not readmitted, making it difficult to train models effectively for the minority class (Class 1). Initial attempts to use SMOTE (Synthetic Minority Over-sampling Technique) resulted in overfitting, so the models instead leveraged class weighting techniques to handle the imbalance.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

## Conclusion
The Balanced-bagging model emerged as the best-performing model in terms of recall for Class 1, though the AUC scores across all models were similar. Future work could focus on improving model performance through more advanced feature engineering or exploring additional resampling techniques.
