# Interpretable Heart Disease Risk Prediction

A deployed machine learning system that predicts heart disease risk and explains model decisions using SHAP.

![Python](https://img.shields.io/badge/Python-3.12-3776AB)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Portfolio-2563EB)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B)
![SHAP](https://img.shields.io/badge/SHAP-Interpretability-16A34A)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Modeling-F7931E)

An end-to-end machine learning project that predicts whether a patient shows evidence of heart disease from structured clinical measurements, with a deployed Streamlit dashboard for real-time risk estimation.

## Live Demo

Try the deployed Streamlit dashboard here:

https://heart-disease-ml-iynovw3aei45qpkn9jgzcv.streamlit.app

## What This Project Demonstrates

- End-to-end ML pipeline from raw data to deployment
- Model comparison using ROC-AUC, recall, precision, and F1
- Hyperparameter tuning with `GridSearchCV`
- SHAP-based model interpretability
- Real-time predictions through a deployed Streamlit dashboard

## Overview

The goal is to build a practical heart disease risk-screening model that can:

- Predict whether heart disease is present
- Compare an interpretable baseline against a nonlinear model
- Evaluate performance beyond accuracy
- Explain which features influence predictions
- Discuss healthcare deployment risks responsibly

This is not a diagnostic system. It is a machine learning demonstration showing how models can support risk prioritization with clinical oversight.

## Problem Statement

Given patient health features such as age, chest pain type, cholesterol, maximum heart rate, exercise-induced angina, and ST depression, predict whether heart disease is present.

The original UCI target `num` records disease severity. This project converts it into a binary target:

- `0`: no heart disease
- `1`: heart disease present (`num > 0`)

## Dataset

The dataset is the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart%2Bdisease), stored locally at:

```text
data/heart_disease_uci.csv
```

Project dataset shape:

```text
920 rows x 16 columns
```

The `id` column is excluded because it is only an identifier. The `dataset` column is excluded because it represents data source/site rather than patient health information.

## Models Used

- **Logistic Regression:** interpretable baseline with scaled numeric features
- **Random Forest:** nonlinear model that captures interactions and supports feature importance
- **Tuned Random Forest:** final model selected after lightweight `GridSearchCV`

The sklearn pipeline includes:

- Median imputation for numeric features
- Most-frequent imputation for categorical features
- One-hot encoding for categorical features
- Standard scaling for Logistic Regression
- Stratified train-test split and cross-validation

## Results

Held-out test set performance:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.842 | 0.841 | 0.882 | 0.861 | 0.902 |
| Random Forest | 0.842 | 0.841 | 0.882 | 0.861 | 0.913 |
| Tuned Random Forest | 0.848 | 0.849 | 0.882 | 0.865 | 0.915 |

Tuned Random Forest best parameters:

```text
max_depth: 8
min_samples_leaf: 3
n_estimators: 200
```

5-fold cross-validation:

| Model | Mean Accuracy | Mean Recall | Mean F1 |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.826 | 0.855 | 0.844 |
| Random Forest | 0.823 | 0.857 | 0.842 |

Recall is especially important in healthcare screening because a false negative means a patient with possible heart disease may be missed.

## Visual Examples

### ROC Curve

![ROC Curve](outputs/roc_curve.png)

The tuned Random Forest achieves the highest ROC-AUC at `0.915`, showing the strongest class separation on the held-out test set.

### SHAP Feature Summary

![SHAP Summary](outputs/shap_summary.png)

SHAP highlights exercise-induced angina, ST depression, cholesterol, chest pain type, age, and maximum heart rate as major model drivers.

## How to Use the App

1. Open the live demo.
2. Enter patient clinical features.
3. Click **Predict Risk**.
4. View the risk probability, classification, and feature-level explanation.

## Streamlit App

The deployed app provides:

- Patient inputs grouped by clinical category
- Tuned model risk prediction
- Color-coded risk interpretation
- Simple feature-level explanation
- Educational medical disclaimer

Run it locally:

```powershell
streamlit run app.py
```

## ROC-AUC Discussion

ROC curves compare true positive rate against false positive rate across multiple classification thresholds. ROC-AUC summarizes this curve into one score.

This matters in healthcare because the threshold may change depending on the workflow. A screening tool may prefer higher recall to catch more possible disease cases, while a follow-up testing workflow may require higher precision to reduce false alarms.

## SHAP Interpretability

SHAP explains how much each feature pushes model predictions higher or lower for heart disease risk.

Top SHAP drivers in the tuned Random Forest include:

- Exercise-induced angina
- ST depression (`oldpeak`)
- Cholesterol
- Chest pain type
- Age
- Maximum heart rate

Feature importance ranks what the model uses most. SHAP adds direction and magnitude, which is valuable for healthcare AI transparency.

## Key Insights

- Exercise-induced angina is the strongest model driver, aligning with cardiovascular stress patterns.
- Recall is central for healthcare screening because missed high-risk cases can delay follow-up.
- Logistic Regression provides a clear, interpretable baseline for interview discussion.
- Random Forest adds predictive strength by capturing nonlinear feature interactions.
- ROC-AUC above `0.90` across models indicates strong held-out class separation.

## Ethical Considerations

Healthcare ML systems require careful governance:

- **Bias:** A model trained on limited populations may not generalize fairly.
- **False negatives:** Missing true disease cases can delay care.
- **False positives:** Excessive alerts can create anxiety and unnecessary testing.
- **Data drift:** Patient populations and clinical practices change over time.
- **Privacy:** Patient data requires strict security and compliance controls.
- **Clinical oversight:** ML should assist clinicians, not replace diagnosis.

## Limitations

- The dataset is modest in size and combines records from multiple sources.
- Several clinical fields contain missing values and require imputation.
- Important variables such as medications, smoking history, and family history are absent.
- Performance on this dataset does not guarantee real-world clinical generalization.

## Tech Stack

- Python
- scikit-learn
- pandas
- NumPy
- SHAP
- Streamlit
- matplotlib
- seaborn

## Folder Structure

```text
heart-disease-ml/
|-- .streamlit/
|   `-- config.toml
|-- data/
|   `-- heart_disease_uci.csv
|-- outputs/
|   |-- roc_curve.png
|   |-- shap_beeswarm.png
|   `-- shap_summary.png
|-- app.py
|-- notebook.ipynb
|-- train.py
|-- model.pkl
|-- tuned_model.pkl
|-- requirements.txt
`-- README.md
```

## How to Run Locally

Create or activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the full training pipeline:

```powershell
python train.py
```

Open the case-study notebook:

```powershell
jupyter notebook notebook.ipynb
```

Launch the Streamlit dashboard:

```powershell
streamlit run app.py
```

Running `train.py` trains the models, evaluates performance, runs lightweight tuning, saves ROC/SHAP plots, and writes:

```text
model.pkl
tuned_model.pkl
outputs/roc_curve.png
outputs/shap_summary.png
outputs/shap_beeswarm.png
```

## Future Improvements

- Validate the model on an external modern clinical dataset
- Add fairness evaluation across demographic groups
- Calibrate predicted probabilities
- Add confidence intervals for metrics
- Add deployment-ready logging and input validation around the Streamlit app
- Add model monitoring examples for data drift

## Conclusion

This project intentionally avoids deep learning because interpretable classical ML is more appropriate for this dataset size and tabular clinical structure. Early risk estimation systems can help prioritize screening workflows and reduce missed high-risk cases.
