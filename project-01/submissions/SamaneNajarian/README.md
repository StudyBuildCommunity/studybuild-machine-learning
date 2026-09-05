# Credit Card Default Prediction

## 📌 Project Overview

This project develops and evaluates machine learning models for predicting whether a credit card customer will default on their payment in the following month.

The analysis focuses on **data exploration, preprocessing, transformation, classification, model evaluation, and risk segmentation**. Since the dataset is imbalanced, multiple evaluation metrics are used instead of relying only on accuracy.

## 📊 Dataset

The dataset contains information on **30,000 credit card customers**, including demographic characteristics, credit limits, bill amounts, payment amounts, and previous payment status.

The target variable is:

* `default payment next month` — whether the customer defaults in the following month.

The target distribution is:

* **Non-default:** 23,364 customers (77.88%)
* **Default:** 6,636 customers (22.12%)

This class imbalance makes metrics such as Recall, F1-score, ROC-AUC, and PR-AUC particularly important.

## 🔍 Exploratory Data Analysis

The analysis included:

* Checking for missing values
* Examining variable distributions
* Detecting potential outliers using the IQR method
* Measuring skewness
* Investigating relationships between variables and default behaviour

Outliers were particularly noticeable in the `PAY_AMT` and `BILL_AMT` variables.

## 🛠️ Data Preprocessing

Different transformations were applied according to the characteristics of the variables.

* `PAY_AMT` variables were transformed using logarithmic transformations.
* `BILL_AMT` variables were transformed using signed square-root transformations where appropriate.
* Numerical and categorical variables were processed separately.
* Categorical variables were encoded using one-hot encoding.
* Ordinal payment-status variables were encoded using ordinal encoding.

The transformations were performed using the training data to avoid data leakage.

## 🤖 Machine Learning Models

Two classification models were developed:

1. **Logistic Regression**
2. **Decision Tree Classifier**

The models were evaluated on the test set using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC
* Confusion Matrix

## 📈 Model Performance

### Logistic Regression

* Accuracy: **0.6757**
* Precision: **0.3657**
* Recall: **0.6353**
* F1-score: **0.4642**
* ROC-AUC: **0.7116**
* PR-AUC: **0.4929**

### Decision Tree

* Accuracy: **0.7503**
* Precision: **0.4500**
* Recall: **0.5795**
* F1-score: **0.5066**
* ROC-AUC: **0.7378**
* PR-AUC: **0.4725**

The Decision Tree achieved stronger overall discrimination and a higher F1-score, while Logistic Regression achieved higher Recall and therefore identified a larger proportion of actual defaulters.

## 🎯 Threshold Optimization and Risk Segmentation

Because the classification threshold affects the balance between Precision and Recall, different thresholds were evaluated.

The threshold of **0.55** produced the highest F1-score:

* Best threshold: **0.55**
* Best F1-score: **0.514**

Customers were subsequently divided into three risk groups based on predicted default probability:

| Risk Group  | Customers | Default Rate |
| ----------- | --------: | -----------: |
| Low Risk    |     3,735 |        12.2% |
| Medium Risk |     1,535 |        26.0% |
| High Risk   |       730 |    **64.5%** |

The substantial difference in observed default rates across the risk groups indicates that the model can meaningfully distinguish customers with different levels of default risk.

## 🔎 Logistic Regression Insights

The logistic regression coefficients were also examined to understand which features contributed most strongly to the predictions.

Among the features with larger absolute coefficients were:

* `EDUCATION_Other`
* `PAY_0`
* `BILL_AMT1`
* `EDUCATION_Graduate school`
* `EDUCATION_University`
* `PAY_AMT2`
* `PAY_AMT1`
* `MARRIAGE_Single`
* `SEX_Female`
* `LIMIT_BAL`

These coefficients provide insight into the statistical relationships learned by the model, but they should not be interpreted as evidence of causality.

## 💼 Management Implications

Customers classified as **High Risk** should receive closer attention, since this group has an observed default rate of **64.5%**.

The model can therefore be used as a **risk-screening and decision-support tool** to prioritize customers for further review or monitoring.

However, the model does not establish that a particular customer will default, nor does it establish that any variable causes default. Predictions should therefore support, rather than replace, professional credit assessment.

Before real-world deployment, the model should be validated using new and representative data and monitored for performance, calibration, stability, and fairness.

## ⚠️ Limitations

* The predictive performance of the models is moderate.
* The dataset is imbalanced.
* Both false-positive and false-negative predictions occur.
* The analysis is based on historical data.
* Model associations should not be interpreted as causal relationships.
* Additional and more recent financial information could potentially improve prediction.

## 🧰 Tools and Technologies

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Jupyter Notebook**

## 📁 Project Structure

```text
Credit-Card-Default/
│
├── Credit_Card_Default.ipynb
├── README.md
└── data/
    └── credit_card.csv
```

## 👩‍💻 Author

**Samane Najarian**

MSc in Mathematical Statistics

This project demonstrates the application of statistical analysis and machine learning techniques to a real-world credit-risk classification problem.
