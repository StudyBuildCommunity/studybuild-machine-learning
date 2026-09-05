# StudyBuild - Project 01: Credit Default Risk Prediction & Risk Segmentation

**Track:** Machine Learning & Explainable AI
**Model:** Logistic Regression
**Dataset:** UCI Default of Credit Card Clients
**Task:** Binary Classification + Risk Segmentation

---

## 1. Business Problem

Credit-card companies need to identify customers who may default on their next payment so that they can prioritize monitoring and risk-management actions.

The objective of this project is to build a reproducible machine-learning workflow that:

* predicts whether a customer will default on the next payment;
* evaluates the model using classification and ranking metrics;
* examines the trade-off between False Positives and False Negatives;
* converts predicted probabilities into Low, Medium, and High risk groups; and
* provides a simple, interpretable model that can support risk-analysis decisions.

This project is an educational prototype and is **not intended to be used as a production credit-scoring system**.

---

## 2. Dataset Source and License

The project uses the **Default of Credit Card Clients** dataset from the UCI Machine Learning Repository.

**Dataset:** Default of Credit Card Clients
**Creator:** I-Cheng Yeh
**UCI Dataset ID:** 350
**DOI:** 10.24432/C55S3H
**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

The dataset contains **30,000 observations and 23 explanatory variables**. It describes credit-card clients and their repayment behavior.

Source:

https://archive.ics.uci.edu/dataset/350/default

Citation:

> Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H.

---

## 3. Target Definition

The target variable is:

`default payment next month`

It is a binary variable:

| Value | Meaning                                       |
| ----- | --------------------------------------------- |
| 0     | Customer does not default on the next payment |
| 1     | Customer defaults on the next payment         |

The target distribution is:

| Class           |  Customers | Percentage |
| --------------- | ---------: | ---------: |
| Non-default (0) |     23,364 |     77.88% |
| Default (1)     |      6,636 |     22.12% |
| **Total**       | **30,000** |   **100%** |

The dataset is therefore imbalanced, with non-default customers representing the majority class.

---

## 4. Variable Overview

The dataset contains several groups of explanatory variables.

### Demographic variables

* `SEX` — customer gender
* `EDUCATION` — education level
* `MARRIAGE` — marital status
* `AGE` — customer age / age group used in the modeling workflow

### Credit information

* `LIMIT_BAL` — amount of given credit

### Repayment status

* `PAY_0`
* `PAY_2`
* `PAY_3`
* `PAY_4`
* `PAY_5`
* `PAY_6`

These variables describe the customer's repayment status over recent months.

### Bill amounts

* `BILL_AMT1`
* `BILL_AMT2`
* `BILL_AMT3`
* `BILL_AMT4`
* `BILL_AMT5`
* `BILL_AMT6`

### Previous payment amounts

* `PAY_AMT1`
* `PAY_AMT2`
* `PAY_AMT3`
* `PAY_AMT4`
* `PAY_AMT5`
* `PAY_AMT6`

The original dataset also contains an `ID` field, which is not used as a predictive feature.

---

## 5. Exploratory Analysis

The exploratory analysis compared default and non-default customers across demographic characteristics, credit information, repayment behavior, bill amounts, and payment amounts.

A key finding was that **recent repayment behavior provides a strong signal of default risk**.

In particular, `PAY_0`, representing the most recent repayment status, showed a clear relationship with default risk. Customers with greater repayment delays tended to have higher predicted default risk.

The analysis also identified some dataset-specific values that required documentation and preprocessing, including unusual category codes in `EDUCATION` and `MARRIAGE`, as well as negative bill amounts.

---

## 6. Preprocessing

The modeling workflow applies preprocessing separately to numerical and categorical variables.

### Categorical preprocessing

Categorical variables are converted to string values and encoded using:

`OneHotEncoder(handle_unknown="ignore")`

For this project:

* `MARRIAGE` categories 0 and 3 were combined as **Other**.
* `EDUCATION` categories 0, 5, and 6 were combined into category **Other (4)**.

### Numerical preprocessing

Numerical variables are converted to numeric values and standardized using:

`StandardScaler()`

### Train / validation / test split

The dataset was divided using a stratified split:

* **70% training:** 21,000 observations
* **15% validation:** 4,500 observations
* **15% test:** 4,500 observations

The split used `random_state=42` to make the workflow reproducible.

Preprocessing is fitted as part of the modeling pipeline using the training data, reducing the risk of data leakage.

---

## 7. Model Choice

### Logistic Regression

Logistic Regression was selected as the baseline model because it is:

* appropriate for binary classification;
* relatively simple and computationally efficient;
* easy to interpret;
* suitable for probability-based risk scoring; and
* useful for explaining the relationship between predictors and predicted default risk.

The model produces a probability of default for each customer. These probabilities are later used for risk segmentation.

The model was trained using standardized numerical variables and one-hot encoded categorical variables.

---

## 8. Model Evaluation

The final test set contains 4,500 observations.

At the default classification threshold of **0.50**, the model produced:

| Metric    | Result |
| --------- | -----: |
| Precision | 0.7071 |
| Recall    | 0.2400 |
| F1-score  | 0.3583 |
| ROC-AUC   | 0.7188 |
| PR-AUC    | 0.4948 |

### Confusion Matrix

|                        | Predicted Non-default | Predicted Default |
| ---------------------- | --------------------: | ----------------: |
| **Actual Non-default** |                 3,405 |                99 |
| **Actual Default**     |                   757 |               239 |

Therefore:

* **TN = 3,405**
* **FP = 99**
* **FN = 757**
* **TP = 239**

Accuracy alone is not sufficient for this problem because the target is imbalanced. A model could obtain relatively high accuracy by primarily predicting the majority class while still missing many actual defaulters.

Precision measures how many predicted defaulters were actually defaulters, while Recall measures how many actual defaulters were successfully identified.

---

## 9. Threshold Comparison

The model's probability threshold affects the balance between Precision and Recall.

Two thresholds were compared:

| Threshold | Precision | Recall |     F1 |  FP |  FN |
| --------: | --------: | -----: | -----: | --: | --: |
|      0.30 |    0.5543 | 0.4608 | 0.5033 | 369 | 537 |
|      0.50 |    0.7071 | 0.2400 | 0.3583 |  99 | 757 |

Reducing the threshold from **0.50 to 0.30** makes the model more sensitive to potential defaults.

This increases Recall from **24.00% to 46.08%** and reduces False Negatives from **757 to 537**.

However, this also increases False Positives from **99 to 369** and reduces Precision from **70.71% to 55.43%**.

Therefore, threshold selection represents a business trade-off between identifying more potential defaulters and avoiding unnecessary interventions for customers who would not default.

---

## 10. Risk-Group Construction

Instead of using only a binary prediction, the model's predicted probabilities were converted into three risk groups:

| Predicted Probability | Risk Group |
| --------------------- | ---------- |
| `< 0.30`              | Low        |
| `0.30 – < 0.50`       | Medium     |
| `>= 0.50`             | High       |

The resulting test-set segmentation was:

| Risk Group | Customers | Defaults | Observed Default Prevalence |
| ---------- | --------: | -------: | --------------------------: |
| Low        |     3,672 |      537 |                      14.62% |
| Medium     |       490 |      220 |                      44.90% |
| High       |       338 |      239 |                      70.71% |

The observed default prevalence increases substantially from Low to Medium to High risk.

This indicates that the model's predicted probabilities provide useful differentiation between groups with different observed levels of default.

The risk thresholds are intended as **business-review cutoffs**, rather than universal or optimal thresholds.

---

## 11. Explainability

Logistic Regression provides interpretable coefficients that indicate the direction of association between predictors and predicted default risk.

For example, the coefficient for `PAY_0` was:

`+0.6538`

This positive coefficient indicates that higher values of the most recent repayment-status variable are associated with higher predicted default risk, holding the other model variables constant.

The largest coefficients from the model included:

| Feature       | Coefficient |
| ------------- | ----------: |
| `EDUCATION_4` |     -0.9626 |
| `PAY_0`       |     +0.6538 |
| `SEX_2`       |     -0.4270 |
| `MARRIAGE_2`  |     -0.3577 |
| `BILL_AMT1`   |     -0.3269 |

Coefficients should be interpreted as **model associations rather than causal effects**. Correlated predictors may also contain overlapping information, making individual coefficient interpretation more difficult.

---

## 12. Business Interpretation

The model can be used as a decision-support tool for prioritizing customers according to predicted risk.

The risk segmentation provides a useful business interpretation:

* **Low-risk customers** have the lowest observed default prevalence and may require less intensive monitoring.
* **Medium-risk customers** have substantially higher observed default prevalence and may benefit from additional monitoring.
* **High-risk customers** have the highest observed default prevalence and could be prioritized for further review.

The threshold comparison also demonstrates that the company can adjust the model's operating point depending on whether it places greater importance on identifying more potential defaulters or reducing unnecessary interventions.

---

## 13. Responsible-ML Considerations

Credit-risk prediction can affect financially significant decisions, so model predictions should not be treated as automatic decisions.

Important considerations include:

* **Fairness:** model performance should be evaluated across relevant demographic groups.
* **Transparency:** customers and decision-makers should not be presented with unexplained automated decisions.
* **Human oversight:** high-impact decisions should include appropriate human review.
* **Data quality:** outdated or biased historical data can produce unreliable predictions.
* **Privacy:** unnecessary personal identifiers should not be included in model outputs.
* **Monitoring:** model performance and calibration should be monitored after deployment.
* **Calibration:** predicted probabilities should be checked to determine whether they correspond appropriately to observed default frequencies.
* **Threshold governance:** classification thresholds should reflect documented business costs and policies.

This project is an educational prototype and should not be used directly to make real-world credit decisions.

---

## 14. Limitations

Several limitations should be considered:

1. The dataset represents a historical population and may not reflect current customer behavior.
2. Logistic Regression provides a useful baseline but may not capture complex nonlinear relationships.
3. The model's Recall at the 0.50 threshold is relatively low.
4. Threshold selection changes the balance between False Positives and False Negatives.
5. Correlated predictors can complicate coefficient interpretation.
6. The model has not been validated on a new external dataset.
7. The risk-group thresholds are business-oriented cutoffs rather than proven optimal thresholds.
8. Model fairness and subgroup performance require additional investigation before deployment.
9. Predicted associations should not be interpreted as causal relationships.

---

## 15. Repository Structure

```text
project-01-credit-default-risk/
│
├── data/
│   └── README.md
│
├── notebook/
│   └── Credit_Default_Risk.ipynb
│
├── outputs/
│   └── risk_groups.csv
│
├── report/
│   └── report.pdf
│
├── README.md
└── requirements.txt
```

The raw dataset may be excluded from the repository depending on project requirements and data-distribution considerations.

---

## 16. How to Run the Project

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd project-01-credit-default-risk
```

### 2. Create a Python environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The main libraries used in the project include:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

### 4. Run the notebook

```bash
jupyter notebook
```

Open:

```text
notebook/Credit_Default_Risk.ipynb
```

Run the notebook cells from top to bottom.

The notebook performs:

1. Data loading and cleaning
2. Exploratory data analysis
3. Feature preprocessing
4. Train/validation/test splitting
5. Logistic Regression training
6. Model evaluation
7. Threshold comparison
8. Risk-group construction
9. Coefficient-based explainability
10. Risk-group output generation

---

## 17. Key Results

The Logistic Regression baseline achieved:

* **ROC-AUC:** 0.7188
* **PR-AUC:** 0.4948
* **Precision at 0.50:** 0.7071
* **Recall at 0.50:** 0.2400
* **F1-score at 0.50:** 0.3583

The risk segmentation showed increasing observed default prevalence:

**Low Risk: 14.62% → Medium Risk: 44.90% → High Risk: 70.71%**

This demonstrates that the model can provide useful probability-based risk differentiation, while also highlighting the importance of threshold selection and responsible use.
