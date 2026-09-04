# Credit Card Default Risk Prediction & Risk Segmentation

An end-to-end, interpretable Machine Learning project that analyzes customer demographic data, billing histories, and payment behaviors to predict credit card default risk and translate model probabilities into a Low / Medium / High risk segmentation for analytical use.

> **Note:** This is an educational risk-modeling prototype built for a Machine Learning & Explainable AI course project. It is **not** an approved credit-scoring system and should not be used for real credit decisions without further validation (see [Limitations](#limitations--responsible-ml-considerations)).

---

## Table of Contents
1. [Business Problem](#business-problem)
2. [Dataset](#dataset)
3. [Variable Overview](#variable-overview)
4. [Data Quality & Class Distribution](#data-quality--class-distribution)
5. [Preprocessing](#preprocessing)
6. [Model](#model)
7. [Evaluation](#evaluation)
8. [Threshold Comparison](#threshold-comparison)
9. [Risk Segmentation](#risk-segmentation)
10. [Explainability](#explainability)
11. [Business Interpretation (Management Summary)](#business-interpretation-management-summary)
12. [Limitations & Responsible ML Considerations](#limitations--responsible-ml-considerations)
13. [Repository Structure](#repository-structure)
14. [How to Run](#how-to-run)

---

## Business Problem

The risk-analytics team of a credit-card company needs to identify customers who are more likely to default on their next payment, so that review and risk-management actions can be prioritized. This project builds a simple, interpretable classification model (Logistic Regression) and converts predicted default probabilities into a basic **Low / Medium / High** risk segmentation for analytical use.

The goal is to demonstrate classification, class-imbalance-aware evaluation, threshold thinking, simple explainability, and responsible interpretation — not to produce a production credit-scoring system.

## Dataset

* **Source:** [UCI Machine Learning Repository – Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
* **DOI:** https://doi.org/10.24432/C55S3H
* **License:** CC BY 4.0
* **Records:** 30,000 credit card clients in Taiwan, April–September 2005
* **Features:** 23 explanatory variables + `ID` + 1 binary target (`default payment next month`)

## Variable Overview

| Variable | Description |
|---|---|
| `LIMIT_BAL` | Amount of given credit (NT dollar), individual + family/supplementary credit |
| `SEX` | 1 = male, 2 = female |
| `EDUCATION` | 1 = graduate school, 2 = university, 3 = high school, 4 = others, 0/5/6 = undocumented (regrouped to "Unknown") |
| `MARRIAGE` | 1 = married, 2 = single, 3 = others, 0 = undocumented ("Unknown") |
| `AGE` | Age in years |
| `PAY_0` to `PAY_6` | Repayment status, most recent (`PAY_0`, September 2005) to oldest (`PAY_6`, April 2005). `-2`/`-1` = paid in full / no consumption, `0` = revolving credit (minimum payment), `1`–`8` = months delayed |
| `BILL_AMT1`–`BILL_AMT6` | Monthly bill statement amount (NT dollar), September to April 2005 |
| `PAY_AMT1`–`PAY_AMT6` | Monthly previous payment amount (NT dollar), September to April 2005 |
| `default payment next month` *(target)* | 1 = default, 0 = no default |

Full data-quality notes (missing values, duplicates, undocumented encodings, numerical range checks) are documented separately in [`data/README.md`](data/README.md).

## Data Quality & Class Distribution

* **Missing values:** none across all 30,000 records.
* **Duplicate rows:** none identified.
* **Target distribution:** ~77.9% non-default (23,364), ~22.1% default (6,636) — a moderate class imbalance that requires evaluation beyond Accuracy (Precision, Recall, F1, ROC-AUC, PR-AUC).
* **Undocumented categorical values:** `EDUCATION` contained values `0`, `5`, `6` outside the documented 1–4 range; `MARRIAGE` contained value `0` outside the documented 1–3 range. Both were investigated and handled during preprocessing (see below).
* **Repayment status:** values `-2` and `0` in `PAY_0`–`PAY_6` are not explicitly defined in the original UCI documentation but were confirmed to represent valid domain-specific statuses (no balance / revolving credit), not missing data.
* **Numerical ranges:** `AGE` (21–79) and `LIMIT_BAL` (10,000–1,000,000 NT$) showed no impossible values. `BILL_AMT*` contained some negative values and `PAY_AMT*` showed heavy right-skew with extreme maxima; neither was treated as an error and both were explored further during EDA.

Key EDA findings that informed modeling:
* Recent repayment status (`PAY_0`) is the single strongest and clearest predictor of default — clients with status `0` (revolving credit) have a low default rate (~13%), while clients with delays (`2+`) show default rates often above 60–70%.
* Recent payment amounts (`PAY_AMT*`) are substantially lower for defaulting clients across all six months, while bill amounts (`BILL_AMT*`) differ only slightly between groups.
* Demographic variables (`SEX`, `EDUCATION`, `MARRIAGE`) show only weak separation between default and non-default clients, suggesting they act as secondary rather than primary predictors.
* `LIMIT_BAL` and `AGE` are right-skewed / weakly separating, respectively.

## Preprocessing

1. **Undocumented categories:** `EDUCATION` values `5` and `6` were regrouped into code `0`, together with the pre-existing undocumented `0`, forming a single "Unknown" category. `MARRIAGE`'s undocumented `0` was kept as-is and documented as "Unknown." This is a fixed, rule-based transformation applied before the train/test split, so it introduces no data leakage.
2. **Feature/target separation:** `ID` was dropped (unique identifier, no predictive value). The target (`default payment next month`) was separated into `y`; all remaining columns form `X`.
3. **Encoding:** The nominal variables `SEX`, `EDUCATION`, `MARRIAGE` were one-hot encoded (`drop_first=True`) to avoid implying a false ordinal relationship. `PAY_0`–`PAY_6` were kept in their original numeric/ordinal form, since their values represent a genuine degree of payment delay.
4. **Train/test split:** Stratified 70/30 split (`random_state=42`) to preserve the ~78/22 class ratio in both sets.
5. **Scaling:** Continuous numerical features (`LIMIT_BAL`, `AGE`, `PAY_0`–`PAY_6`, `BILL_AMT1`–`6`, `PAY_AMT1`–`6`) were standardized with `StandardScaler`, fit **only** on the training set and applied to both train and test, preventing test-set information from leaking into preprocessing. One-hot encoded dummy variables were left unscaled.

## Model

**Logistic Regression** (scikit-learn, default hyperparameters, `random_state=42`) was used as the required baseline classifier — chosen for its interpretability and suitability for the Explainable AI track. Both class predictions (`predict`) and default probabilities (`predict_proba`) were generated; the probabilities are used for threshold analysis and risk segmentation.

## Evaluation

At the default 0.5 threshold:

| Metric | Value |
|---|---|
| Accuracy | 0.81 |
| Precision (default class) | 0.70 |
| Recall (default class) | 0.24 |
| F1-score (default class) | 0.36 |
| ROC-AUC | 0.716 |
| PR-AUC (Average Precision) | 0.498 |

**Why Accuracy is not enough:** a naive "always predict no default" model would score ~78% accuracy without any predictive value, since defaults represent only ~22% of clients. At the default threshold, the model also misses a large share of true defaulters (1,514 false negatives out of 1,991 actual defaulters). PR-AUC (0.498) is far above the PR-AUC of a random classifier (~0.221, the positive class rate), confirming the model carries real signal despite low recall at this threshold.

## Threshold Comparison

To avoid tuning the threshold on the test set, classification thresholds were compared using out-of-fold probabilities from 5-fold cross-validation on the training data:

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.2 | 0.33 | 0.70 | 0.45 |
| 0.3 | 0.57 | 0.47 | **0.52** |
| 0.4 | 0.64 | 0.39 | 0.48 |
| 0.5 | 0.71 | 0.25 | 0.37 |
| 0.7 | 0.74 | 0.04 | 0.08 |

Threshold **0.3** (maximizing F1) was selected and evaluated once on the untouched test set, confirming consistent performance (Precision = 0.562, Recall = 0.456, F1 = 0.503). Compared to the default 0.5 threshold, this reduces missed defaulters from 1,514 to 1,084 false negatives, at the cost of increasing false positives from 204 to 707.

**Business trade-off (false negatives vs. false positives):** a missed defaulter (false negative) typically causes a direct financial loss on the outstanding credit, while a false positive mainly incurs an operational cost (manual review of a client who was never actually at risk). Given this likely cost asymmetry, the shift toward higher recall at threshold 0.3 is favorable, and Recall / PR-AUC — not Accuracy or Precision alone — should be the metrics prioritized for ongoing monitoring. This reasoning is a simplifying assumption appropriate for an educational project; real deployment would require the company's finance and risk teams to quantify the actual costs involved.

## Risk Segmentation

Using predicted default probabilities (at the threshold points established above), clients are segmented into three risk groups:

| Risk Group | Probability Range | Clients | Observed Default Rate |
|---|---|---|---|
| Low | < 0.22 | 5,502 | 13.0% |
| Medium | 0.22 – 0.30 | 1,884 | 19.5% |
| High | ≥ 0.30 | 1,614 | 56.2% |

The observed default rate increases monotonically from Low to High, confirming the segmentation is practically meaningful — the High risk group shows more than **4x** the default rate of the Low risk group and should be prioritized for closer review.

## Explainability

Standardized Logistic Regression coefficients were used to interpret feature influence. The most recent repayment status (`PAY_0`) shows the strongest and most reliable positive association with default risk, consistent with the EDA findings. `BILL_AMT1` and `PAY_AMT2` show the largest negative coefficients.

The largest raw coefficients belong to the `EDUCATION` dummy variables, but this is likely a statistical artifact: the dropped baseline category (`EDUCATION = "Unknown"`) is a very small group (345 clients) that happened to show a 0% observed default rate, inflating the apparent effect of every other education category rather than reflecting a genuine real-world relationship.

**Important caveat:** these coefficients describe statistical association, not causation. Several predictors are correlated with one another (e.g., `PAY_0`–`PAY_6`, `BILL_AMT1`–`6`), which can make individual coefficients unstable — conclusions should focus on broad, robust patterns (recent repayment behavior matters most) rather than precise magnitudes for any single feature.

## Business Interpretation (Management Summary)

**Which clients require closer review:** The High risk group (1,614 clients, ~18% of the test set) shows a 56.2% observed default rate — more than four times the portfolio average (~22.1%) — and should be prioritized for closer review and proactive risk-management action. The Medium risk group (1,884 clients, 19.5%) warrants lighter monitoring, while the Low risk group (5,502 clients, 13.0%) requires minimal intervention.

**Supporting evidence:** ROC-AUC of 0.716 and PR-AUC of 0.498 (well above the random baseline of ~0.221), a clear and monotonic increase in observed default rate across risk groups, and a top predictor (recent repayment status) that is consistent between the EDA and the model coefficients.

**What the model cannot establish:** Causal drivers of default; stable individual feature effects in the presence of correlated predictors; generalization to economic conditions or client populations different from Taiwan, 2005; and literal, calibrated probability-of-default values.

## Limitations & Responsible ML Considerations

* **Historical, geographically limited data:** the dataset reflects Taiwan credit clients between April and September 2005; financial conditions and client behavior may have changed substantially since then and elsewhere.
* **No causal claims:** model coefficients describe predictive association only; they must not be interpreted as causal effects.
* **Not an approved credit score:** this is a classroom risk-segmentation prototype, not a production credit-scoring system.
* **Sensitive variables:** the dataset includes demographic attributes (`SEX`, `EDUCATION`, `MARRIAGE`, `AGE`). Using such variables in real credit decisions raises ethical and governance concerns (potential discriminatory impact); subgroup performance and fairness analysis would be required before any real-world use.
* **Multicollinearity:** several groups of features (repayment statuses across months, bill amounts across months) are highly correlated, which can make individual coefficient interpretation unstable.
* **No probability calibration:** predicted probabilities have not been calibrated and should not be read as literal default likelihoods without further validation.
* **No external validation:** the model has not been tested on data from other time periods, institutions, or geographies.
* **Threshold and risk-group boundaries** were derived from cross-validated model performance and a simplifying cost assumption (false negatives more costly than false positives), not from real business cost data — a real deployment would need the finance/risk teams to supply actual costs.

## Repository Structure

```text
credit-default-risk/
├── README.md                     # This file
├── requirements.txt               # Python dependencies (exact versions used)
├── data/
│   ├── default of credit card clients.xls
│   └── README.md                  # Initial data-quality checks and variable 
├── notebooks/
│   └── credit_risk_analysis.ipynb # Full reproducible analysis: EDA → preprocessing →
│                                   # baseline model → evaluation → threshold analysis →
│                                   # risk segmentation → explainability
├── figures/                       # Saved copies of all key visualizations
├── outputs/
│   └── risk_groups.csv            # client_id, predicted_probability, risk_group
└── report/
    └── management_summary.md      # Standalone business-facing summary (Q10)
```

## How to Run

1. Clone the repository and place the dataset file (`default of credit card clients.xls`) inside `data/`.
2. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter and run the notebook top to bottom:
   ```bash
   jupyter notebook notebooks/credit_risk_analysis.ipynb
   ```
   (Use "Restart Kernel and Run All" to ensure a clean, reproducible execution — the notebook uses fixed random seeds throughout.)
4. Outputs are written to `figures/` (visualizations) and `outputs/risk_groups.csv` (final risk-group table).