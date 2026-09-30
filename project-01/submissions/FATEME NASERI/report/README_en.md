# Credit Card Default Prediction

## 1. Overview

Predict the probability that a credit card customer will default on
payment next month, using an interpretable, business-aware machine
learning pipeline — not just a model that scores well on paper, but
one whose threshold, error costs, and risk tiers are tied to an
actual business decision.

## 2. Dataset

[UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

* 30,000 customers, 24 attributes
* Default rate: **22.12%** (6,636 default / 23,364 non-default)
* No missing values
* `EDUCATION` and `MARRIAGE` contain codes outside the documented
  label set (e.g. undefined categories beyond the official 1–4 /
  1–3 ranges). These are kept as their own category rather than
  dropped — they represent real customers carrying real financial
  data, and dropping them would discard genuine information.
* Zero and negative values in financial columns are meaningful
  (e.g. a credit balance, an early/overpayment) and are **not**
  treated as errors or removed.

## 3. Methodology

```
Raw Dataset
   ↓
Data Inspection
   ↓
EDA
   ↓
Data Cleaning
   ↓
Train / Validation / Test Split
   ↓
Preprocessing Pipeline
   ├── One-Hot Encoding
   ├── Signed Log Transformation
   └── Robust Scaling
   ↓
Feature Engineering
   └── N_DELAY_MONTHS
   ↓
Class Imbalance Handling
   └── class_weight="balanced"
   ↓
5-Fold Stratified Cross-Validation
   ↓
Hyperparameter Selection
   ↓
Train + Validation Retraining
   ↓
Business-based Threshold Selection
   ↓
Final Evaluation on Untouched Test Set
   ↓
Calibration + Risk Groups
   ↓
Logistic Regression vs Decision Tree
```

## 4. Exploratory Data Analysis — Key Findings

* **Gender:** fewer women in the dataset; default rate is 24% for
  women vs. 20% for men.
* **Credit limit (`LIMIT_BAL`):** right-skewed — mean is roughly
  double the median for `BILL_AMT1`, a classic sign of a few very
  high-balance customers pulling the average up. Customers were
  split into 5 credit-limit groups; lower-limit groups show a
  higher default rate.
* **Last payment status (`PAY_0`):** the strongest categorical
  signal. Lowest default rate at status 0/-1/-2 (paid or
  minimum-paid); highest default rate for customers 3–7 months
  overdue.
* **Education:** default rate rises slightly with the `EDUCATION`
  code, but the effect is weak — the gap between sub-groups is
  small. Category 4 corresponds to customers outside the
  documented education labels.
* **Marital status:** the undefined/out-of-range category (3) has
  a default rate close to the "single" group, so no clean
  interpretation could be drawn from this feature alone.
* **Outliers:** `LIMIT_BAL` and `AGE` have under 1% outliers;
  `BILL_AMT*` and `PAY_AMT*` columns have 8–10%. These were **not**
  removed — they represent real customers, and dropping them would
  discard thousands of genuine records.

## 5. Preprocessing

* Removed `ID`
* **One-Hot Encoding** (`drop_first=True`): `SEX_1`, `EDUCATION_1`,
  `MARRIAGE_1` serve as reference categories; test-set columns are
  reindexed to match train exactly (no train/test mismatch)
* **Signed log transformation** on skewed financial variables
  (`LIMIT_BAL`, `BILL_AMT1–6`, `PAY_AMT1–6`)
* **RobustScaler** fit on the training set only — no leakage into
  validation/test

Comparing log-transformed vs. raw features on the initial baseline
model:

| Metric | Log-transform | Raw |
|---|---|---|
| ROC-AUC | 0.73 | 0.70 |
| Accuracy | 0.73 | 0.69 |
| Recall (class 1) | 0.61 | 0.63 |
| F1 (class 1) | 0.50 | 0.47 |

Most metrics improved with the log transform; recall dropped
slightly, which is why accuracy alone was not treated as sufficient
evidence of improvement — see Section 8.

## 6. Feature Engineering

Six candidate features were engineered from the raw payment history:

| Feature | Logistic coef. | LASSO coef. | Note |
|---|---|---|---|
| `N_DELAY_MONTHS` | 0.464 | 0.364 | Count of months (of 6) with an actual payment delay. Strongest engineered predictor. |
| `MAX_DELAY` | 0.322 | 0.085 | Worst single delay in 6 months. Coefficient shrinks heavily under LASSO — overlaps with `PAY_0`/`N_DELAY_MONTHS`. |
| `UTILIZATION` | 0.248 | 0.088 | Average bill / credit limit. Same FICO-style signal used by real credit scorers, but overlaps with `BILL_AMT*`/`LIMIT_BAL` already in the model. |
| `AVG_DELAY` | -0.039 | 0.000 | Average delay across 6 months. Zeroed out entirely by LASSO. |
| `BILL_TREND` | 0.028 | 0.009 | Change in bill amount over time. Weak signal on its own. |
| `PAYMENT_RATIO` | -0.00009 | -0.00007 | Total payments / total bills. A near-linear combination of existing features — negligible added value for a linear model. |

**Only `N_DELAY_MONTHS` was kept.** The other five did not produce a
meaningful, stable improvement, and LASSO's own behavior confirms
this: it zeroed `AVG_DELAY` completely and shrank `MAX_DELAY` /
`UTILIZATION` to a fraction of their unregularized size — a sign of
multicollinearity with features already in the model, not of a weak
model. Re-running the full pipeline with all 6 features vs.
`N_DELAY_MONTHS` alone produced essentially identical PR-AUC
(≈0.50 either way), confirming the simpler feature set is not
leaving performance on the table.

One category worth flagging: `EDUCATION_4` (the undefined-education
group) has a large, unstable coefficient in plain Logistic
Regression (around -0.8 to -0.96 depending on the run) that LASSO
zeroes out — a sign that the coefficient is driven by a small
sub-group sample size rather than a strong, reliable relationship.

## 7. Handling Class Imbalance

Two approaches were compared:

1. **`class_weight="balanced"`** in Logistic Regression / LASSO
2. **SMOTE** (oversampling), applied to the training set only

| Model | PR-AUC (class_weight) | PR-AUC (SMOTE) |
|---|---|---|
| Logistic Regression | ~0.493 | 0.479 |
| LASSO | ~0.493 | 0.48 |

SMOTE **decreased** PR-AUC in both models. `class_weight="balanced"`
was kept going forward — a deliberate choice backed by a direct
comparison, not an assumption.

## 8. Model Selection & Cross-Validation

C values `[0.001, 0.01, 0.1, 1, 10, 100]` were tested via 5-fold
Stratified Cross-Validation:

| C | Mean ROC-AUC |
|---|---|
| 0.001 | 0.7312 |
| 0.01 | 0.7484 |
| 0.1 | 0.7513 |
| 1 | 0.7513 |
| 10 | 0.7513 |
| 100 | 0.7513 |

Performance plateaus from C=0.1 onward — increasing C (weaker
regularization) buys essentially nothing. **C=0.1** was selected:
when the difference is negligible, the simpler (more regularized)
model is preferred.

Cross-validation also showed that switching between Logistic
Regression and LASSO made no practical difference in performance.
**Logistic Regression was chosen as the primary/final model** for
its simplicity and interpretability; **LASSO was kept as a
comparison model** to study the effect of regularization and
feature selection (e.g. confirming which coefficients are
unstable, as with `EDUCATION_4` above).

## 9. Business-Based Threshold Selection

Model output is a probability, not a decision — turning it into one
requires a threshold, and the cost of the two error types is not
equal for this business:

* **False Negative** (calling a risky customer safe) is the more
  expensive error — a customer who defaults on credit they should
  never have received.
* **False Positive** (calling a safe customer risky) costs less —
  a rejected credit increase or loan request.

A 5:1 cost ratio was used:

```
Cost(t) = FP(t) × 1 + FN(t) × 5
```

Searching thresholds from 0.10 to 0.70 (step 0.01) on the
**validation set**, the minimum-cost threshold was **0.37**. It was
then applied once, unchanged, to the untouched test set.

| Threshold | TP | FN | FP | TN | Recall | Precision |
|---|---|---|---|---|---|---|
| 0.30 | 1185 | 142 | 3009 | 1664 | 0.893 | 0.283 |
| **0.37** | 1028 | 299 | 1907 | 2766 | 0.775 | 0.350 |

0.37 was preferred over 0.30: despite lower recall, the drop in
false positives is large enough that total cost is lower under the
defined cost function. Threshold behavior was checked on both
validation and test sets and found consistent — evidence the model
and threshold generalize rather than being tuned to one split.

## 10. Calibration

A calibration curve (10 quantile bins) and **Brier Score** were
used to check whether predicted probabilities match real-world
default rates, not just rank customers correctly.

* **Test Brier Score: 0.1926** (Logistic Regression)

## 11. Final Results (Test Set)

Logistic Regression and a cross-validated Decision Tree (tuned on
`max_depth`, `min_samples_leaf`, `min_samples_split` via PR-AUC)
were retrained on Train+Validation and evaluated once on the
untouched test set:

| Metric | Logistic Regression | Decision Tree |
|---|---|---|
| Best CV PR-AUC | 0.5132 | 0.5277 |
| Test ROC-AUC | 0.7473 | 0.7559 |
| Test PR-AUC | 0.5039 | 0.5180 |
| Test Brier Score ↓ | 0.1937 | 0.1886 |
| Threshold | 0.39 | 0.41 |
| Precision | 0.3478 | 0.3596 |
| Recall | 0.7295 | 0.7189 |
| F1 | 0.4710 | 0.4794 |
| Total Cost ↓ | 3610 | 3564 |

Decision Tree edges out Logistic Regression on most metrics and
total cost; Logistic Regression has a slightly higher recall
(catches marginally more true defaulters). The gap (46 units of
cost) is small relative to the scale of both models — the two are
close, not decisively different, and Logistic Regression's
interpretability remains a real advantage for this use case.

**Reference single-model run** (Logistic Regression, threshold
0.37, matching Section 9): Test ROC-AUC **0.7475**, PR-AUC
**0.5026**, confusion matrix TN=2673, FP=2000, FN=320, TP=1007,
total cost **3600**.

## 12. Risk Segmentation

Predicted probabilities were grouped into Low / Medium / High risk
tiers. Two threshold pairs were tested:

**Thresholds 0.30 / 0.60**

| Group | Count | Share | Default rate |
|---|---|---|---|
| Low | 1,806 | 30.1% | 7.86% |
| Medium | 2,901 | 48.35% | 17.58% |
| High | 1,293 | 21.55% | 52.20% |

**Thresholds 0.37 / 0.70** (aligned with the final selected
threshold)

| Group | Count | Share | Default rate |
|---|---|---|---|
| Low | 3,065 | 51.08% | 9.78% |
| Medium | 2,000 | 33.33% | 24.2% |
| High | 935 | 15.58% | 58.18% |

High-risk customers default at roughly **6.6× the rate** of
low-risk customers under the first split — a clear, monotonic
gradient that supports using this segmentation for prioritization.

## 13. Interpretation

Coefficients were converted to **odds ratios** (exponentiated) for
business-readable interpretation: a ratio above 1 increases the
odds of default, below 1 decreases it, and exactly 1 has no effect.

* **`PAY_0`**: each one-unit worsening of the most recent payment
  status increases the odds of default by roughly **52%**.
* **`SEX_2` (female)** vs. the reference `SEX_1` (male): **9% lower**
  odds of default, holding other features constant.
* **`N_DELAY_MONTHS`** and **`PAY_0`** are the strongest positive
  predictors overall.

These are statistical associations, not causal claims — the model
cannot and does not establish *why* these relationships exist.

## 14. Business Recommendations

* Use this model as a **risk screening and prioritization tool**,
  not an automatic accept/reject system for credit decisions.
* **High-risk customers** (15.58% of the test set, 58.18% observed
  default rate) should receive the most scrutiny; **Medium-risk**
  customers warrant supplementary review; **Low-risk** customers
  can follow the standard process.
* At the final threshold, the model correctly identified 1,007 of
  1,327 true defaulters in the test set — but missed 320. The model
  is useful, not perfect, and should not be the sole basis for a
  credit decision.
* Before real deployment: validate on new/independent data,
  re-check calibration, re-derive the threshold from actual
  business costs (the 5:1 ratio used here is an assumption), and
  monitor performance drift across customer segments and over time.
* Additional data — income, employment status, external debt,
  recent financial changes — would likely improve risk assessment
  beyond what this dataset alone can support.

## 15. Limitations

* Coefficients describe correlation, not causation.
* The 5:1 FN:FP cost ratio is an assumption, not a value derived
  from the business's actual loss data.
* The dataset is a single snapshot (Taiwan, 2005) — it does not
  reflect current economic conditions or a different market.
* PR-AUC of ~0.50 (vs. a 0.22 baseline) is a solid result for a
  linear model on this well-known, high-class-overlap dataset, but
  it is not a "solved" problem — roughly a quarter of true
  defaulters are still missed at the chosen threshold.

## 16. Project Structure

```
credit-card-default-prediction/
├── final_code_preprocessing.py              # Steps 1–5: EDA, cleaning, preprocessing, initial CV
├── train_th0.1_result.py                    # Step 6: C=0.1 results, categorical interpretation, odds ratios
├── p3_trained2_applyfeature1.py             # Step 7: feature engineering (6 candidates)
├── p3_trained2smote.py                      # Step 8: SMOTE comparison
├── p3_trained4_C_tune_featureselection.py   # Step 9: final feature (N_DELAY_MONTHS), C re-tuning, CV
├── p3_trained5_thresh_group_th_compare.py   # Step 9: business threshold, risk segmentation
├── p3_trained6_tree_cv.py                   # Step 10: Decision Tree comparison
├── p3_trained5_thresh_group_th_cv_last.py   # Step 10: final threshold/CV pass
├── default of credit card clients_cleaned.xlsx
└── README.md
```

*(File extensions assumed as `.py` — rename to match your actual files if different.)*

## 17. Setup & Reproducibility

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install pandas numpy scikit-learn imbalanced-learn matplotlib openpyxl
```

Place the dataset in the project root (or update the `path` variable
at the top of each script), then run the scripts in the order listed
in Section 16 — each stage builds on the previous one's conclusions
(e.g. the final feature set and chosen threshold).

## 18. Conclusion

This project implements a complete pipeline: initial inspection,
cleaning, exploratory analysis, preprocessing, feature engineering,
class-imbalance handling, cross-validation, hyperparameter tuning,
business-cost-based threshold selection, calibration, and final
evaluation. In the final comparison, Logistic Regression and a
tuned Decision Tree were evaluated on an identical, untouched test
set; both models identify defaulters with similar overall skill,
but differ in their error patterns and metric trade-offs — a
reminder that model choice should follow the business's evaluation
priorities, not a single aggregate score.
