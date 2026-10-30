# 🚗 Traffic Volume Analysis

## 📌 Project Overview

This project analyzes hourly westbound traffic volume on Interstate 94 (I-94) between Minneapolis and St. Paul, Minnesota. The dataset contains traffic, weather, holiday, and time-related information collected from 2012 to 2018.

The main goal of this project is to explore the factors associated with variations in hourly traffic volume and identify meaningful patterns in traffic behavior across different times, weather conditions, and holidays.

### 📊 Dataset

The dataset contains **48,204 hourly observations** and **8 input features**, with `traffic_volume` as the target variable.

The data were collected from Minnesota Department of Transportation Automatic Traffic Recorder (ATR) station 301, located approximately midway between Minneapolis and St. Paul.

### 🧩 Features

| Feature | Type | Description |
|---|---|---|
| `holiday` | Categorical | U.S. national holidays and the Minnesota State Fair |
| `temp` | Continuous | Average temperature in Kelvin |
| `rain_1h` | Continuous | Amount of rainfall during the hour (mm) |
| `snow_1h` | Continuous | Amount of snowfall during the hour (mm) |
| `clouds_all` | Integer | Percentage of cloud cover |
| `weather_main` | Categorical | Short description of the prevailing weather condition |
| `weather_description` | Categorical | More detailed description of the weather condition |
| `date_time` | Date/Time | Local date and time of the observation |
| `traffic_volume` | Integer | Hourly westbound traffic volume reported by ATR station 301 |


## 📁 Project Structure

```text
.
├── README.md
├── data/
│   ├── Metro_Interstate_Traffic_Volume.csv
│   └── data_cleaning.csv
├── notebooks/
│   ├── traffic-analysis.ipynb
│   └── traffic_volume_prediction.ipynb
└── artifacts/
    ├── evaluation_reports/
    │   ├── cv_folds.csv
    │   ├── legacy_predictions.csv
    │   └── weather_group.csv
    ├── hyperparameter_tuning/
    │   ├── LightGBM_trials.csv
    │   ├── optuna_history.db
    │   ├── RandomForest_trials.csv
    │   ├── tuning_summary.json
    │   └── XGBoost_trials.csv
    └── trained_models/
        ├── LightGBM_optimizer.joblib
        ├── RandomForest_optimizer.joblib
        └── XGBoost_optimizer.joblib
```

- **`data/`** contains the original dataset and the cleaned, feature-engineered dataset used in modeling.
- **`notebooks/`** contains the exploratory analysis and the main notebook covering data preparation, regression models, optimization, and final evaluation.
- **`artifacts/evaluation_reports/`** stores cross-validation, prediction, and weather-group evaluation outputs.
- **`artifacts/hyperparameter_tuning/`** contains Optuna trial results and optimization history for Random Forest, XGBoost, and LightGBM.
- **`artifacts/trained_models/`** contains the saved optimized model objects.

## 🧹 Data Cleaning & Feature Engineering

The raw dataset was cleaned and prepared through a separate reproducible data-cleaning and feature-engineering process.

### 🔧 Data Cleaning

- Merged duplicate hourly records, reducing the dataset from **48,204 to 40,575 observations**.
- Identified and corrected invalid temperature values and converted temperature from **Kelvin to Celsius**.
- Detected and corrected an extreme rainfall outlier (**9,831.3 mm/h**).
- Normalized inconsistent weather labels.
- Removed the unreliable `holiday` feature due to excessive missing values.
- Preserved the original `traffic_volume` target without modification.
- Documented missing time periods and suspicious low traffic values without artificially changing the data.

### ⚙️ Feature Engineering

New features were created to capture important traffic patterns, including:

- Calendar features such as **year, month, weekday, hour, weekend, and season**.
- **Cyclical time features** for hour, weekday, and month.
- Weekday **rush-hour indicators**.
- Weather-related features such as **temperature, freezing, rain, snow, and precipitation**.
- Log-transformed precipitation features.

The final dataset contains **40,575 observations, 23 model features, the `traffic_volume` target, and 4 QA columns**. QA columns are used only for validation and are not used for model training.

📓 **Notebook:** [`traffic_volume_prediction.ipynb` — Section 4](<notebooks/traffic_volume_prediction.ipynb#4>).
📓 **Cleaned Dataset:** `data/data_cleaning.csv`

## 📈 Linear Regression Modeling

The cleaned hourly data were modeled with ordinary least-squares linear regression on the 23 engineered features (numeric inputs standardized, `season` one-hot encoded, the `qa_` columns never used). To respect the time structure of the data, the oldest 80% of the hours (2012-10-02 to 2017-10-26, 32,460 rows) were used for training and the newest 20% (to 2018-09-30, 8,115 rows) for testing, with the scaler and encoder fitted on the training part only. The baseline model reaches a test R² of 0.857 (RMSE ≈ 745 vehicles/hour, against a standard deviation of ≈ 1,985 for the target), with almost identical training and test scores. Residual diagnostics, however, show a clear weakness: the errors follow a daily wave, consecutive hours have strongly correlated errors (Durbin-Watson 0.85), and the precipitation flags are almost perfectly collinear (VIF > 1,000). A single straight line cannot draw the double-peak daily traffic curve, so four optional variants were tested. Adding the second daily harmonic of the hour together with weekday/weekend interactions (V4) clearly gave the best model, cutting the test RMSE by about 19%.

| Model | Model inputs | Test R² | Test RMSE | Test MAE |
|---|---:|---:|---:|---:|
| **A** – Baseline (23 features) | 25 | 0.857 | 744.9 | 559.2 |
| **V1** – Cyclical encoding only (raw `hour`, `month`, `day_of_week` dropped) | 22 | 0.838 | 792.5 | 605.5 |
| **V2** – Log-transformed target (`log1p`) | 25 | 0.620 | 1213.7 | 832.1 |
| **V3** – Trained without the 46 flagged low-volume hours | 25 | 0.857 | 744.6 | 558.9 |
| **V4** – Hour harmonics × weekend interactions | 31 | **0.906** | **603.9** | **426.0** |

*RMSE and MAE are in vehicles per hour, computed on the chronological test set.*

The decision to leave weather labels out of the features was also tested. The 37 textual weather descriptions were merged into six groups with a similar effect on driving conditions (clear/cloudy, reduced visibility such as mist, fog, haze and smoke, light rain/drizzle, heavy rain, snow/ice, and severe storms covering thunderstorms and squalls). Adding these groups to the baseline changed the test RMSE only from 744.9 to 744.0, and a partial F-test found no significant gain (p = 0.30); removing all weather information costs only about 2 RMSE units, so hour of day and weekday dominate hourly traffic in a linear model. These results support the choice to exclude the labels, with two caveats: a linear model cannot capture interactions such as fog mattering more during rush hour, and the severe-storm group has only 239 hours. 📓 **Notebook:** [`traffic_volume_prediction.ipynb` — Sections 5–6](<notebooks/traffic_volume_prediction.ipynb#5>).

## 🌲 Random Forest Modeling

The final notebook section (Section 7) implements **Random Forest regression** using the same chronological training/test split and the same 23 engineered input features as the linear models. The four `qa_` columns are excluded. The 32,460 earlier observations are used for training and the latest 8,115 observations are reserved for testing, so model selection does not use the final test period.

### Pipeline and baseline

The Random Forest pipeline passes numeric columns through without scaling and one-hot encodes the categorical `season` feature. Preprocessing is fitted inside the training pipeline. The baseline uses 200 trees, unlimited tree depth, a minimum of one sample per leaf, and all features available at each split. A fixed random seed (`random_state=42`) makes the run reproducible, while `n_jobs=-1` enables parallel processing.

### Parameter comparison

Five parameter settings were compared on five chronological, calendar-based validation windows contained within the training data. Each validation block is predicted using only observations that come before that block. The windows run from April 2015 through October 2017; the final window ends at the training-period boundary. The table reports the mean across the five windows.

| Configuration | `n_estimators` | `max_depth` | `min_samples_leaf` | `max_features` | Mean R² | Mean MAE | Mean RMSE | RMSE std. dev. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Larger leaves (selected)** | 200 | None | 5 | 1.0 | **0.935** | **300.286** | **499.895** | 54.700 |
| Fewer features | 200 | None | 1 | 0.7 | 0.931 | 313.033 | 513.928 | 55.927 |
| More trees | 400 | None | 1 | 1.0 | 0.929 | 318.566 | 522.809 | 59.532 |
| Baseline | 200 | None | 1 | 1.0 | 0.929 | 319.167 | 523.717 | 59.289 |
| Limited depth | 200 | 25 | 1 | 1.0 | 0.929 | 319.346 | 524.111 | 60.659 |

The selected configuration is the one with the lowest mean validation RMSE: 200 trees, unlimited depth, `min_samples_leaf=5`, and `max_features=1.0`. Increasing the minimum leaf size regularizes the model and improves its validation performance among the tested configurations. This is a small, targeted parameter comparison rather than an exhaustive search.

### Test-set comparison

After selecting the parameters, the model is refit on the full training period and evaluated on the held-out test period. Scores are in vehicles per hour, except R².

| Model | Train R² | Test R² | Test RMSE | Test MAE |
|---|---:|---:|---:|---:|
| Linear Regression A | 0.858 | 0.857 | 744.872 | 559.248 |
| Linear Regression V4 | 0.907 | 0.906 | 603.882 | 425.975 |
| Random Forest baseline | 0.993 | 0.937 | 493.966 | 286.556 |
| **Random Forest tuned** | **0.969** | **0.942** | **476.296** | **270.062** |

The tuned Random Forest achieves the strongest test results among these four models. Its test RMSE is about 21% lower than Linear Regression V4, and its MAE is about 36% lower. Compared with the baseline forest, tuning reduces test RMSE by about 18 vehicles/hour and test MAE by about 16 vehicles/hour. The lower training score and improved test score after increasing `min_samples_leaf` are consistent with reduced overfitting.

### Feature importance and error analysis

Permutation importance measures the increase in test RMSE after shuffling an input column; the individual-feature calculation uses 10 repeats. The largest individual contributions are `hour_cos` (1,513.810 RMSE units), `hour` (506.806), `hour_sin` (212.592), and `is_rush_hour` (157.176). Because these time features overlap in the information they encode, the notebook also shuffles related features as groups. The grouped increases in RMSE are 2,165.29 for hour/commute features and 637.38 for weekday/weekend features. Weather groups have much smaller measured effects in this test evaluation: 30.62 for temperature/freezing, 0.45 for precipitation, and 2.52 for cloud cover. Importance indicates predictive reliance in this model and dataset; it does not establish causation.

The notebook also examines errors by hour, weekday/weekend, and season. On the test set, summer has the lowest seasonal MAE (202.9 vehicles/hour), while winter has the highest (332.0). Weekend bias is −50.1 vehicles/hour and spring bias is −71.0, where bias is defined as actual minus predicted; negative values indicate average overprediction. These breakdowns show that overall scores do not describe performance equally well for every time segment.

📓 **Notebook:** [`traffic_volume_prediction.ipynb` — Section 7](<notebooks/traffic_volume_prediction.ipynb#7>).

## ⚡ XGBoost & LightGBM Modeling

XGBoost and LightGBM were implemented as gradient boosting baselines and evaluated using the same chronological train-test split and 23 engineered input features as the previous models. The preprocessing pipeline passes numeric features through and one-hot encodes `season`, resulting in **25 model inputs after preprocessing**. Both models were evaluated using R², RMSE, and MAE.

### Model setup

Both models used 300 boosting estimators, a learning rate of 0.1, and a fixed random seed (`random_state=42`) for reproducibility. XGBoost used a maximum tree depth of 6, while LightGBM used 31 leaves with no explicit maximum depth. These are baseline configurations; extensive hyperparameter tuning and final model selection are outside the scope of this section.

### Test-set results

| Model | Test R² | Test RMSE | Test MAE |
|---|---:|---:|---:|
| XGBoost baseline | 0.9420 | 474.352 | 274.293 |
| LightGBM baseline | **0.9429** | **470.897** | **269.214** |

LightGBM slightly outperformed XGBoost, achieving a lower RMSE and MAE and a marginally higher R². The performance difference is relatively small, so these results alone do not establish which model should be selected as the final model.

### Feature importance

Permutation importance was calculated on the test set using 10 repeats and the increase in RMSE after shuffling each feature. The results show that **time-related features dominate predictions in both models**.

For XGBoost, the most influential features were `hour_cos` (1,048.44), `hour` (860.91), and `day_of_week` (393.57). For LightGBM, the top three were `hour` (931.91), `hour_cos` (682.82), and `day_of_week` (291.86). The `is_rush_hour` feature had a larger importance score in LightGBM (266.06) than in XGBoost (115.98), while temperature had a moderate contribution in both models.

These findings reinforce the importance of daily and weekly traffic patterns. Permutation importance reflects how much a model relies on a feature for prediction; it does not establish a causal relationship.

📓 **Notebook:** [`traffic_volume_prediction.ipynb` — Section 8](<notebooks/traffic_volume_prediction.ipynb#8>).

## 🔬 Hyperparameter Tuning + Model Evaluation

Building upon the tree-based and gradient boosting baselines, an end-to-end Bayesian optimization and diagnostic framework was implemented in Section 9 (`notebooks/traffic_volume_prediction.ipynb`). This final phase focused on automated hyperparameter tuning using Optuna, artifact persistence, multi-metric model benchmarking, and granular residual diagnostics to select the best production model.

### Optuna Optimization Setup

To maximize predictive accuracy without data leakage, automated hyperparameter tuning was conducted for **XGBoost**, **LightGBM**, and **Random Forest** on the chronological training set (**32,460 observations**) and evaluated on the held-out test period.

- **Optimization Engine:** Optuna configured with a Tree-structured Parzen Estimator (TPE) sampler over **300 trials per model**.
- **Validation Strategy:** **5-fold Time-Series Cross-Validation**, preserving temporal order across every training and validation fold.
- **Preprocessing Consistency:** All candidate models were evaluated on the exact same **25 preprocessed features** to ensure a fair baseline for comparison.
- **Objective Metric:** Minimizing mean Cross-Validation Root Mean Squared Error (**CV RMSE**).

### Optimal Hyperparameter Configurations

Across 300 trials per algorithm, Optuna identified the following optimal configurations:

- **Random Forest (Best Overall):** `bootstrap=True`, `n_estimators=800`, `max_depth=16`, `min_samples_split=9`, `min_samples_leaf=4`, `max_features='sqrt'`, `ccp_alpha=0.0`, `max_samples=0.907022`.
- **XGBoost:** `n_estimators=300`, `max_depth=7`, `learning_rate=0.014517`, `min_child_weight=20`, `subsample=0.693909`, `colsample_bytree=0.820198`, `colsample_bylevel=0.636542`, `gamma=0.05483`, `reg_alpha=0.0`, `reg_lambda=0.114826`, `max_delta_step=0`.
- **LightGBM:** `max_depth=9`, `n_estimators=400`, `num_leaves=53`, `learning_rate=0.013813`, `min_child_samples=71`, `subsample=0.605365`, `subsample_freq=4`, `colsample_bytree=0.600241`, `reg_alpha=0.007411`, `reg_lambda=0.160984`, `min_split_gain=0.125121`.

### Optimized Model Comparison

After tuning, each model was refit on the complete training set (`n_train = 32,460`) and benchmarked on the chronological test set across multiple regression metrics:

| Model | Features | Train R² | Test R² | CV RMSE | Train RMSE | Test RMSE | Test MAE | Test MAPE (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost (Tuned) | 25 | 0.9535 | 0.9441 | 485.54 | 428.94 | 465.84 | 265.25 | 13.42 |
| LightGBM (Tuned) | 25 | 0.9543 | 0.9439 | 484.62 | 425.29 | 466.56 | 263.71 | 13.37 |
| **Random Forest (Tuned)** | **25** | **0.9617** | **0.9446** | **484.27** | **388.94** | **463.76** | **263.44** | **12.59** |

*Note: Exact unrounded values from evaluation logs — Random Forest (`CV RMSE: 484.2659`, `Train RMSE: 388.9369`, `Test RMSE: 463.7611`, `MAE: 263.4351`, `MAPE: 12.5938%`), XGBoost (`CV RMSE: 485.5363`, `Train RMSE: 428.9383`, `Test RMSE: 465.8370`, `MAE: 265.2480`, `MAPE: 13.4167%`), and LightGBM (`CV RMSE: 484.6184`, `Train RMSE: 425.2945`, `Test RMSE: 466.5627`, `MAE: 263.7058`, `MAPE: 13.3677%`). RMSE and MAE are measured in vehicles per hour.*

**Key Takeaways:**
- **Overall Winner:** **Random Forest emerged as the best-performing model** across every evaluation metric, achieving the lowest CV RMSE (**484.27**), lowest Test RMSE (**463.76**), lowest Test MAE (**263.44**), lowest Test MAPE (**12.59%**), and highest Test R² (**0.9446**).
- **Model Convergence:** While LightGBM achieved a slightly lower CV RMSE (**484.62** vs. **485.54**), Test MAE (**263.71** vs. **265.25**), and Test MAPE (**13.37%** vs. **13.42%**), XGBoost edged it out in Test RMSE (**465.84** vs. **466.56**) and Test R² (**0.9441** vs. **0.9439**). Overall, Optuna optimization successfully narrowed the performance gap among the candidates.
- **Artifact Persistence:** All fitted pipelines (`RandomForest_optimizer.joblib`, `XGBoost_optimizer.joblib`, `LightGBM_optimizer.joblib`), SQLite study history (`optuna_history.db`), trial histories (`*_trials.csv`), parameter summaries (`tuning_summary.json`), and evaluation exports (`legacy_predictions.csv`, `weather_group.csv`, `cv_folds.csv`) are systematically organized under `../artifacts/`.

### Optimization Analysis & Residual Diagnostics

To interpret the tuning behavior and verify out-of-sample reliability for the winning **Random Forest** model, two sets of visual diagnostics were generated:

#### 1. Optuna Optimization Visuals
- **Optimization History:** The search started at an initial CV RMSE of approximately **494** and showed rapid improvement during the first **30 to 50 trials**, eventually stabilizing into a reliable best-value plateau at **484.27**.
- **Hyperparameter Importances:** Relative impact analysis across the 300 trials confirmed that **`max_features`** (**score: 0.81 / ~82%**) was overwhelmingly the most influential parameter, followed by **`bootstrap`** (**score: 0.13 / 13%**), together accounting for ~**95%** of the performance variance. **`min_samples_split`** showed a very minor effect (**score: 0.03**), while the remaining parameters—**`ccp_alpha`** (**0.01**), **`max_depth`** (**<0.01**), **`min_samples_leaf`** (**<0.01**), and **`n_estimators`** (**<0.01**)—had negligible isolated impact within their explored ranges. Selecting `max_features='sqrt'` and `bootstrap=True` effectively decorrelated the ensemble and prevented overfitting without relying on aggressive post-pruning (`ccp_alpha=0.0`).
- **Parallel Coordinates:** Multi-dimensional trajectory analysis maps each trial by its Cross-Validation RMSE, where darker lines represent better-performing regions of the search space (reaching **484.27**). Tracing the optimal (yellow) path across the axes highlights the winning interaction: `bootstrap=True`, `max_features='sqrt'`, `n_estimators=800`, `max_depth=16`, `min_samples_split=9`, `min_samples_leaf=4`, `ccp_alpha=0.0`, and `max_samples=0.907`.

#### 2. Six-Panel Residual Diagnostics Dashboard
- **Actual vs. Predicted:** Predictions closely track the 45-degree perfect-prediction line (yellow) across the entire traffic volume range (**0 to 7,000 vehicles/hour**), confirming high accuracy (Test R² = **0.9446**), though slight **heteroscedasticity** is visible as error spread widens above **4,000 vehicles/hour**.
- **Residuals vs. Predicted:** Residuals are centered close to zero overall (mean residual of approximately **-9 vehicles/hour**), indicating no massive systemic bias. Variance expands at higher predicted volumes alongside isolated extreme negative residuals (representing over-predictions during high-volume periods). Conditional error analysis shows non-uniform bias across specific regimes: approximately **-60 vehicles/hour in spring**, **-10 on weekends**, and **-960 on federal holidays**.
- **Residual Distribution & Normal Q-Q Plot:** The error histogram is centered around zero and approximately bell-shaped, but exhibits **leptokurtic behavior (heavy tails)**. In the Normal Q-Q Plot, the middle quantiles align tightly with the theoretical normal line, while the tails deviate significantly (particularly the lower tail dropping sharply), confirming that rare extreme errors occur more frequently than in a strictly normal distribution.
- **Residuals by Hour of Day:** The mean residual (white line) stays flat near zero across the entire 24-hour cycle, proving the model maintains no systematic diurnal bias. However, the error variance (purple shaded area, **±1 std**) widens noticeably during volatile rush hours (**6 AM – 9 AM** and **3 PM – 6 PM**).
- **Last 7 Days of the Test Period:** The chronological overlay for the final test week (**2018-09-24 to 2018-10-01**) confirms exceptional tracking precision across both bimodal weekday rush-hour peaks and unimodal weekend traffic drops.

📓 **Notebook:** [`traffic_volume_prediction.ipynb` — Section 9](<notebooks/traffic_volume_prediction.ipynb#9>).

## 10. Final Model & Final Analysis

### 10.1 Final Model Selection

Random Forest, XGBoost, and LightGBM were tuned and evaluated using time-series cross-validation. Random Forest was selected as the final model because it achieved the lowest cross-validation RMSE. The test-set results of all three models were very close, indicating similar predictive performance.

### 10.2 Final Model Performance

| Metric | Final Random Forest |
|---|---:|
| Test R² | 0.9446 |
| Test RMSE | 463.8 vehicles/hour |
| Test MAE | 263.5 vehicles/hour |
| Test WAPE | 8.0% |
| Test MAPE | 12.59% |

The model explains approximately 94.5% of the variation in traffic volume on the test set.

Compared with the weekday-hour mean baseline, the final model reduced RMSE by **6.8%**. Compared with Linear Regression V4, RMSE decreased by **23.2%** and MAE decreased by **38.1%**.

### 10.3 Feature Importance and SHAP Analysis

SHAP analysis was performed on a sample of 1,500 observations from the 8,115-hour test set to understand the main drivers of the model's predictions.

| Feature group | Mean absolute SHAP contribution | Share |
|---|---:|---:|
| Hour and commute patterns | 1,995.09 | 77.45% |
| Weekday/weekend patterns | 358.49 | 13.92% |
| Month and season | 98.63 | 3.83% |
| Temperature and freezing conditions | 76.72 | 2.98% |
| Year | 26.95 | 1.05% |
| Cloud conditions | 14.76 | 0.57% |
| Precipitation | 5.44 | 0.21% |

Time-of-day and commute patterns were the dominant contributors to the model's predictions, followed by weekday/weekend patterns. These results describe model behavior and should not be interpreted as proof of causal relationships.

### 10.4 Error Analysis by Day and Hour

The model achieved a WAPE of **7.2% on weekdays** and **10.5% on weekends**. Errors varied by hour, with the highest MAE occurring at 07:00 (455.3 vehicles/hour), followed by 16:00 (454.1 vehicles/hour).

| Metric | Weekdays | Weekends |
|---|---:|---:|
| MAE | 256.8 | 280.0 |
| WAPE | 7.2% | 10.5% |

The largest errors were concentrated in specific periods rather than being evenly distributed across the test set. This suggests that unusual traffic patterns and differences between expected and actual conditions remain important challenges.

### 10.5 Holiday Analysis

Holiday periods were a major source of prediction error.

| Metric | Holiday hours | Non-holiday hours |
|---|---:|---:|
| WAPE | 46.9% | 7.2% |
| MAE | 1,089.5 | 240.9 |
| RMSE | 1,603.8 | 388.2 |
| Bias | -954.2 | 15.2 |

Holiday hours accounted for approximately **2.7% of the test set** but contributed around **11% of total absolute error**. All of the 20 largest absolute errors were overpredictions, and 16 occurred during holidays, mainly between 06:00 and 08:00.

This pattern suggests that the model often expected more traffic than actually occurred during these holiday periods. Holiday labels and unusual traffic conditions should therefore be investigated further.

### 10.6 Performance Under Different Conditions

**Weather conditions**

| Weather category | WAPE |
|---|---:|
| Clear/Cloudy | 7.3% |
| Reduced Visibility | 12.0% |
| Light Rain/Drizzle | 6.4% |
| Heavy Rain | 6.5% |
| Snow/Ice | 9.4% |
| Severe Storm | 6.2% |

**Traffic volume levels**

| Actual traffic volume | WAPE |
|---|---:|
| Below 1,000 vehicles/hour | 22.7% |
| 1,000–3,000 vehicles/hour | 16.2% |
| 3,000–5,000 vehicles/hour | 7.0% |
| 5,000 or more vehicles/hour | 5.0% |

The model performed better at higher traffic volumes. Relative errors were substantially higher when traffic volume was low, where even moderate absolute errors can produce large percentage errors. Reduced-visibility conditions also showed weaker performance than clear/cloudy conditions.

### 10.7 Seasonal Performance

| Season | MAE (vehicles/hour) | WAPE |
|---|---:|---:|
| Winter | 327.6 | 10.4% |
| Spring | 261.8 | 7.7% |
| Summer | 195.7 | 5.8% |
| Fall | 273.1 | 8.2% |

The lowest errors occurred in summer, while winter showed the highest MAE and WAPE. Seasonal differences may reflect changes in traffic patterns and operating conditions and warrant further investigation.

### 10.8 Limitations

Despite its strong overall performance, the model has several limitations:

1. **Missing data:** Approximately 22.8% of expected hourly observations are missing, including a gap of roughly ten months. This limits the continuity of the time series.
2. **Limited geographic coverage:** The dataset represents one station and direction on I-94, so the results may not generalize to other roads or locations.
3. **Historical data:** The dataset ends in September 2018 and may not fully represent current traffic patterns.
4. **Holiday-related errors:** Holiday labels and atypical traffic patterns need further validation.
5. **Low-volume predictions:** Relative errors are higher when actual traffic volume is low.
6. **Limited temporal context:** The final model does not include explicit lag features representing traffic in previous hours or on previous days.
7. **Unexpected events and sensor quality:** Sudden traffic changes may reflect unusual events or data-quality issues. These cases should be investigated rather than automatically interpreted as road closures or genuine changes in demand.
8. **Generalization and validation:** Performance should be confirmed on a separate, untouched time period and, where possible, on additional stations.

### 10.9 Recommendations for Future Work

The following improvements could strengthen the model and its practical usefulness:

1. **Improve data quality:** Investigate missing observations, validate holiday labels, and establish consistent data-cleaning rules.
2. **Detect anomalies:** Develop checks for sensor failures, sudden drops, and unusual traffic patterns, using external information where available.
3. **Explore temporal features:** Test lag features, such as traffic volume one hour earlier and 24 hours earlier, provided the data are sufficiently continuous and features are constructed without data leakage.
4. **Improve difficult cases:** Investigate holiday periods, low-volume observations, reduced visibility, and winter conditions separately.
5. **Evaluate additional features:** Test relevant weather, event, and calendar features using a validation strategy that respects chronological order.
6. **Quantify uncertainty:** Consider prediction intervals to communicate the range of plausible traffic volumes rather than relying only on point predictions.
7. **Test generalization:** Evaluate the selected model on an independent future time period and, if possible, data from other stations.

### 10.10 Final Conclusion

The final Random Forest model achieved strong predictive performance on the test set, with an R² of **94.46%**, an RMSE of **463.8 vehicles/hour**, an MAE of **263.5 vehicles/hour**, and a WAPE of **8.0%**. It outperformed the weekday-hour mean baseline and improved substantially over Linear Regression V4.

The analysis showed that hour-of-day and commute patterns were the most influential feature group, followed by weekday/weekend patterns. However, performance was not uniform across all conditions. Holiday periods accounted for a disproportionate share of prediction errors, while low-volume traffic and reduced-visibility conditions presented additional challenges.

**Overall, the model captures routine traffic patterns well, but its predictions are less reliable when traffic deviates from typical patterns.** Improving holiday labels and data quality, detecting potential sensor anomalies, evaluating temporal features, and validating the model on independent data are important next steps toward a more robust traffic forecasting system.

📓 **Notebook:** [`traffic_volume_prediction.ipynb` — Section 10](<notebooks/traffic_volume_prediction.ipynb#10>).

## 👥 Contributors

The project was developed collaboratively. The table below maps the project sections to the team members responsible for them.

| Project Sections | Contributor |
|---|---|
| [1. Metadata](<notebooks/traffic_volume_prediction.ipynb#1>), [2. Exploratory Data Analysis](<notebooks/traffic_volume_prediction.ipynb#2>), [3. Data Visualization](<notebooks/traffic_volume_prediction.ipynb#3>) | [Samane Najarian](https://github.com/SamaneNajarian) |
| [4. Data Cleaning & Feature Engineering](<notebooks/traffic_volume_prediction.ipynb#4>) | [Morteza Abbasi](https://github.com/morteza-codi) |
| [5. Linear Regression](<notebooks/traffic_volume_prediction.ipynb#5>), [6. Weather Groups — Does Removing the Weather Labels Matter?](<notebooks/traffic_volume_prediction.ipynb#6>) | [Fateme Naseri](https://github.com/fatemenaseri) |
| [7. Random Forest](<notebooks/traffic_volume_prediction.ipynb#7>) | [Mona Zareian](https://github.com/mona-az) |
| [8. XGBoost & LightGBM](<notebooks/traffic_volume_prediction.ipynb#8>) | [Maral Farahmandfar](https://github.com/MaralFarahmandfar) |
| [9. Hyperparameter Tuning + Model Evaluation](<notebooks/traffic_volume_prediction.ipynb#9>) | [Alireza Naghdi](https://github.com/Alirezanaghdi-dev) |
| [10. Final Model & Final Analysis](<notebooks/traffic_volume_prediction.ipynb#10>) | [Fateme Naseri](https://github.com/fatemenaseri) |
