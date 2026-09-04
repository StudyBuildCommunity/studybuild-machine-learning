# Data Description and Initial Data Quality Checks

## Dataset Overview

The dataset used in this project contains **30,000 credit card client records** and is used to predict whether a client will default on their next payment.

The dataset contains **23 explanatory variables** and one binary target variable:

- `0`: No default
- `1`: Default

An `ID` column is also included as a unique identifier for each record.

## Missing Values and Duplicate Records

Initial data quality checks showed that:

- No missing values were found in the dataset.
- No duplicate rows were identified.

Therefore, no missing-value imputation or duplicate removal was required at this stage.

## Variable Types and Encodings

Although all variables are stored numerically, not all of them should be interpreted as continuous numerical variables. Some columns use numerical values as category encodings.

### Sex

The `SEX` variable contains two valid categories:

- `1`: Male
- `2`: Female

Only these two values were observed in the dataset.

### Education

According to the provided variable description:

- `1`: Graduate school
- `2`: University
- `3`: High school
- `4`: Others

However, the dataset also contains the values `0`, `5`, and `6`, which are not explicitly defined in the provided variable description. These values are therefore documented as unusual or undocumented encodings and will be interpreted cautiously during the analysis.

### Marital Status

According to the provided variable description:

- `1`: Married
- `2`: Single
- `3`: Others

The dataset also contains the value `0`, which is not explicitly defined in the provided variable description. This value is therefore documented as an unusual or undocumented encoding.

### Repayment Status

The repayment-status variables are:

- `PAY_0`
- `PAY_2`
- `PAY_3`
- `PAY_4`
- `PAY_5`
- `PAY_6`

The provided variable description defines repayment status values such as:

- `-1`: Pay duly
- `1`: Payment delay for one month
- `2`: Payment delay for two months
- ...
- `8`: Payment delay for eight months
- `9`: Payment delay for nine months and above

However, the dataset also contains the values `-2` and `0`, which are not explicitly defined in the provided variable description. These values are therefore documented and will require cautious interpretation during further analysis.

## Numerical Value Checks

The numerical summaries showed no clearly impossible values for variables such as `AGE` and `LIMIT_BAL`.

For example:

- `AGE` ranges from 21 to 79 years.
- `LIMIT_BAL` ranges from 10,000 to 1,000,000 NT dollars.

These values do not indicate obvious data-entry errors based on the initial inspection.

## Bill Statement Amounts

The `BILL_AMT1` to `BILL_AMT6` variables represent monthly bill statement amounts.

Some negative values were observed in these variables. At this stage, these values are not assumed to be errors and will not be removed. Their distributions and potential interpretation will be explored further during Exploratory Data Analysis (EDA).

## Previous Payment Amounts

The `PAY_AMT1` to `PAY_AMT6` variables represent previous payment amounts.

Some of these variables contain very large maximum values compared with their median values, suggesting that their distributions may be right-skewed and contain extreme observations.

However, large payment amounts are not automatically considered invalid or erroneous. Their distributions will be investigated further during EDA.

## Initial Data Quality Conclusion

The initial data quality checks indicate that the dataset is complete and contains no duplicate records.

Several unusual or undocumented numerical encodings were identified, particularly in:

- `EDUCATION`
- `MARRIAGE`
- `PAY_*` repayment-status variables

In addition, negative bill amounts and extreme payment amounts were observed. These values are not removed at this stage because unusual or extreme values do not necessarily indicate data errors.

Further investigation of variable distributions, class differences, skewness, and extreme observations will be conducted during the Exploratory Data Analysis stage.