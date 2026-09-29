# Credit Card Default Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stefamors/Credit-default-prediction/blob/main/credit_default_prediction.ipynb)

Comparative study of supervised classifiers ...


Comparative study of supervised classifiers for predicting whether a credit card client will default on next month's payment.

**Authors:** Stefano Morselli, Niccolò Poli

## Dataset

[Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (UCI Machine Learning Repository), downloaded at runtime from its Kaggle mirror via `kagglehub`; the data are not redistributed in this repository.

- 30,000 clients of a Taiwanese bank, April–September 2005.
- **Demographics:** sex, age, education, marital status.
- **Repayment status** for each of the six months (paid duly / revolving / *k* months late).
- **Financials:** monthly bill amount and amount actually paid.
- **Target:** `default.payment.next.month` (1 = default). Classes are imbalanced: about 78% non-default and 22% default.

## Methodology

1. **EDA.** Class balance, marginal distributions, and a correlation matrix. Repayment statuses are strongly autocorrelated across months, as are bill amounts. The most recent repayment statuses (September, August) have the highest correlation with the target.
2. **Feature engineering.**
   - `Avg_Utilization` = mean six-month bill (clipped at 0) divided by the credit limit.
   - Undocumented categories of education and marital status are merged into "Others".
   - Sex, education and marital status are one-hot encoded.
3. **Split and scaling.**
   - Stratified 30/70 train/test split (9,000 / 21,000 rows), `random_state=42`.
   - The small training set keeps the RBF-SVM grid search tractable; the large test set lowers the variance of the reported metrics.
   - `StandardScaler` is fitted on the training set only.
4. **Models.** Logistic Regression, Linear SVM, RBF-kernel SVM and Random Forest.
   - All use `class_weight='balanced'`.
   - Hyperparameters are tuned with 5-fold `GridSearchCV`, maximising F1 on the default class, which trades off false alarms (precision) against missed defaults (recall).

## Results (test set, 21,000 clients)

| Model | Best hyperparameters | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | C = 0.01 | 70.0% | 39.1% | 63.4% | 48.4% | 0.725 |
| Linear SVM | C = 0.01 | 70.6% | 39.7% | 63.1% | 48.7% | 0.725 |
| Kernel SVM (RBF) | C = 10, γ = 0.01 | 77.1% | 48.5% | 57.1% | 52.5% | 0.757 |
| **Random Forest** | 100 trees, max_depth = 10, min_samples_leaf = 5 | **80.0%** | **54.7%** | 54.5% | **54.6%** | **0.782** |

- **Linear models** reach the highest recall, but most of their alarms are false positives (precision below 40%).
- **Non-linear models** trade some recall for substantially higher precision.
- **Random Forest** has the best F1 and AUC, and its ROC curve lies above the others over almost the whole FPR range.
- **Feature importance.** Random Forest importance (impurity-based) is dominated by the September and August repayment statuses, followed by `Avg_Utilization`, payment amounts and credit limit. The sex, education and marital-status dummies rank last. Impurity-based importance is biased against binary features, so this ranking should be confirmed with permutation importance.

### Removing demographic dummies

The models were retrained without the sex, education and marital-status dummies (set `DROP_DEMOGRAPHICS = True`; age is kept). Performance was essentially unchanged: Random Forest F1 went from 54.6% to 53.8%.

Removing protected attributes does **not** by itself make the model unbiased. Remaining features can act as proxies for the removed attributes (fairness through unawareness), so an actual fairness assessment would require measuring error rates across groups.

## Reproducing

```bash
pip install -r requirements.txt
jupyter notebook credit_default_prediction.ipynb
```

The notebook also runs as-is on Google Colab. The full grid search, dominated by the RBF SVM, takes several minutes.

## Data citation

Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H (CC BY 4.0)

Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473–2480.
