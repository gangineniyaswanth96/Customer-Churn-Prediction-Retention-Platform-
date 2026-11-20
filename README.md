# Customer Churn Prediction & Retention Platform

A telecom company is losing customers (**churn**). This project predicts which customers are likely to leave, so the company can act early with retention offers (discounts, tech support, contract upgrades), and explains **why** each customer is at risk.

- **Problem type:** binary classification (`Churn` = 1 left, 0 stayed)
- **Dataset:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): 7,043 customers, 21 columns (demographics, phone and internet services, contract, billing)
- **Metrics:** recall on the churn class and ROC-AUC. Missing a churner (a lost customer) costs more than an unneeded retention offer, and with only 26.5% churn, accuracy is misleading.

## Results

All models are evaluated on the same held-out test set: 1,409 customers (20%, stratified split).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Lazy (predict all stay) | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.806 | 0.658 | 0.556 | 0.603 | 0.842 |
| Logistic Regression (balanced) | 0.740 | 0.507 | 0.786 | 0.616 | 0.842 |
| Decision Tree | 0.730 | 0.495 | 0.807 | 0.614 | 0.830 |
| Random Forest | 0.762 | 0.536 | 0.759 | 0.628 | 0.845 |
| XGBoost | 0.752 | 0.522 | 0.783 | 0.627 | 0.840 |
| **XGBoost (tuned)** | 0.750 | 0.519 | **0.810** | **0.633** | **0.846** |

Recall is the share of real churners the model catches. The full table is saved in `data/model_results.csv`.

5-fold stratified cross-validation on the training set (ROC-AUC):

| Model | Mean | Std |
|---|---|---|
| Logistic Regression (balanced) | 0.846 | 0.012 |
| Random Forest | 0.846 | 0.011 |
| XGBoost | 0.844 | 0.011 |

**Takeaways**
- A model that predicts "nobody churns" gets 73.5% accuracy but catches 0 churners. That is why accuracy is not used to choose models.
- Class weights (`class_weight="balanced"`, or `scale_pos_weight` in XGBoost) raise recall from 0.556 to about 0.79, at the cost of lower precision.
- The models are essentially **tied**: the cross-validation differences (0.002 at most) are much smaller than one standard deviation. Tuning XGBoost with `RandomizedSearchCV` raised its CV ROC-AUC only from 0.844 to 0.849. On this dataset, a simple and interpretable Logistic Regression performs as well as tuned XGBoost.
- **Saved model:** tuned XGBoost (`models/xgb_tuned.joblib`). It has the highest test recall and ROC-AUC (by a small margin), catching 303 of the 374 churners in the test set, and SHAP can explain each of its predictions.

## What drives churn

Found in the EDA and confirmed by the Logistic Regression coefficients, tree feature importance and SHAP:

- **Contract** is the strongest driver: month-to-month customers churn 42.7%, one-year 11.3%, two-year 2.8%.
- **Tenure:** new customers churn the most (median tenure is 10 months for churners vs. 38 for customers who stayed).
- **Internet type:** fiber optic customers churn 41.9%, DSL 19.0%.
- **Support services:** customers without tech support churn 41.6% vs. 15.2% with it (online security shows the same pattern).
- **Payment method:** electronic check payers churn 45.3%, compared with 15 to 19% for the other methods.
- **Gender** has no effect (26.9% vs. 26.2%).

**Retention ideas:** push long-term contracts, offer free tech support to new fiber customers, and encourage automatic payment.

## Notebooks

| Notebook | What it covers |
|---|---|
| `01_eda.ipynb` | Loading, cleaning (`TotalCharges` stored as text, 11 missing values), EDA, encoding, train/test split, scaling. Saves `data/telco_clean.csv`. |
| `02_baseline_model.ipynb` | Logistic Regression baseline, why accuracy misleads, confusion matrix, precision, recall, F1, ROC-AUC, class weights, decision threshold, coefficients. Creates `data/model_results.csv`. |
| `03_tree_models.ipynb` | Overfitting in an unlimited Decision Tree, a controlled Decision Tree, Random Forest, XGBoost, feature importance. Adds its models to `data/model_results.csv`. |
| `04_tuning_shap.ipynb` | 5-fold cross-validation (with a Pipeline for Logistic Regression to avoid leakage), XGBoost tuning with `RandomizedSearchCV`, SHAP explanations for all customers and for individual customers. Saves the final model to `models/`. |

Every code cell has a Markdown cell before it that explains what it does and why.

## Project structure
```
data/
  telco_dataset.csv        original data (never modified)
  telco_clean.csv          cleaned and encoded data (notebook 01)
  model_results.csv        test-set scores of every model (notebooks 02 to 04)
models/
  xgb_tuned.joblib         final tuned XGBoost model (notebook 04)
  feature_columns.joblib   column order the model expects
notebooks/
  01_eda.ipynb
  02_baseline_model.ipynb
  03_tree_models.ipynb
  04_tuning_shap.ipynb
src/                       reusable code (later stages)
requirements.txt
```

## How to run

**1. Install the packages**
```bash
pip install -r requirements.txt
```
On macOS, XGBoost also needs the OpenMP runtime. Without it, `import xgboost` fails with "XGBoost Library (libxgboost.dylib) could not be loaded".
```bash
brew install libomp
```

**2. Run the notebooks in order (01, 02, 03, 04).** Each one uses files created by the one before it. Open them in VS Code or Jupyter and choose **Run All**, or run them from the project root:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_baseline_model.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_tree_models.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_tuning_shap.ipynb
```

Notebook 02 rebuilds `data/model_results.csv` from scratch, so if you re-run it, re-run 03 and 04 afterwards to add their models back. Notebooks 03 and 04 can be re-run on their own without creating duplicate rows.

Tested with Python 3.12.11, pandas 2.3.1, NumPy 2.2.6, scikit-learn 1.7.2, XGBoost 3.1.2 and SHAP 0.54.1 on macOS (Apple Silicon).

## Roadmap
- [x] Data cleaning, EDA and data preparation (Lessons 1 to 4)
- [x] Logistic Regression baseline and evaluation on imbalanced data (Lesson 5)
- [x] Tree-based models: Decision Tree, Random Forest, XGBoost (Lesson 6)
- [x] Cross-validation, hyperparameter tuning and SHAP explanations (Lesson 6B)
- [ ] Neural network (MLP), compared honestly with XGBoost on the same test set (Lesson 7)
- [ ] Sequence deep learning (LSTM/GRU) on behavioral data, e.g. the KKBox dataset (Lesson 8)
- [ ] Retention platform: prioritize customers by churn risk and customer value, map segments to actions, FastAPI and Streamlit dashboard, MLflow (Lesson 9)
