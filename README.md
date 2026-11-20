# Customer Churn Prediction & Retention Platform

Predicts which customers of a telecom company are likely to cancel their service (**churn**), and explains why, so the company can act early with retention offers such as discounts, tech support or contract upgrades.

- **Dataset:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): 7,043 customers, 21 columns (demographics, services, contract, billing). 26.5% of the customers churned.
- **Problem type:** binary classification (`Churn` = 1 left, 0 stayed)
- **Metrics:** recall on the churn class and ROC-AUC. Missing a churner costs more than an unneeded offer, and with imbalanced classes accuracy is misleading.

## Notebooks
1. `01_eda.ipynb`: cleaning, exploratory analysis, encoding, train/test split, scaling
2. `02_baseline_model.ipynb`: Logistic Regression baseline, evaluation on imbalanced data, class weights, decision threshold
3. `03_tree_models.ipynb`: Decision Tree (and overfitting), Random Forest, XGBoost, feature importance
4. `04_tuning_shap.ipynb`: 5-fold cross-validation, XGBoost tuning, SHAP explanations, saving the final model
5. `05_neural_network.ipynb`: neural network (MLP) with dropout and early stopping, and the final model choice

## Results
All models are tested on the same held-out test set: 1,409 customers (20%, stratified split).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Lazy (predict all stay) | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.806 | 0.658 | 0.556 | 0.603 | 0.842 |
| Logistic Regression (balanced) | 0.740 | 0.507 | 0.786 | 0.616 | 0.842 |
| Decision Tree | 0.730 | 0.495 | 0.807 | 0.614 | 0.830 |
| Random Forest | 0.762 | 0.536 | 0.759 | 0.628 | 0.845 |
| XGBoost | 0.752 | 0.522 | 0.783 | 0.627 | 0.840 |
| **XGBoost (tuned)** | 0.750 | 0.519 | **0.810** | **0.633** | **0.846** |
| Neural Network (MLP) | 0.745 | 0.513 | **0.810** | 0.628 | 0.842 |

Recall is the share of real churners a model catches. The lazy model shows why accuracy is misleading: 73.5% accuracy, but 0 churners caught.

## Final model: tuned XGBoost
Saved as `models/xgb_tuned.joblib`. It catches 303 of the 374 churners in the test set (recall 0.810).
- The main models are essentially tied: in 5-fold cross-validation, Logistic Regression (balanced), Random Forest and XGBoost scored 0.844 to 0.846 ROC-AUC (std about 0.011).
- Tuned XGBoost has the best test ROC-AUC (0.846, by a small margin), and its cross-validation ROC-AUC was 0.849.
- The neural network was not better: ROC-AUC 0.842, the same recall, and 7 more false alarms. On small tabular data like this, tree models usually match or beat neural networks.
- XGBoost needs no scaling and can explain every prediction with SHAP, which a retention team needs to choose the right offer.

## What drives churn
Found in the EDA and confirmed by the model coefficients, feature importance and SHAP:
- **Contract** is the strongest driver: month-to-month 42.7% churn, one-year 11.3%, two-year 2.8%.
- **Tenure:** new customers churn the most (median tenure 10 months for churners vs. 38 for those who stayed).
- **Fiber optic** internet: 41.9% churn vs. 19.0% for DSL.
- **No tech support or online security:** about 42% churn vs. about 15% with them.
- **Electronic check** payment: 45.3% churn vs. 15 to 19% for the other methods.
- **Gender** has no effect (26.9% vs. 26.2%).

## How to run
```bash
pip install -r requirements.txt
brew install libomp   # macOS only: XGBoost needs the OpenMP runtime
```
Run the notebooks in order (01 to 05), because each one uses files created by the one before it. Open them in VS Code or Jupyter and choose **Run All**, or run them all from the project root:
```bash
for nb in notebooks/0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$nb"; done
```
Notebook 02 rebuilds `data/model_results.csv` from scratch, so after re-running it, re-run notebooks 03 to 05 as well.

Tested with Python 3.12.11, scikit-learn 1.7.2, XGBoost 3.1.2, SHAP 0.54.1, TensorFlow 2.21.0 and Keras 3.15.1 on macOS (Apple Silicon).

## Next steps
- Sequence deep learning (LSTM/GRU) on behavioral data (KKBox dataset)
- Retention platform: prioritize customers by churn risk and customer value, FastAPI and Streamlit dashboard, MLflow
