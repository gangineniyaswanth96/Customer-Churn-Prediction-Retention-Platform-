# Customer Churn Prediction & Retention Platform

A telecom company is losing customers (**churn**). This project predicts which customers are likely to leave, so the company can act early with retention offers (discounts, tech support, contract upgrades).

- **Problem type:** binary classification (`Churn` = 1 left / 0 stayed)
- **Dataset:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): 7,043 customers, 21 columns

## Project structure
```
data/
  telco_dataset.csv   original data (never modified)
  telco_clean.csv     encoded data, created by notebook 01
notebooks/
  01_eda.ipynb        Lessons 1-4: load, clean, EDA, data preparation
src/                  reusable code (later lessons)
requirements.txt
```

## How to run
```bash
pip install -r requirements.txt
```
Open `notebooks/01_eda.ipynb` in VS Code (or `jupyter notebook`) and choose **Run All**.

## Progress
- [x] Lesson 1: Load & first look
- [x] Lesson 2: Data cleaning
- [x] Lesson 3: Exploratory data analysis
- [x] Lesson 4: Encoding, train/test split, scaling
- [ ] Lesson 5+: Modelling
