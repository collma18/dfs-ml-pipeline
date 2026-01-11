# Mining Stock Return Prediction

This project predicts **30-day stock returns** following mining feasibility (DFS) announcements on the ASX using machine learning.

It combines:
- Project fundamentals (NPV, IRR, costs, production metrics)
- Historical stock price data  

to model how the market reacts after DFS announcements.

---

## Project Overview

The pipeline:
- Cleans and standardises DFS project data  
- Downloads historical stock prices from Yahoo Finance  
- Constructs pre/post announcement event windows  
- Engineers financial and market features  
- Trains multiple machine learning models  
- Evaluates predictive performance  

---

## Installation

pip install -r requirements.txt

Example requirements.txt:

pandas  
numpy  
yfinance  
scikit-learn  
xgboost  
matplotlib  
openpyxl  

---

## How to Run

python unified_ml_pipeline.py

---

## Input Files

dfs_features_safe.xlsx  
- DFS project fundamentals  
- Announcement dates  
- Delisted stock prices (backup source)  

---

## Output Files

ml_output.xlsx  
- Engineered features  
- Event study results  
- Final modelling dataset  

model_results_FINAL_v2.xlsx  
- Train/test metrics  
- Model rankings  

PNG charts  
- Feature distributions  
- Model performance comparisons  

---

## Models

- Random Forest  
- XGBoost  
- Ridge Regression  
- Linear Regression  
- Simple ensemble model  

---

## Evaluation Metrics

- Directional accuracy  
- R²  
- MAE  
- RMSE  

---

## Notes

- Uses time-based train/test split  
- Includes market-adjusted (abnormal) returns  
- Caching enabled for faster repeated runs  
- Yahoo Finance used for price data  
