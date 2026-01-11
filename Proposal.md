# Project Proposal: Forecasting Post-DFS Price Targets for ASX Gold Developers

**Category:** Machine Learning and Finance Tools

## Problem Statement / Motivation
Definitive Feasibility Studies (DFS) are the key document published by resource companies before a decision to invest is decided. This announcements can hint at a material change in the company but determining the stock price reaction is not as simple as it seems. Some projects surge on strong economics and sentiment, while others drift lower due to funding risk or poor market conditions.  

This project aims to build a Python-based forecasting tool that predicts a stock’s price level (and return) at 30 trading days after a DFS announcement, by analysing key macroeconomic data as well as they key features of the feasibility study to see if there is any predictable returns following these announcements. 

## Planned Approach and Technologies
1. **Data collection:** Historical daily stock data for ASX gold developers, gold spot price (USD/AUD), and the ASX All Ordinaries Index. DFS publications will be extracted from ASX announcements.  
2. **Feature engineering:** Gold price changes, market indices, DFS fundamentals (NPV, IRR, CAPEX, grade, AISC, production), and context features such as market cap and volatility.  
3. **Modeling:** Start with linear models, then methods such as Gradient Boosted Trees for non-linear effects. Baseline comparisons will include average historical returns and gold-beta regressions. The model will be trained from 2000-2018, and tested on 2019-2024 (exact split determined after DFS have been collected to ensure around a 75/25% split). The models that will be used will be Linear Regression, Ridge, Random Forest and XGBoost.
4. **Evaluation:** Use time-series blocked cross-validation and metrics such as MAE, RMSE, and calibration accuracy.  


**Technologies:** Python 3.10+, pandas, scikit-learn, XGBoost/lightGBM, matplotlib/Plotly, yahoo finance API.

## Expected Challenges and Mitigation
- **Data quality, quantity and extraction:** Finding the best way to collect a large enough dataset of DFS studies and accurately pull out the correct data. Use AI API to send PDF and extract the important information. If there is insuffient data can extend the scope of the project to include other minerals, including copper, nickle, lithium and iron. Manual validation of some DFS will be done periodically to ensure the correct data points are being collected.
- **Market noise and external events:** Include control features (gold price movement, index return) to isolate event effects.  
- **Company Size and Other Project:** A small study from a large producer could skew results. Try to use market cap or possibly % of reserves to identify outliers.

## Success Criteria
- Outperforms naive baselines (e.g., mean post-DFS return) on MAE/MAPE.    
- The results should be easy to interpret, highlighting which project or market variables most influence the predicted stock movements. 
- Generates reproducible, well-documented code.

## Stretch Goals
- Create a model for different time frames (ie 1, 90, 180 days)
- Test results using walk forward validation
- Integrate automatic data collection for DFS.
- Extend the tool to other commodities.
- Create a visualisation and interface dashboard for users to select and view different criteria.