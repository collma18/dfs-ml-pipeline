# Mining Stock Returns Prediction Pipeline

A comprehensive machine learning pipeline for predicting stock returns following mining feasibility study (DFS) announcements on the Australian Stock Exchange (ASX).

## Overview

This pipeline analyzes the impact of Definitive Feasibility Study (DFS) announcements on mining company stock prices. It builds machine learning models to predict both **raw returns** and **market-adjusted (abnormal) returns** over a 30-day post-announcement window.

### Key Objectives

- **Benchmark**: Outperform naive baselines (constant mean, past mean) on predicting 30-day stock returns

## What It Does

1. **Data Engineering**: Processes DFS announcement data and normalizes various financial metrics
2. **Market Data**: Downloads historical price data from Yahoo Finance and delisted stocks database
3. **Event Study**: Constructs 30-day pre/post announcement windows for each DFS event
4. **Feature Engineering**: Creates ML-ready features including ratios, momentum indicators, and categorical encodings
5. **Model Training**: Trains multiple regression models (Random Forest, XGBoost, Ridge, OLS, Ensemble)
6. **Evaluation**: Comprehensive performance analysis with visualizations

## Pipeline Architecture

```
INPUT DATA
├── DFS Data Sheet (project fundamentals)
└── Delisted Share Prices (historical backup)
        ↓
FEATURE ENGINEERING
├── Unit normalization (AUD/ton, t/y, etc.)
├── NPV/IRR selection (post-tax → pre-tax fallback)
├── Commodity price imputation
├── Metal categorization (bucket/major)
└── Financial ratios (NPV/CapEx, NPV/MktCap, etc.)
        ↓
MARKET DATA COLLECTION
├── Download index (^AXMM) prices
├── Download individual stock prices (parallel)
├── Build event windows (t-30 to t+30)
└── Calculate returns & momentum
        ↓
PREPROCESSING
├── Train/test split (80/20 by time)
├── Feature scaling (log, winsorize, robust/standard)
├── One-hot encoding (categorical features)
└── Missing value handling
        ↓
MODEL TRAINING
├── Random Forest
├── XGBoost
├── Ridge Regression
├── OLS Regression
└── Ensemble (Top-3 average)
        ↓
OUTPUTS
├── Excel workbooks (results, metrics)
├── Feature distribution plots
└── Performance comparison charts
```

## Project Structure

```
.
├── dfs_features_safe.xlsx          # INPUT: Source data
├── ml_output.xlsx                  # OUTPUT: Preprocessing results
├── model_results_FINAL_v2.xlsx     # OUTPUT: Model performance
├── _cache/                         # Cached data (auto-generated)
│   ├── inputs_cache.pkl
│   └── yf/                         # Yahoo Finance cache
├── features_original.png           # Feature distributions (raw)
├── features_scaled.png             # Feature distributions (scaled)
├── 1_raw_returns_r2.png           # Model comparison: Raw R²
├── 2_abnormal_returns_r2.png      # Model comparison: Abnormal R²
├── 3_overfitting_gap.png          # Model comparison: R² Gap
├── 4_raw_direction_accuracy.png   # Model comparison: Raw Direction
├── 5_abnormal_direction_accuracy.png
├── 7_raw_mae.png                  # Model comparison: MAE
├── 8_abnormal_mae.png
├── 9_raw_rmse.png                 # Model comparison: RMSE
├── 10_abnormal_rmse.png
└── 11_model_rankings.png          # ⭐ Final rankings table
```

## Installation

### Requirements

```bash
pip install pandas numpy yfinance scikit-learn xgboost scipy matplotlib seaborn openpyxl --break-system-packages
```

### Key Dependencies

- **pandas**: Data manipulation
- **yfinance**: Market data download
- **scikit-learn**: ML models (Random Forest, Ridge, OLS)
- **xgboost**: Gradient boosting
- **scipy**: Statistical functions (winsorization)
- **matplotlib/seaborn**: Visualizations

## Configuration

Edit the `SETTINGS` section at the top of the script:

```python
# File paths
IN_PATH = Path(r"C:\Users\User\Desktop\Test1\dfs_features_safe.xlsx")
OUT_PATH = Path(r"C:\Users\User\Desktop\Test1\ml_output.xlsx")
RESULTS_PATH = Path(r"C:\Users\User\Desktop\Test1\model_results_FINAL_v2.xlsx")

# Event window settings
N_TRADING_DAYS = 30           # Days before/after announcement
CALENDAR_BUFFER_DAYS = 260    # Historical data buffer
INDEX_TICKER = "^AXMM"        # ASX All Mining Index

# Train/test split
TEST_FRAC = 0.20              # 20% test set (time-based)

# Momentum indicator
MOMENTUM_WINDOW = 30          # Index momentum lookback period

# Caching
USE_CACHE = True              # Enable/disable caching
```

### Model Hyperparameters

The script includes optimized hyperparameters for each model:

```python
RF_PARAMS = {
    'n_estimators': 30,
    'max_depth': 4,
    'min_samples_split': 25,
    # ... (see script for full config)
}

XGB_PARAMS = {
    'n_estimators': 40,
    'max_depth': 4,
    'learning_rate': 0.10,
    # ... (see script for full config)
}
```

## Usage

### Basic Execution

```bash
python unified_ml_pipeline.py
```

### What Happens

1. **Data Loading** (5-10 seconds)
   - Reads Excel input
   - Checks cache for previous runs
   
2. **Feature Engineering** (10-20 seconds)
   - Normalizes units
   - Computes ratios
   - Categorizes metals
   
3. **Market Data Download** (2-5 minutes)
   - Downloads ~100-200 stocks in parallel
   - Uses cache for previously downloaded data
   
4. **Preprocessing** (30-60 seconds)
   - Scales features
   - Creates visualizations
   
5. **Model Training** (1-2 minutes)
   - Trains 5 models × 2 targets = 10 models total
   - Generates predictions
   
6. **Results Export** (10 seconds)
   - Saves Excel files
   - Creates 14 PNG charts

**Total Runtime**: ~5-10 minutes (first run), ~2-3 minutes (cached)

## Input Data Format

### Required Sheets

#### 1. `DFS Data`
Expected columns (position-based):
- Column 2: `Ticker` (ASX code)
- Column 3: `announcement_date` (date)
- `npv_post_tax_aud`, `npv_pre_tax_aud`
- `irr_post_tax_pct`, `irr_pre_tax_pct`
- `payback_years`
- `aisc_aud_per_unit`, `aisc_unit`
- `annual_production`, `annual_production_unit`
- `base_case_commodity_price_aud_per_unit`, `commodity_price_unit`
- `mkt_cap_aud`
- `mine_life_years`
- `initial_capex_aud`
- `permitting_status`
- `primary_metal`

#### 2. `Delisted Share Prices`
- Column 0: `Ticker`
- Column 1: `Date`
- Column 5: `Close` price

### Unit Conversions

The pipeline automatically normalizes to standard units:

| Metric | Standard Unit |
|--------|---------------|
| Price | AUD/ton |
| Production | ton/year |
| AISC | AUD/ton |

Supported input units:
- **Price/AISC**: `aud/t`, `aud/kg`, `aud/lb`, `aud/oz`
- **Production**: `t/y`, `kg/y`, `lb/y`, `oz/y`

## Features

### Core Features (Scaled)

1. **NPV/Market Cap Ratio** (log → robust scale)
2. **NPV/CapEx Ratio** (winsorize → log → robust scale)
3. **IRR %** (winsorize → robust scale)
4. **Mine Life** (standard scale)
5. **Index Momentum** (30-day, standard scale)

### Categorical Features

1. **Metal Bucket**: precious, battery, base, bulk, critical, specialty, other
2. **NPV Flag**: post-tax (1) vs pre-tax (0)
3. **Permitting Status**: 0, 1, 2, etc.

### Derived Features

- Operating Margin: `(Price - AISC) / Price`
- Annual Revenue: `Production × Price`
- Primary Metal Major: gold, lithium, copper, graphite, iron ore, nickel, other

## Models

### 1. Random Forest
- **Purpose**: Captures non-linear relationships
- **Strengths**: Handles interactions, robust to outliers
- **Tuning**: Restricted depth/samples to prevent overfitting

### 2. XGBoost
- **Purpose**: Gradient boosting with regularization
- **Strengths**: Feature interactions, L1/L2 regularization
- **Tuning**: Low learning rate, high lambda/alpha

### 3. Ridge Regression
- **Purpose**: Linear baseline with L2 penalty
- **Strengths**: Interpretable, fast, handles multicollinearity

### 4. OLS Regression
- **Purpose**: Pure linear baseline
- **Strengths**: Simplest model, interpretable coefficients

### 5. Ensemble (Top-3)
- **Purpose**: Combine best models
- **Method**: Simple average of top 3 models by test R²
- **Strengths**: Reduces variance, often best performer

## Output Files

### 1. `ml_output.xlsx`

| Sheet | Description |
|-------|-------------|
| `model_features` | Engineered features before event merge |
| `events` | Event study data (t-30 to t+30) |
| `ml_training_table` | Full merged dataset |
| `preprocessing_summary` | Feature scaling statistics |
| `overall_metrics` | Baseline model metrics |
| `split_metrics` | Train/test baseline metrics |
| `skipped_events` | Events excluded (missing data) |
| `ml_ready` | Final modeling dataset |

### 2. `model_results_FINAL_v2.xlsx`

| Sheet | Description |
|-------|-------------|
| `model_metrics` | Train/test metrics for all models |
| `summary` | Key performance indicators |
| `xgb_params` | XGBoost hyperparameters used |

### 3. Visualizations

**Feature Distributions**:
- `features_original.png`: Raw feature histograms
- `features_scaled.png`: Scaled feature histograms

**Model Performance** (11 charts):
- R² scores (raw, abnormal)
- Overfitting gap (ML models only)
- Directional accuracy (raw, abnormal)
- MAE (raw, abnormal)
- RMSE (raw, abnormal)
- **Model rankings table** ⭐

## 📐 Evaluation Metrics

### Primary Metrics

1. **Directional Accuracy**: % of predictions with correct sign
   - Target: 75-77%
   - Random baseline: 50%

2. **R² Gap**: `Train R² - Test R²`
   - Target: < 0.25
   - Indicates overfitting if too high

### Secondary Metrics

3. **R² Score**: Variance explained
4. **MAE**: Mean Absolute Error
5. **RMSE**: Root Mean Squared Error
6. **Correlation**: Pearson correlation between predictions and actuals

### Ranking Methodology

Models are ranked using a composite score:

```
Score = 0.40 × Direction + 0.20 × R² + 0.20 × (1 - MAE_norm) + 0.20 × (1 - RMSE_norm)
```

**Rationale**:
- Direction (40%): Most important for trading decisions
- R² (20%): Context on variance explained
- MAE (20%): Average prediction error
- RMSE (20%): Penalizes large errors

## Baselines

### Naive Past Mean
Predicts next 30 days = average of past 30 days
- Raw returns: historical stock returns
- Abnormal returns: historical (stock - index) returns

### Constant Mean
Predicts constant value = training set mean
- Simple but often competitive
- Useful sanity check

## 🔍 Key Design Decisions

### 1. Time-Based Train/Test Split
- **Why**: Prevents look-ahead bias
- **How**: Last 20% of events chronologically → test set
- **Alternative rejected**: Random split (unrealistic)

### 2. Imputation Strategy
- **Commodity prices**: Nearest same-metal observation within 365 days
- **NPV/IRR**: Post-tax preferred, fallback to pre-tax
- **Rationale**: Preserve information, avoid dropping rows

### 3. Feature Scaling
- **Robust Scaler**: For skewed distributions (NPV, CapEx, prices)
- **Standard Scaler**: For symmetric distributions (IRR, momentum)
- **Log Transform**: For highly skewed positive values
- **Winsorization**: Caps outliers at 5th/95th percentile

### 4. Ensemble Method
- **Top-3 average** vs weighted average
- **Rationale**: Simplicity, robustness, avoids overfitting to validation set

### 5. Caching System
- **Yahoo Finance data**: Cached by ticker + date range
- **Excel inputs**: Cached by file fingerprint (mtime + size)
- **Benefit**: 3-5x speedup on repeated runs

##  Common Issues

### 1. Missing Market Data
**Problem**: Stock not found on Yahoo Finance

**Solutions**:
- Check ticker format (should be `XXX.AX`)
- Verify announcement date (must be within trading history)
- Add to delisted prices sheet if delisted

### 2. Unit Conversion Errors
**Problem**: Unknown unit in data

**Solutions**:
- Check `UNIT_CONVERSION` mapping
- Add new unit to mapping dictionaries
- Verify spelling/capitalization

### 3. Low Directional Accuracy
**Problem**: Models < 60% directional accuracy

**Diagnostics**:
- Check feature distributions (look for outliers)
- Review train/test split (ensure chronological)
- Examine skipped events (data quality issues)

**Solutions**:
- Increase `MOMENTUM_WINDOW`
- Add more features (e.g., sector indicators)
- Tune hyperparameters

### 4. High R² Gap (Overfitting)
**Problem**: Train R² much higher than test R²

**Solutions**:
- Reduce model complexity:
  - Random Forest: decrease `max_depth`, increase `min_samples_leaf`
  - XGBoost: increase `reg_lambda`, `reg_alpha`
- Add more training data
- Check for data leakage

## Advanced Usage

### Custom Feature Engineering

Add new features to the `engineer_features()` function:

```python
def engineer_features(dfs: pd.DataFrame) -> pd.DataFrame:
    # ... existing code ...
    
    # Add your custom feature
    features_df["my_custom_ratio"] = (
        features_df["numerator"] / features_df["denominator"]
    )
    
    return features_df
```

Then add to `preprocessing_config` in `preprocess_for_ml()`:

```python
preprocessing_config = {
    # ... existing features ...
    'my_custom_ratio': ['log', 'robust'],  # transformation steps
}
```

### Hyperparameter Tuning

For XGBoost:

```python
XGB_PARAMS = {
    'n_estimators': 40,        # More trees → better fit (but slower)
    'max_depth': 4,            # Deeper → more complex (overfitting risk)
    'learning_rate': 0.10,     # Lower → more conservative (slower learning)
    'reg_lambda': 5.0,         # Higher → more L2 regularization
    'reg_alpha': 1.0,          # Higher → more L1 regularization
    'min_child_weight': 6,     # Higher → more conservative splits
    # ...
}
```

### Adding New Models

```python
from sklearn.ensemble import GradientBoostingRegressor

# Add to models dictionary in train_and_evaluate()
models = {
    'Random Forest': RandomForestRegressor(**RF_PARAMS),
    'XGBoost': XGBRegressor(**XGB_PARAMS),
    'Ridge': Ridge(**RIDGE_PARAMS),
    'OLS': LinearRegression(),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1
    )
}
```

## References

### Metal Classifications

**Buckets**:
- **Precious**: Gold
- **Battery**: Lithium, Nickel, Graphite, Vanadium, Cobalt
- **Base**: Copper, Zinc, Lead, Tin, Molybdenum
- **Bulk**: Iron Ore, Coal, Bauxite, Alumina, Potash, Phosphate, Kaolin
- **Critical**: NDPR, Dysprosium, Scandium, Rare Earths, Niobium, Zirconium, Uranium
- **Specialty**: Tungsten, Zircon

### Data Sources

- **Stock Prices**: Yahoo Finance API (via yfinance)
- **Index**: ASX All Mining Index (^AXMM)
- **Delisted Stocks**: Manual database (from Excel input)

## Debugging

Enable verbose output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check intermediate outputs:

```python
# After feature engineering
print(features_df[['Ticker', 'best_npv_aud', 'npv_to_mktcap']].head())

# After scaling
print(ml_df_processed[['npv_to_mktcap_scaled']].describe())

# After model training
print(results_raw['XGBoost']['train_metrics'])
```





