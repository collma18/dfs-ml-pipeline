"""
Machine Learning Models Module

Contains model definitions, hyperparameters, and training functions.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# MODEL HYPERPARAMETERS
# =========================

RF_PARAMS = {
    'n_estimators': 30,
    'max_depth': 4,
    'min_samples_split': 25,
    'min_samples_leaf': 12,
    'max_features': 0.6,
    'min_impurity_decrease': 0.005,
    'random_state': 42,
    'n_jobs': -1
}

XGB_PARAMS = {
    'n_estimators': 40,
    'max_depth': 4,
    'learning_rate': 0.10,
    'subsample': 0.70,
    'colsample_bytree': 0.65,
    'colsample_bylevel': 0.65,
    'reg_lambda': 5.0,
    'reg_alpha': 1.0,
    'min_child_weight': 6,
    'gamma': 0.3,
    'random_state': 42,
    'n_jobs': -1
}

RIDGE_PARAMS = {
    'alpha': 5.0,
    'random_state': 42
}


# =========================
# DATA PREPARATION
# =========================

def get_valid_data(df, mask, target_col, feature_cols):
    """
    Extract valid data for modeling
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    mask : pd.Series (bool)
        Boolean mask for filtering (e.g., train/test split)
    target_col : str
        Name of target variable column
    feature_cols : list
        List of feature column names
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, pd.Index]
        X (features), y (target), indices
    """
    subset = df[mask].copy()
    subset = subset.dropna(subset=[target_col])
    subset = subset.dropna(subset=feature_cols)
    X = subset[feature_cols].values
    y = subset[target_col].values
    indices = subset.index
    return X, y, indices


# =========================
# MODEL TRAINING
# =========================

def train_and_evaluate(X_train, y_train, X_test, y_test, target_name):
    """
    Train all models and return predictions + metrics
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training targets
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test targets
    target_name : str
        Name of target variable (for printing)
        
    Returns
    -------
    Tuple[dict, dict]
        results (model objects + metrics), predictions (train/test)
    """
    print(f"\n{'=' * 70}")
    print(f"TARGET: {target_name}")
    print(f"{'=' * 70}")
    print(f"Train: {len(X_train)} | Test: {len(X_test)} | Features: {X_train.shape[1]}")

    models = {
        'Random Forest': RandomForestRegressor(**RF_PARAMS),
        'XGBoost': XGBRegressor(**XGB_PARAMS),
        'Ridge': Ridge(**RIDGE_PARAMS),
        'OLS': LinearRegression()
    }

    results = {}
    predictions = {}

    for name, model in models.items():
        print(f"\n⟳ Training {name}...")

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_metrics = {
            'MAE': mean_absolute_error(y_train, train_pred),
            'RMSE': np.sqrt(mean_squared_error(y_train, train_pred)),
            'R2': r2_score(y_train, train_pred),
            'Corr': np.corrcoef(y_train, train_pred)[0, 1]
        }

        test_metrics = {
            'MAE': mean_absolute_error(y_test, test_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, test_pred)),
            'R2': r2_score(y_test, test_pred),
            'Corr': np.corrcoef(y_test, test_pred)[0, 1]
        }

        train_dir_acc = np.mean(np.sign(y_train) == np.sign(train_pred))
        test_dir_acc = np.mean(np.sign(y_test) == np.sign(test_pred))

        r2_gap = train_metrics['R2'] - test_metrics['R2']
        rmse_gap = test_metrics['RMSE'] - train_metrics['RMSE']

        results[name] = {
            'model': model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'train_dir_acc': train_dir_acc,
            'test_dir_acc': test_dir_acc,
            'r2_gap': r2_gap,
            'rmse_gap': rmse_gap
        }

        predictions[name] = {
            'train': train_pred,
            'test': test_pred
        }

        if r2_gap < 0.20:
            gap_status = "✓✓ Excellent"
        elif r2_gap < 0.30:
            gap_status = "✓ Good"
        elif r2_gap < 0.50:
            gap_status = "⚠ Moderate"
        else:
            gap_status = "✗ High"

        print(f"  Train R²: {train_metrics['R2']:.4f} | Test R²: {test_metrics['R2']:.4f}")
        print(f"  Gap: {r2_gap:.4f} {gap_status} | Dir: {test_dir_acc:.2%}")

    # Ensemble
    print(f"\n{'=' * 70}")
    print("ENSEMBLE: Simple Average (Top 3)")
    print(f"{'=' * 70}")

    model_test_r2 = [(name, results[name]['test_metrics']['R2']) for name in models.keys()]
    model_test_r2.sort(key=lambda x: x[1], reverse=True)
    top_3_models = [name for name, _ in model_test_r2[:3]]

    print(f"\nTop 3 by test R²: {', '.join(top_3_models)}")

    train_avg = np.mean([predictions[m]['train'] for m in top_3_models], axis=0)
    test_avg = np.mean([predictions[m]['test'] for m in top_3_models], axis=0)

    avg_train_metrics = {
        'MAE': mean_absolute_error(y_train, train_avg),
        'RMSE': np.sqrt(mean_squared_error(y_train, train_avg)),
        'R2': r2_score(y_train, train_avg),
        'Corr': np.corrcoef(y_train, train_avg)[0, 1]
    }
    avg_test_metrics = {
        'MAE': mean_absolute_error(y_test, test_avg),
        'RMSE': np.sqrt(mean_squared_error(y_test, test_avg)),
        'R2': r2_score(y_test, test_avg),
        'Corr': np.corrcoef(y_test, test_avg)[0, 1]
    }
    avg_train_dir = np.mean(np.sign(y_train) == np.sign(train_avg))
    avg_test_dir = np.mean(np.sign(y_test) == np.sign(test_avg))
    avg_r2_gap = avg_train_metrics['R2'] - avg_test_metrics['R2']

    results['Ensemble_Top3'] = {
        'model': 'average_top3',
        'train_metrics': avg_train_metrics,
        'test_metrics': avg_test_metrics,
        'train_dir_acc': avg_train_dir,
        'test_dir_acc': avg_test_dir,
        'r2_gap': avg_r2_gap,
        'rmse_gap': avg_test_metrics['RMSE'] - avg_train_metrics['RMSE']
    }
    predictions['Ensemble_Top3'] = {'train': train_avg, 'test': test_avg}

    if avg_r2_gap < 0.20:
        gap_status = "✓✓"
    elif avg_r2_gap < 0.30:
        gap_status = "✓"
    else:
        gap_status = "⚠"

    print(f"\nEnsemble Performance:")
    print(f"  Train R²: {avg_train_metrics['R2']:.4f} | Test R²: {avg_test_metrics['R2']:.4f}")
    print(f"  Gap: {avg_r2_gap:.4f} {gap_status} | Dir: {avg_test_dir:.2%}")

    return results, predictions


def create_metrics_df(results, target_name, n_train, n_test):
    """
    Create detailed metrics dataframe
    
    Parameters
    ----------
    results : dict
        Results dictionary from train_and_evaluate
    target_name : str
        Target variable name
    n_train : int
        Number of training samples
    n_test : int
        Number of test samples
        
    Returns
    -------
    pd.DataFrame
        Metrics dataframe with train/test splits
    """
    rows = []
    for model_name, res in results.items():
        rows.append({
            'Target': target_name,
            'Model': model_name,
            'Split': 'Train',
            'n': n_train,
            'R2': res['train_metrics']['R2'],
            'RMSE': res['train_metrics']['RMSE'],
            'Dir_Acc': res['train_dir_acc'],
            'R2_Gap': res['r2_gap']
        })
        rows.append({
            'Target': target_name,
            'Model': model_name,
            'Split': 'Test',
            'n': n_test,
            'R2': res['test_metrics']['R2'],
            'RMSE': res['test_metrics']['RMSE'],
            'Dir_Acc': res['test_dir_acc'],
            'R2_Gap': res['r2_gap']
        })
    return pd.DataFrame(rows)