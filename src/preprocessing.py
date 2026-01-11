"""
Feature Preprocessing Module

Handles feature scaling, transformations, and preparation for ML models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, StandardScaler
from scipy.stats import mstats


# =========================
# FEATURE SELECTION
# =========================

CORE_FEATURES_SCALED = [
    'npv_to_mktcap_scaled',
    'npv_to_capex_scaled',
    'best_irr_pct_scaled',
    'mine_life_years_scaled',
    'index_momentum_30d_scaled'
]

CORE_FEATURES_CATEGORICAL = [
    'metal_bucket',
    'npv_flag',
    'permitting_status'
]


# =========================
# PREPROCESSING FUNCTIONS
# =========================

def compute_index_momentum(row, n=30):
    """
    Compute index momentum for a single row
    
    Parameters
    ----------
    row : pd.Series
        Row with index close prices
    n : int
        Lookback window
        
    Returns
    -------
    float
        Momentum value
    """
    try:
        p0 = row.get(f"xmm_close_t0", np.nan)
        p_n = row.get(f"xmm_close_t{-n}", np.nan)
        if pd.notna(p0) and pd.notna(p_n) and p_n != 0:
            return (p0 / p_n) - 1.0
        return np.nan
    except:
        return np.nan


def winsorize_column(series, limits=(0.05, 0.05)):
    """
    Winsorize column to handle outliers
    
    Parameters
    ----------
    series : pd.Series
        Column to winsorize
    limits : tuple
        (lower, upper) percentile limits
        
    Returns
    -------
    pd.Series
        Winsorized series
    """
    return pd.Series(
        mstats.winsorize(series.dropna(), limits=limits),
        index=series.dropna().index
    ).reindex(series.index)


def log_transform(series, offset=1e-6):
    """
    Apply log transformation with offset for negative values
    
    Parameters
    ----------
    series : pd.Series
        Series to transform
    offset : float
        Offset for negative values
        
    Returns
    -------
    pd.Series
        Log-transformed series
    """
    min_val = series.min()
    if min_val <= 0:
        shift = abs(min_val) + offset
        return np.log1p(series + shift)
    return np.log1p(series)


def fit_scaler_train_only(train_df, test_df, column, scaler_type='robust'):
    """
    Fit scaler on training data only and transform both train and test
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe
    test_df : pd.DataFrame
        Test dataframe
    column : str
        Column name to scale
    scaler_type : str
        'robust' or 'standard'
        
    Returns
    -------
    Tuple[pd.Series, pd.Series, Scaler]
        Transformed train series, test series, fitted scaler
    """
    if scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    train_vals = train_df[column].dropna().values.reshape(-1, 1)
    scaler.fit(train_vals)

    train_transformed = pd.Series(
        scaler.transform(train_df[column].values.reshape(-1, 1)).flatten(),
        index=train_df.index
    )
    test_transformed = pd.Series(
        scaler.transform(test_df[column].values.reshape(-1, 1)).flatten(),
        index=test_df.index
    )

    return train_transformed, test_transformed, scaler


def preprocess_for_ml(ml_df, momentum_window=30):
    """
    Preprocess features for ML modeling
    
    Parameters
    ----------
    ml_df : pd.DataFrame
        Merged ML dataframe
    momentum_window : int
        Window for momentum calculation
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, list, list]
        Processed dataframe, summary stats, scaled feature names, categorical feature names
    """
    print("\n" + "=" * 60)
    print("ML FEATURE PREPROCESSING")
    print("=" * 60)

    # Momentum feature
    print(f"\n⟳ Computing index momentum (t-{momentum_window} to t0)...")
    momentum_values = [compute_index_momentum(row, n=momentum_window)
                       for _, row in ml_df.iterrows()]
    ml_df = ml_df.assign(index_momentum_30d=momentum_values)

    momentum_valid = ml_df["index_momentum_30d"].notna().sum()
    print(f"✓ Index momentum computed for {momentum_valid}/{len(ml_df)} rows")

    # Preprocessing pipeline
    print("\n⟳ Preprocessing features...")

    train_df = ml_df[ml_df["split"] == "train"].copy()
    test_df = ml_df[ml_df["split"] == "test"].copy()
    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

    original_features = {}
    scaled_features = {}

    preprocessing_config = {
        'best_npv_aud': ['log', 'robust'],
        'initial_capex_aud': ['log', 'robust'],
        'npv_to_mktcap': ['log', 'robust'],
        'annual_revenue_aud': ['log', 'robust'],
        'npv_to_capex': ['winsorize', 'log', 'robust'],
        'best_irr_pct': ['winsorize', 'robust'],
        'payback_years': ['standard'],
        'mine_life_years': ['standard'],
        'operating_margin': ['standard'],
        'index_momentum_30d': ['standard'],
    }

    # Collect all new columns
    new_columns = {}

    for feature, steps in preprocessing_config.items():
        if feature not in ml_df.columns:
            print(f"  ⚠ Skipping {feature} (not found)")
            continue

        original_features[feature] = ml_df[feature].copy()

        train_col = train_df[feature].copy()
        test_col = test_df[feature].copy()

        for step in steps:
            if step == 'winsorize':
                train_col = winsorize_column(train_col)
                test_col = winsorize_column(test_col)
            elif step == 'log':
                train_col = log_transform(train_col)
                test_col = log_transform(test_col)
            elif step in ['robust', 'standard']:
                train_scaled, test_scaled, scaler = fit_scaler_train_only(
                    pd.DataFrame({feature: train_col}),
                    pd.DataFrame({feature: test_col}),
                    feature,
                    scaler_type=step
                )
                train_col = train_scaled
                test_col = test_scaled

        # Store in dictionary
        new_col = f"{feature}_scaled"
        new_columns[new_col] = pd.Series(index=ml_df.index, dtype=float)
        new_columns[new_col].loc[train_df.index] = train_col
        new_columns[new_col].loc[test_df.index] = test_col

        scaled_features[feature] = new_columns[new_col].copy()
        print(f"  ✓ {feature}: {' → '.join(steps)}")

    # Add all new columns at once
    ml_df = pd.concat([ml_df, pd.DataFrame(new_columns)], axis=1)

    # Create visualizations
    print("\n⟳ Creating histograms...")
    create_feature_histograms(original_features, scaled_features, preprocessing_config)

    # Summary
    summary_df = create_preprocessing_summary(original_features, scaled_features, preprocessing_config)
    
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print("\n" + summary_df.to_string(index=False))

    scaled_feature_cols = [f"{f}_scaled" for f in preprocessing_config.keys() 
                          if f"{f}_scaled" in ml_df.columns]
    categorical_features = ['primary_metal_major', 'metal_bucket', 'npv_flag', 'irr_flag',
                            'commodity_price_imputed_flag', 'permitting_status']

    print(f"\n✓ Scaled features: {len(scaled_feature_cols)}")
    print(f"✓ Categorical features: {len(categorical_features)}")

    return ml_df, summary_df, scaled_feature_cols, categorical_features


def create_feature_histograms(original_features, scaled_features, preprocessing_config):
    """
    Create histogram visualizations for features
    
    Parameters
    ----------
    original_features : dict
        Dictionary of original feature series
    scaled_features : dict
        Dictionary of scaled feature series
    preprocessing_config : dict
        Preprocessing configuration
        
    Returns
    -------
    None
        Saves PNG files
    """
    from pathlib import Path
    
    n_features = len(preprocessing_config)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))

    # Original features
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes1 = axes1.flatten() if n_features > 1 else [axes1]

    for idx, (feature, _) in enumerate(preprocessing_config.items()):
        if feature in original_features:
            ax = axes1[idx]
            data = original_features[feature].dropna()

            ax.hist(data, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
            ax.set_title(f'{feature}\n(Original)', fontsize=10, fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(axis='y', alpha=0.3)

            ax.text(0.02, 0.98,
                    f'n={len(data)}\nμ={data.mean():.2e}\nσ={data.std():.2e}',
                    transform=ax.transAxes, fontsize=8,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    for idx in range(n_features, len(axes1)):
        axes1[idx].axis('off')

    plt.tight_layout()
    save_path = Path.cwd() / "features_original.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print("✓ Saved: features_original.png")
    plt.close()

    # Scaled features
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes2 = axes2.flatten() if n_features > 1 else [axes2]

    for idx, (feature, steps) in enumerate(preprocessing_config.items()):
        if feature in scaled_features:
            ax = axes2[idx]
            data = scaled_features[feature].dropna()

            ax.hist(data, bins=50, edgecolor='black', alpha=0.7, color='coral')
            ax.set_title(f'{feature}_scaled\n({" → ".join(steps)})', fontsize=10, fontweight='bold')
            ax.set_xlabel('Scaled Value')
            ax.set_ylabel('Frequency')
            ax.grid(axis='y', alpha=0.3)

            ax.text(0.02, 0.98,
                    f'n={len(data)}\nμ={data.mean():.2f}\nσ={data.std():.2f}',
                    transform=ax.transAxes, fontsize=8,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    for idx in range(n_features, len(axes2)):
        axes2[idx].axis('off')

    plt.tight_layout()
    save_path = Path.cwd() / "features_scaled.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print("✓ Saved: features_scaled.png")
    plt.close()


def create_preprocessing_summary(original_features, scaled_features, preprocessing_config):
    """
    Create summary dataframe of preprocessing statistics
    
    Parameters
    ----------
    original_features : dict
        Dictionary of original feature series
    scaled_features : dict
        Dictionary of scaled feature series
    preprocessing_config : dict
        Preprocessing configuration
        
    Returns
    -------
    pd.DataFrame
        Summary statistics
    """
    summary_rows = []
    for feature in preprocessing_config.keys():
        if feature in original_features and feature in scaled_features:
            orig = original_features[feature].dropna()
            scaled = scaled_features[feature].dropna()

            summary_rows.append({
                'Feature': feature,
                'n_valid': len(orig),
                'orig_mean': orig.mean(),
                'orig_std': orig.std(),
                'orig_min': orig.min(),
                'orig_max': orig.max(),
                'scaled_mean': scaled.mean(),
                'scaled_std': scaled.std(),
                'scaled_min': scaled.min(),
                'scaled_max': scaled.max(),
            })

    return pd.DataFrame(summary_rows)