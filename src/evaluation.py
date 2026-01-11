"""
Model Evaluation and Visualization Module

Contains functions for computing metrics, creating visualizations,
and evaluating model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# BENCHMARK COMPUTATION
# =========================

def _series_from_row(row: pd.Series, prefix: str, t_min: int, t_max: int) -> pd.Series:
    """Extract time series from row with given prefix"""
    vals = {t: row.get(f"{prefix}_t{t}", np.nan) for t in range(t_min, t_max + 1)}
    s = pd.Series(vals, dtype="float64")
    s.index = s.index.astype(int)
    return s


def compute_naive_benchmarks(row: pd.Series, n: int = 30) -> pd.Series:
    """
    Compute naive baseline predictions
    
    Parameters
    ----------
    row : pd.Series
        Event row with close prices
    n : int
        Number of trading days
        
    Returns
    -------
    pd.Series
        Benchmark predictions and realized returns
    """
    stock_close = _series_from_row(row, "close", -n, n)
    idx_close = _series_from_row(row, "xmm_close", -n, n)

    c0, c30 = stock_close.get(0, np.nan), stock_close.get(n, np.nan)
    i0, i30 = idx_close.get(0, np.nan), idx_close.get(n, np.nan)

    realized_raw_30d = (c30 / c0 - 1.0) if np.isfinite(c0) and np.isfinite(c30) and c0 != 0 else np.nan
    realized_idx_30d = (i30 / i0 - 1.0) if np.isfinite(i0) and np.isfinite(i30) and i0 != 0 else np.nan

    stock_daily = stock_close.pct_change()
    idx_daily = idx_close.pct_change()
    pre_slice = list(range(-(n - 1), 1))

    mean_stock_pre = stock_daily.loc[pre_slice].replace([np.inf, -np.inf], np.nan).mean()
    pred_raw_30d = (1.0 + mean_stock_pre) ** n - 1.0 if np.isfinite(mean_stock_pre) else np.nan

    abn_daily = (stock_daily - idx_daily).replace([np.inf, -np.inf], np.nan)
    mean_abn_pre = abn_daily.loc[pre_slice].mean()
    pred_abn_30d = (1.0 + mean_abn_pre) ** n - 1.0 if np.isfinite(mean_abn_pre) else np.nan

    realized_abn_30d = (realized_raw_30d - realized_idx_30d
                        if np.isfinite(realized_raw_30d) and np.isfinite(realized_idx_30d)
                        else np.nan)

    return pd.Series({
        "realized_raw_ret_30d": realized_raw_30d,
        "realized_index_ret_30d": realized_idx_30d,
        "realized_abn_ret_30d": realized_abn_30d,
        "naive_pred_raw_ret_30d": pred_raw_30d,
        "naive_pred_abn_ret_30d": pred_abn_30d,
        "naive_mean_daily_raw_pre30": mean_stock_pre,
        "naive_mean_daily_abn_pre30": mean_abn_pre,
    })


# =========================
# METRICS COMPUTATION
# =========================

def _safe_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute Mean Absolute Percentage Error safely"""
    mask = y_true.notna() & y_pred.notna() & (y_true != 0)
    return float((np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])).mean()) if mask.sum() else np.nan


def _safe_smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute Symmetric Mean Absolute Percentage Error safely"""
    mask = y_true.notna() & y_pred.notna()
    if mask.sum() == 0:
        return np.nan
    denom = (np.abs(y_true[mask]) + np.abs(y_pred[mask])).replace(0, np.nan)
    return float((2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom).mean())


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Calculate comprehensive regression metrics
    
    Parameters
    ----------
    y_true : pd.Series
        True values
    y_pred : pd.Series
        Predicted values
        
    Returns
    -------
    dict
        Dictionary of metrics
    """
    mask = y_true.notna() & y_pred.notna() & np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask].astype(float), y_pred[mask].astype(float)
    n = int(mask.sum())

    if n == 0:
        return {"n": 0, "MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "R2": np.nan,
                "MAPE": np.nan, "sMAPE": np.nan, "Corr": np.nan, "Directional_Acc": np.nan,
                "Mean_True": np.nan, "Mean_Pred": np.nan, "Median_True": np.nan, "Median_Pred": np.nan}

    err = yp - yt
    mae, mse, rmse = float(np.abs(err).mean()), float((err ** 2).mean()), float(np.sqrt((err ** 2).mean()))
    ss_res, ss_tot = float(((yt - yp) ** 2).sum()), float(((yt - yt.mean()) ** 2).sum())
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    corr = float(yt.corr(yp)) if n >= 2 else np.nan
    dir_mask = (yt != 0) & (yp != 0)
    directional_acc = float((np.sign(yt[dir_mask]) == np.sign(yp[dir_mask])).mean()) if dir_mask.sum() else np.nan

    return {
        "n": n, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2,
        "MAPE": _safe_mape(yt, yp), "sMAPE": _safe_smape(yt, yp),
        "Corr": corr, "Directional_Acc": directional_acc,
        "Mean_True": float(yt.mean()), "Mean_Pred": float(yp.mean()),
        "Median_True": float(yt.median()), "Median_Pred": float(yp.median())
    }


def make_metrics_table(df: pd.DataFrame, tasks) -> pd.DataFrame:
    """
    Generate metrics table for multiple tasks
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with predictions and actuals
    tasks : list
        List of (task_name, y_col, pred_col, model_name) tuples
        
    Returns
    -------
    pd.DataFrame
        Metrics table
    """
    rows = []
    for task_name, y_col, p_col, model_name in tasks:
        if y_col not in df.columns or p_col not in df.columns:
            rows.append({"Task": task_name, "Model": model_name, "Error": f"Missing {y_col} or {p_col}"})
            continue
        m = regression_metrics(df[y_col], df[p_col])
        m.update({"Task": task_name, "Model": model_name, "y_col": y_col, "pred_col": p_col})
        rows.append(m)

    out = pd.DataFrame(rows)
    preferred = ["Task", "Model", "n", "MAE", "RMSE", "MSE", "R2", "Corr",
                 "Directional_Acc", "MAPE", "sMAPE", "Mean_True", "Mean_Pred",
                 "Median_True", "Median_Pred", "y_col", "pred_col", "Error"]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[cols]


# =========================
# VISUALIZATION FUNCTIONS
# =========================

def create_performance_visualizations(all_models_data, output_dir, n_trading_days=30):
    """
    Create comprehensive performance visualization charts
    
    Parameters
    ----------
    all_models_data : list
        List of model performance dictionaries
    output_dir : Path
        Directory to save charts
    n_trading_days : int
        Number of trading days for window
        
    Returns
    -------
    None
        Saves PNG files to output_dir
    """
    model_names = [m['name'] for m in all_models_data]
    colors = ['#95a5a6', '#7f8c8d', '#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#27ae60']
    
    # Helper function to create horizontal bar chart
    def create_hbar_chart(data_vals, title, xlabel, filename, 
                         vlines=None, xlim_adjust=0.15, lower_better=False):
        fig, ax = plt.subplots(figsize=(10, 7))
        y_pos = np.arange(len(model_names))
        
        min_val = min(data_vals)
        max_val = max(data_vals)
        x_range = max_val - min_val
        x_min = min_val - 0.1 * abs(x_range) if not lower_better else 0
        x_max = max_val + xlim_adjust * abs(x_range)
        
        bars = ax.barh(y_pos, data_vals, color=colors[:len(model_names)], 
                      alpha=0.8, edgecolor='black', linewidth=0.5)
        
        if vlines:
            for val, color, style, label in vlines:
                ax.axvline(x=val, color=color, linestyle=style, linewidth=1.5, 
                          alpha=0.7, label=label)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        ax.set_xlim([x_min, x_max])
        
        if vlines:
            ax.legend(fontsize=9, loc='lower right')
        
        # Add values at end of bars
        for i, v in enumerate(data_vals):
            format_str = f'  {v:.3f}' if not (lower_better or 'Dir' in xlabel or '%' in xlabel) else f'  {v:.1f}%' if '%' in xlabel else f'  {v:.3f}'
            ax.text(v + 0.01 * x_range, i, format_str, va='center', ha='left', 
                   fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    # Create all charts
    create_hbar_chart(
        [m['raw_r2'] for m in all_models_data],
        'Raw Returns - Test R²', 'Test R²',
        '1_raw_returns_r2.png',
        vlines=[(0, 'red', '--', None)]
    )
    
    create_hbar_chart(
        [m['abn_r2'] for m in all_models_data],
        'Abnormal Returns - Test R²', 'Test R²',
        '2_abnormal_returns_r2.png',
        vlines=[(0, 'red', '--', None)]
    )
    
    # R² Gap (ML models only)
    ml_indices = [i for i, m in enumerate(all_models_data) if not np.isnan(m['raw_gap'])]
    ml_names = [all_models_data[i]['name'] for i in ml_indices]
    gap_vals = [(all_models_data[i]['raw_gap'] + all_models_data[i]['abn_gap']) / 2 
                for i in ml_indices]
    ml_colors = [colors[i] for i in ml_indices]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ml_y_pos = np.arange(len(ml_names))
    max_gap = max(gap_vals)
    
    ax.barh(ml_y_pos, gap_vals, color=ml_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axvline(x=0.25, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Target (<0.25)')
    ax.set_yticks(ml_y_pos)
    ax.set_yticklabels(ml_names, fontsize=10)
    ax.set_xlabel('Avg R² Gap (Train - Test)', fontsize=11, fontweight='bold')
    ax.set_title('Overfitting Check (ML Models Only)', fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    ax.set_xlim([0, max_gap * 1.15])
    
    for i, v in enumerate(gap_vals):
        color = 'darkgreen' if v < 0.25 else 'darkorange' if v < 0.35 else 'darkred'
        ax.text(v + 0.005, i, f'  {v:.3f}', va='center', ha='left', 
               fontsize=10, fontweight='bold', color=color)
    
    plt.tight_layout()
    plt.savefig(output_dir / '3_overfitting_gap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 3_overfitting_gap.png")
    plt.close()
    
    # Directional Accuracy charts
    raw_dir_vals = [m['raw_dir'] * 100 for m in all_models_data]
    abn_dir_vals = [m['abn_dir'] * 100 for m in all_models_data]
    
    for vals, title, filename in [
        (raw_dir_vals, 'Raw Returns - Direction Accuracy', '4_raw_direction_accuracy.png'),
        (abn_dir_vals, 'Abnormal Returns - Direction Accuracy', '5_abnormal_direction_accuracy.png')
    ]:
        fig, ax = plt.subplots(figsize=(10, 7))
        y_pos = np.arange(len(model_names))
        
        ax.barh(y_pos, vals, color=colors[:len(model_names)], alpha=0.8, 
               edgecolor='black', linewidth=0.5)
        ax.axvline(x=50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random (50%)')
        ax.axvline(x=75, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Target (75%)')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names, fontsize=10)
        ax.set_xlabel('Directional Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim([45, 90])
        ax.invert_yaxis()
        
        for i, v in enumerate(vals):
            ax.text(v + 0.5, i, f'  {v:.1f}%', va='center', ha='left', 
                   fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    # MAE and RMSE charts
    for metric, metric_name, file_prefix in [
        ('mae', 'MAE', '7_'),
        ('rmse', 'RMSE', '9_')
    ]:
        for ret_type, title_prefix, file_suffix in [
            ('raw', 'Raw Returns', f'{metric}.png'),
            ('abn', 'Abnormal Returns', f'abnormal_{metric}.png')
        ]:
            vals = [m[f'{ret_type}_{metric}'] for m in all_models_data]
            create_hbar_chart(
                vals,
                f'{title_prefix} - {metric_name.upper()} (Lower is Better)',
                f'{metric_name.upper()}',
                f'{file_prefix}{file_suffix}',
                lower_better=True
            )
    
    print("\n✓ All performance visualizations created!")


def create_ranking_table(scores_df, output_dir):
    """
    Create model rankings table visualization
    
    Parameters
    ----------
    scores_df : pd.DataFrame
        Dataframe with model scores
    output_dir : Path
        Directory to save chart
        
    Returns
    -------
    None
        Saves PNG file to output_dir
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Create ranking table
    table_data = []
    for idx, (_, row) in enumerate(scores_df.iterrows()):
        rank = idx + 1
        model = row['Model']
        r2 = f"{row['Avg R²']:.3f}"
        dir_acc = f"{row['Avg Dir']*100:.1f}%"
        mae = f"{row['Avg MAE']:.3f}"
        rmse = f"{row['Avg RMSE']:.3f}"
        gap = f"{row['Avg Gap']:.3f}" if not np.isnan(row['Avg Gap']) else '-'
        
        table_data.append([f"#{rank}", model, r2, dir_acc, mae, rmse, gap])
    
    table = ax.table(cellText=table_data,
                     colLabels=['Rank', 'Model', 'R²', 'Dir%', 'MAE', 'RMSE', 'Gap'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.10, 0.28, 0.12, 0.12, 0.12, 0.13, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(7):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#2c3e50')
                cell.set_text_props(weight='bold', color='white', size=11)
            else:  # Data rows
                if i == 1:  # Best model
                    cell.set_facecolor('#d5f4e6')
                elif i == 2:  # Second best
                    cell.set_facecolor('#ebf5fb')
                elif i == 3:  # Third best
                    cell.set_facecolor('#fef5e7')
                else:
                    cell.set_facecolor('white')
                    
                if i <= 3 and j == 1:  # Highlight top 3 models
                    cell.set_text_props(weight='bold', size=11)
    
    ax.set_title('Model Rankings\n(40% Direction + 20% R² + 20% MAE + 20% RMSE)', 
                 fontsize=14, fontweight='bold', pad=30)
    
    # Add footnote
    fig.text(0.5, 0.05, 
            'Ranking Criteria: Equal emphasis on directional accuracy (40%) and error metrics (40% total: MAE + RMSE)\n'
            'with R² (20%) for variance explanation. Baselines: Naive (30-day avg), Constant (train mean)',
            ha='center', fontsize=9, style='italic', color='#34495e',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_dir / '11_model_rankings.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 11_model_rankings.png")
    plt.close()