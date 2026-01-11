"""
Main Entry Point for Mining DFS ML Pipeline

This script orchestrates the complete machine learning pipeline for
predicting stock returns following DFS announcements.
"""

from pathlib import Path
import warnings
import pandas as pd
import numpy as np

# Import from our modules
from data_loader import (
    load_data, engineer_features, build_event_data, 
    get_close_series_yf, clean_ticker_column, normalize_dates
)
from preprocessing import preprocess_for_ml, CORE_FEATURES_SCALED, CORE_FEATURES_CATEGORICAL
from models import train_and_evaluate, get_valid_data, create_metrics_df, XGB_PARAMS
from evaluation import (
    compute_naive_benchmarks, regression_metrics, make_metrics_table,
    create_performance_visualizations, create_ranking_table
)

# Suppress warnings
warnings.filterwarnings('ignore')

# =========================
# CONFIGURATION
# =========================

# File paths - UPDATE THESE TO YOUR PATHS
IN_PATH = Path(r"C:\Users\User\Desktop\Test1\dfs_features_safe.xlsx")
OUT_PATH = Path(r"C:\Users\User\Desktop\Test1\ml_output.xlsx")
RESULTS_PATH = Path(r"C:\Users\User\Desktop\Test1\model_results_FINAL_v2.xlsx")

# Create directories if they don't exist
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

# Event study parameters
N_TRADING_DAYS = 30
CALENDAR_BUFFER_DAYS = 260
INDEX_TICKER = "^AXMM"
TEST_FRAC = 0.20
MOMENTUM_WINDOW = 30

# Cache settings - CRITICAL: Use same location as monolithic version
USE_CACHE = True
CACHE_DIR = IN_PATH.parent / "_cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_INPUTS = CACHE_DIR / "inputs_cache.pkl"
CACHE_YF_DIR = CACHE_DIR / "yf"
CACHE_YF_DIR.mkdir(exist_ok=True)


# =========================
# MAIN EXECUTION
# =========================

def main():
    """Main pipeline execution"""
    
    print("=" * 70)
    print("UNIFIED ML PIPELINE - DATA PREP + MODEL TRAINING")
    print("=" * 70)

    # =========================
    # PART 1: DATA PREPARATION
    # =========================
    print("\n" + "=" * 70)
    print("PART 1: DATA PREPARATION & FEATURE ENGINEERING")
    print("=" * 70)

    # 1. Load data - PASS CACHE PARAMETERS EXPLICITLY
    dfs, delisted_map = load_data(
        input_path=IN_PATH, 
        use_cache=USE_CACHE, 
        cache_inputs_path=CACHE_INPUTS
    )

    # 2. Feature engineering
    print("\n⟳ Engineering features...")
    features_df = engineer_features(dfs)

    # Select final features
    final_features = [
        "best_npv_aud", "best_irr_pct", "payback_years", "initial_capex_aud", "mine_life_years",
        "npv_to_mktcap", "npv_to_capex", "annual_revenue_aud", "operating_margin",
        "permitting_status", "npv_flag", "irr_flag", "commodity_price_imputed_flag",
        "primary_metal_major", "metal_bucket"
    ]

    model_df = features_df[["Ticker", "announcement_date"] + final_features].copy()
    print(f"✓ Created {len(final_features)} features for {len(model_df)} records")

    # 3. Download index data - PASS CACHE PARAMETERS
    print("\n⟳ Downloading index data...")
    ann_min = pd.to_datetime(dfs["announcement_date"]).min()
    ann_max = pd.to_datetime(dfs["announcement_date"]).max()
    idx_start = ann_min - pd.Timedelta(days=CALENDAR_BUFFER_DAYS)
    idx_end = ann_max + pd.Timedelta(days=CALENDAR_BUFFER_DAYS)
    index_close_s = get_close_series_yf(
        ticker=INDEX_TICKER, 
        start=idx_start, 
        end=idx_end, 
        use_cache=USE_CACHE, 
        cache_dir=CACHE_YF_DIR
    )

    # 4. Build event data - PASS ALL PARAMETERS
    pivot_df, skip_df = build_event_data(
        dfs=dfs, 
        delisted_map=delisted_map, 
        index_close_s=index_close_s, 
        n_trading_days=N_TRADING_DAYS,
        calendar_buffer_days=CALENDAR_BUFFER_DAYS,
        max_workers=10,
        use_cache=USE_CACHE,
        cache_yf_dir=CACHE_YF_DIR
    )

    # 5. Add benchmarks
    if not pivot_df.empty:
        print("\n⟳ Computing benchmarks...")
        bench = pivot_df.apply(compute_naive_benchmarks, axis=1, n=N_TRADING_DAYS)
        pivot_df = pd.concat([pivot_df, bench], axis=1)

        # 6. Train/test split
        pivot_df = normalize_dates(pivot_df)
        pivot_df = pivot_df.sort_values(["announcement_date", "row_id"]).reset_index(drop=True)

        n_total = len(pivot_df)
        n_test = int(np.ceil(n_total * TEST_FRAC))
        n_train = max(n_total - n_test, 1)

        pivot_df["split"] = np.where(pivot_df.index < n_train, "train", "test")
        train_df = pivot_df[pivot_df["split"] == "train"]

        # Constant baselines from train set
        C_raw = train_df["realized_raw_ret_30d"].dropna().mean()
        C_abn = train_df["realized_abn_ret_30d"].dropna().mean()
        pivot_df["const_pred_raw_ret_30d"] = C_raw
        pivot_df["const_pred_abn_ret_30d"] = C_abn

        print(f"✓ Split: TRAIN={n_train}, TEST={n_test}")
        print(f"  Constant baselines: Raw={C_raw:.6f}, Abn={C_abn:.6f}")

        # 7. Metrics
        print("\n⟳ Computing baseline metrics...")
        eval_specs = [
            ("Raw_30d", "realized_raw_ret_30d", "naive_pred_raw_ret_30d", "NaivePastMean"),
            ("Abnormal_30d", "realized_abn_ret_30d", "naive_pred_abn_ret_30d", "NaivePastMean"),
            ("Raw_30d", "realized_raw_ret_30d", "const_pred_raw_ret_30d", "ConstantTrainMean"),
            ("Abnormal_30d", "realized_abn_ret_30d", "const_pred_abn_ret_30d", "ConstantTrainMean"),
        ]

        overall_metrics_df = make_metrics_table(pivot_df, eval_specs)

        split_frames = []
        for split_name in ["train", "test"]:
            part = pivot_df[pivot_df["split"] == split_name]
            m = make_metrics_table(part, eval_specs)
            m.insert(0, "Split", split_name)
            split_frames.append(m)
        split_metrics_df = pd.concat(split_frames, ignore_index=True)

        print("\n" + "=" * 60)
        print("BASELINE METRICS")
        print("=" * 60)
        print(overall_metrics_df[["Task", "Model", "n", "MAE", "RMSE", "R2", "Corr"]].to_string(index=False))

    # 8. Merge features with events
    print("\n⟳ Creating ML training table...")
    features_df = clean_ticker_column(features_df)
    features_df = normalize_dates(features_df)
    pivot_df = clean_ticker_column(pivot_df, col="ticker")
    pivot_df = normalize_dates(pivot_df)

    ml_df = pivot_df.merge(
        features_df,
        left_on=["ticker", "announcement_date"],
        right_on=["Ticker", "announcement_date"],
        how="left",
        suffixes=("", "_feat")
    )

    print(f"✓ ML table: {len(ml_df)} rows, {len(ml_df.columns)} columns")

    # 9. Preprocess features - PASS OUTPUT DIRECTORY
    print("\n⟳ Preprocessing features for ML...")
    ml_df_processed, summary_df, scaled_cols, cat_cols = preprocess_for_ml(
        ml_df=ml_df, 
        momentum_window=MOMENTUM_WINDOW,
        output_dir=OUT_PATH.parent
    )

    # 10. Save intermediate outputs
    print(f"\n⟳ Saving intermediate outputs to {OUT_PATH}...")
    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        model_df.to_excel(writer, sheet_name="model_features", index=False)
        pivot_df.to_excel(writer, sheet_name="events", index=False)
        ml_df_processed.to_excel(writer, sheet_name="ml_training_table", index=False)
        summary_df.to_excel(writer, sheet_name="preprocessing_summary", index=False)
        
        if not pivot_df.empty:
            overall_metrics_df.to_excel(writer, sheet_name="overall_metrics", index=False)
            split_metrics_df.to_excel(writer, sheet_name="split_metrics", index=False)
        if not skip_df.empty:
            skip_df.to_excel(writer, sheet_name="skipped_events", index=False)

        # ML-ready sheet
        modeling_cols = ['row_id', 'ticker', 'announcement_date', 'split',
                         'realized_raw_ret_30d', 'realized_abn_ret_30d'] + scaled_cols + cat_cols
        ml_ready_df = ml_df_processed[modeling_cols].copy()
        ml_ready_df.to_excel(writer, sheet_name="ml_ready", index=False)

    print(f"✓ Intermediate outputs saved!")

    # =========================
    # PART 2: MODEL TRAINING
    # =========================
    print("\n" + "=" * 70)
    print("PART 2: ML MODEL TRAINING & EVALUATION")
    print("=" * 70)

    print("\n🎯 Goal: Improve Ensemble Directional Accuracy to 75-77%")
    print("         While maintaining Gap < 0.25")

    # Prepare features for modeling
    ml_ready_encoded = ml_ready_df.copy()
    for cat_col in CORE_FEATURES_CATEGORICAL:
        if cat_col in ml_ready_encoded.columns:
            dummies = pd.get_dummies(ml_ready_encoded[cat_col], prefix=cat_col, drop_first=True, dtype=int)
            ml_ready_encoded = pd.concat([ml_ready_encoded, dummies], axis=1)

    feature_cols = CORE_FEATURES_SCALED + [col for col in ml_ready_encoded.columns
                                           if any(col.startswith(cat + '_') for cat in CORE_FEATURES_CATEGORICAL)]

    print(f"\n✓ Total modeling features: {len(feature_cols)}")

    TARGET_RAW = 'realized_raw_ret_30d'
    TARGET_ABN = 'realized_abn_ret_30d'

    train_mask = ml_ready_encoded['split'] == 'train'
    test_mask = ml_ready_encoded['split'] == 'test'

    # Train models for RAW returns
    print("\n" + "=" * 70)
    print("MODELING: RAW RETURNS")
    print("=" * 70)

    X_train_raw, y_train_raw, train_idx_raw = get_valid_data(ml_ready_encoded, train_mask, TARGET_RAW, feature_cols)
    X_test_raw, y_test_raw, test_idx_raw = get_valid_data(ml_ready_encoded, test_mask, TARGET_RAW, feature_cols)

    results_raw, predictions_raw = train_and_evaluate(
        X_train_raw, y_train_raw, X_test_raw, y_test_raw, TARGET_RAW
    )

    # Train models for ABNORMAL returns
    print("\n" + "=" * 70)
    print("MODELING: ABNORMAL RETURNS")
    print("=" * 70)

    X_train_abn, y_train_abn, train_idx_abn = get_valid_data(ml_ready_encoded, train_mask, TARGET_ABN, feature_cols)
    X_test_abn, y_test_abn, test_idx_abn = get_valid_data(ml_ready_encoded, test_mask, TARGET_ABN, feature_cols)

    results_abn, predictions_abn = train_and_evaluate(
        X_train_abn, y_train_abn, X_test_abn, y_test_abn, TARGET_ABN
    )

    # Calculate summary metrics
    avg_xgb_gap = (results_raw['XGBoost']['r2_gap'] + results_abn['XGBoost']['r2_gap']) / 2
    avg_xgb_dir = (results_raw['XGBoost']['test_dir_acc'] + results_abn['XGBoost']['test_dir_acc']) / 2
    avg_ens_gap = (results_raw['Ensemble_Top3']['r2_gap'] + results_abn['Ensemble_Top3']['r2_gap']) / 2
    avg_ens_dir = (results_raw['Ensemble_Top3']['test_dir_acc'] + results_abn['Ensemble_Top3']['test_dir_acc']) / 2

    # Create metrics dataframes
    metrics_raw_df = create_metrics_df(results_raw, TARGET_RAW, len(X_train_raw), len(X_test_raw))
    metrics_abn_df = create_metrics_df(results_abn, TARGET_ABN, len(X_train_abn), len(X_test_abn))
    metrics_all_df = pd.concat([metrics_raw_df, metrics_abn_df], ignore_index=True)

    # =========================
    # VISUALIZATIONS
    # =========================
    print("\n⟳ Creating comprehensive performance visualization...")
    
    # Prepare data for all models including baselines
    all_models_data = []
    
    # Naive Past Mean
    test_mask_piv = pivot_df['split'] == 'test'
    naive_metrics_raw = regression_metrics(
        pivot_df[test_mask_piv]['realized_raw_ret_30d'], 
        pivot_df[test_mask_piv]['naive_pred_raw_ret_30d']
    )
    naive_metrics_abn = regression_metrics(
        pivot_df[test_mask_piv]['realized_abn_ret_30d'], 
        pivot_df[test_mask_piv]['naive_pred_abn_ret_30d']
    )
    all_models_data.append({
        'name': 'Naive Past Mean',
        'raw_r2': naive_metrics_raw['R2'],
        'abn_r2': naive_metrics_abn['R2'],
        'raw_dir': naive_metrics_raw['Directional_Acc'],
        'abn_dir': naive_metrics_abn['Directional_Acc'],
        'raw_mae': naive_metrics_raw['MAE'],
        'abn_mae': naive_metrics_abn['MAE'],
        'raw_rmse': naive_metrics_raw['RMSE'],
        'abn_rmse': naive_metrics_abn['RMSE'],
        'raw_gap': np.nan,
        'abn_gap': np.nan,
    })
    
    # Constant Mean
    const_metrics_raw = regression_metrics(
        pivot_df[test_mask_piv]['realized_raw_ret_30d'], 
        pivot_df[test_mask_piv]['const_pred_raw_ret_30d']
    )
    const_metrics_abn = regression_metrics(
        pivot_df[test_mask_piv]['realized_abn_ret_30d'], 
        pivot_df[test_mask_piv]['const_pred_abn_ret_30d']
    )
    all_models_data.append({
        'name': 'Constant Mean',
        'raw_r2': const_metrics_raw['R2'],
        'abn_r2': const_metrics_abn['R2'],
        'raw_dir': const_metrics_raw['Directional_Acc'],
        'abn_dir': const_metrics_abn['Directional_Acc'],
        'raw_mae': const_metrics_raw['MAE'],
        'abn_mae': const_metrics_abn['MAE'],
        'raw_rmse': const_metrics_raw['RMSE'],
        'abn_rmse': const_metrics_abn['RMSE'],
        'raw_gap': np.nan,
        'abn_gap': np.nan,
    })
    
    # ML Models
    for model_name in ['Random Forest', 'XGBoost', 'Ridge', 'OLS', 'Ensemble_Top3']:
        display_name = 'Ensemble Top-3' if model_name == 'Ensemble_Top3' else model_name
        all_models_data.append({
            'name': display_name,
            'raw_r2': results_raw[model_name]['test_metrics']['R2'],
            'abn_r2': results_abn[model_name]['test_metrics']['R2'],
            'raw_dir': results_raw[model_name]['test_dir_acc'],
            'abn_dir': results_abn[model_name]['test_dir_acc'],
            'raw_mae': results_raw[model_name]['test_metrics']['MAE'],
            'abn_mae': results_abn[model_name]['test_metrics']['MAE'],
            'raw_rmse': results_raw[model_name]['test_metrics']['RMSE'],
            'abn_rmse': results_abn[model_name]['test_metrics']['RMSE'],
            'raw_gap': results_raw[model_name]['r2_gap'],
            'abn_gap': results_abn[model_name]['r2_gap'],
        })
    
    # Create visualizations
    create_performance_visualizations(all_models_data, OUT_PATH.parent, N_TRADING_DAYS)
    
    # Calculate rankings
    scores = []
    for m in all_models_data:
        avg_r2 = (m['raw_r2'] + m['abn_r2']) / 2
        avg_dir = (m['raw_dir'] + m['abn_dir']) / 2
        avg_mae = (m['raw_mae'] + m['abn_mae']) / 2
        avg_rmse = (m['raw_rmse'] + m['abn_rmse']) / 2
        avg_gap = (m['raw_gap'] + m['abn_gap']) / 2 if not np.isnan(m['raw_gap']) else np.nan
        
        # Composite score: Direction (40%), R² (20%), MAE (20%), RMSE (20%)
        dir_component = avg_dir * 0.40
        r2_component = max(0, min(avg_r2, 0.3)) / 0.3 * 0.20
        mae_component = max(0, 1 - avg_mae/0.5) * 0.20
        rmse_component = max(0, 1 - avg_rmse/0.5) * 0.20
        
        composite = dir_component + r2_component + mae_component + rmse_component
        
        scores.append({
            'Model': m['name'],
            'Avg R²': avg_r2,
            'Avg Dir': avg_dir,
            'Avg MAE': avg_mae,
            'Avg RMSE': avg_rmse,
            'Avg Gap': avg_gap,
            'Score': composite
        })
    
    scores_df = pd.DataFrame(scores).sort_values('Score', ascending=False)
    
    # Create ranking table
    create_ranking_table(scores_df, OUT_PATH.parent)
    
    # Print detailed ranking
    print("\n" + "=" * 100)
    print("MODEL PERFORMANCE SUMMARY (Test Set) - RANKED BY COMPOSITE SCORE")
    print("=" * 100)
    print(f"{'Rank':<6} {'Model':<18} {'Avg R²':<10} {'Avg Dir':<10} {'Avg MAE':<10} {'Avg RMSE':<11} {'Avg Gap':<10} {'Score':<8}")
    print("-" * 100)
    for idx, (_, row) in enumerate(scores_df.iterrows()):
        rank = f"#{idx+1}"
        model = row['Model']
        r2 = f"{row['Avg R²']:.3f}"
        dir_acc = f"{row['Avg Dir']*100:.1f}%"
        mae = f"{row['Avg MAE']:.3f}"
        rmse = f"{row['Avg RMSE']:.3f}"
        gap = f"{row['Avg Gap']:.3f}" if not np.isnan(row['Avg Gap']) else '-'
        score = f"{row['Score']:.3f}"
        print(f"{rank:<6} {model:<18} {r2:<10} {dir_acc:<10} {mae:<10} {rmse:<11} {gap:<10} {score:<8}")
    print("=" * 100)

    # Save final results
    print(f"\n⟳ Saving final results to {RESULTS_PATH}...")

    with pd.ExcelWriter(RESULTS_PATH, engine='openpyxl') as writer:
        metrics_all_df.to_excel(writer, sheet_name='model_metrics', index=False)

        # Summary metrics
        summary_rows = [
            {'Model': 'XGBoost', 'Metric': 'Avg_Gap', 'Value': avg_xgb_gap},
            {'Model': 'XGBoost', 'Metric': 'Avg_Direction', 'Value': avg_xgb_dir},
            {'Model': 'Ensemble', 'Metric': 'Avg_Gap', 'Value': avg_ens_gap},
            {'Model': 'Ensemble', 'Metric': 'Avg_Direction', 'Value': avg_ens_dir},
        ]
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name='summary', index=False)

        # Hyperparameters
        params_info = pd.DataFrame({
            'Model': ['XGBoost'] * len(XGB_PARAMS),
            'Parameter': list(XGB_PARAMS.keys()),
            'Value': [str(v) for v in XGB_PARAMS.values()]
        })
        params_info.to_excel(writer, sheet_name='xgb_params', index=False)

    print("✓ Final results saved!")

    # =========================
    # FINAL SUMMARY
    # =========================
    print("\n" + "=" * 70)
    print("🎯 FINAL RESULTS")
    print("=" * 70)

    print("\n📊 ENSEMBLE (Recommended Model):")
    print(f"  Raw Returns:")
    print(f"    Test R² = {results_raw['Ensemble_Top3']['test_metrics']['R2']:.4f}")
    print(f"    Gap = {results_raw['Ensemble_Top3']['r2_gap']:.4f}")
    print(f"    Direction = {results_raw['Ensemble_Top3']['test_dir_acc']:.2%}")

    print(f"\n  Abnormal Returns:")
    print(f"    Test R² = {results_abn['Ensemble_Top3']['test_metrics']['R2']:.4f}")
    print(f"    Gap = {results_abn['Ensemble_Top3']['r2_gap']:.4f}")
    print(f"    Direction = {results_abn['Ensemble_Top3']['test_dir_acc']:.2%}")

    print(f"\n  Average Metrics:")
    print(f"    Gap = {avg_ens_gap:.4f} {'✓ PASS' if avg_ens_gap < 0.25 else '✗ FAIL'} (Target: < 0.25)")
    print(f"    Direction = {avg_ens_dir:.2%} {'✓ PASS' if avg_ens_dir >= 0.75 else '⚠ CLOSE'} (Target: 75-77%)")

    print("\n📁 OUTPUTS CREATED:")
    print(f"  1. {OUT_PATH} (preprocessing & intermediate data)")
    print(f"  2. {RESULTS_PATH} (model results)")
    print(f"  3. {OUT_PATH.parent / 'features_original.png'}")
    print(f"  4. {OUT_PATH.parent / 'features_scaled.png'}")
    print(f"\n  📊 MODEL COMPARISON CHARTS (10 total):")
    print(f"  5-14. Performance visualization charts")

    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()