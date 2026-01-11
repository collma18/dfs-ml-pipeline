"""
Data Loading and Feature Engineering Module

Handles all data loading, preprocessing, unit conversions, 
and feature engineering for the ML pipeline.
"""

from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import pickle
import hashlib
from typing import Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf
import logging

# Suppress warnings
warnings.filterwarnings("ignore", message=".*Timestamp\\.utcnow is deprecated.*", category=Warning)
warnings.filterwarnings('ignore')
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# =========================
# UNIT CONVERSION MAPPINGS
# =========================

UNIT_CONVERSION = {
    "aud/t": 1.0, 
    "aud/kg": 1000.0, 
    "aud/lb": 2204.62262, 
    "aud/oz": 35273.9619,
}

PROD_CONVERSION = {
    "t/y": 1.0, 
    "kg/y": 1.0 / 1000.0, 
    "lb/y": 1.0 / 2204.62262, 
    "oz/y": 1.0 / 35273.9619,
}

PRICE_CONVERSION = UNIT_CONVERSION.copy()

MAJOR_MAP = {
    "gold": "gold", 
    "lithium": "lithium", 
    "copper": "copper",
    "graphite": "graphite", 
    "iron ore": "iron ore", 
    "nickel": "nickel",
}

BUCKET_MAP = {
    "gold": "precious",
    "lithium": "battery", "nickel": "battery", "graphite": "battery",
    "vanadium": "battery", "cobalt": "battery",
    "copper": "base", "zinc": "base", "lead": "base", "tin": "base", "molybdenum": "base",
    "iron ore": "bulk", "coal": "bulk", "bauxite": "bulk", "alumina": "bulk",
    "potash": "bulk", "phosphate": "bulk", "kaolin": "bulk",
    "ndpr": "critical", "dysprosium": "critical", "scandium": "critical",
    "rare earths": "critical", "niobium": "critical", "zirconium": "critical", "uranium": "critical",
    "tungsten": "specialty", "zircon": "specialty",
}


# =========================
# CACHE HELPER FUNCTIONS
# =========================

def file_fingerprint(path: Path) -> str:
    """Generate fingerprint based on file metadata"""
    stat = path.stat()
    raw = f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_pickle(path: Path):
    """Load pickled object from file"""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: Path):
    """Save object to pickle file"""
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def yf_cache_path(ticker: str, start: pd.Timestamp, end: pd.Timestamp, cache_dir: Path) -> Path:
    """Generate cache file path for Yahoo Finance data"""
    key = f"{ticker}__{start.date()}__{end.date()}".encode("utf-8")
    name = hashlib.md5(key).hexdigest() + ".pkl"
    return cache_dir / name


# =========================
# DATA CLEANING HELPERS
# =========================

def select_best_value(df: pd.DataFrame, post_col: str, pre_col: str,
                      output_col: str, flag_col: str) -> pd.DataFrame:
    """Select post-tax value if available, else pre-tax, with flag"""
    df[post_col] = pd.to_numeric(df[post_col], errors="coerce")
    df[pre_col] = pd.to_numeric(df[pre_col], errors="coerce")

    df[output_col] = df[post_col].fillna(df[pre_col])
    df[flag_col] = pd.Series(pd.NA, index=df.index, dtype="Int8")
    df.loc[df[post_col].notna(), flag_col] = 1
    df.loc[df[post_col].isna() & df[pre_col].notna(), flag_col] = 0
    return df


def normalize_units(df: pd.DataFrame, value_col: str, unit_col: str,
                    conversion_map: dict, output_col: str) -> pd.DataFrame:
    """Generic unit normalization to standard units"""
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df[unit_col] = df[unit_col].astype(str).str.strip().str.lower()
    factor = df[unit_col].map(conversion_map)
    df[output_col] = df[value_col] * factor
    return df


def clean_ticker_column(df: pd.DataFrame, col: str = "Ticker") -> pd.DataFrame:
    """Standardize ticker column"""
    df[col] = df[col].astype(str).str.strip().str.upper()
    return df


def normalize_dates(df: pd.DataFrame, date_col: str = "announcement_date") -> pd.DataFrame:
    """Normalize datetime column"""
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    return df


# =========================
# DATA LOADING
# =========================

def load_data(input_path: Path, use_cache: bool = True, 
              cache_inputs_path: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """
    Load and cache Excel data
    
    Parameters
    ----------
    input_path : Path
        Path to Excel file with DFS data
    use_cache : bool
        Whether to use cached data
    cache_inputs_path : Path, optional
        Path to cache file
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, pd.Series]]
        DFS dataframe and delisted price series dict
    """
    excel_fp = file_fingerprint(input_path)

    if use_cache and cache_inputs_path and cache_inputs_path.exists():
        cached = load_pickle(cache_inputs_path)
        if cached.get("excel_fp") == excel_fp:
            print("✓ Loaded data from cache")
            return cached["dfs"], cached["delisted_map"]

    print("⟳ Loading data from Excel...")
    dfs = pd.read_excel(input_path, sheet_name="DFS Data")
    delisted = pd.read_excel(input_path, sheet_name="Delisted Share Prices")

    # Process DFS data
    dfs = clean_ticker_column(dfs.assign(Ticker=dfs.iloc[:, 2]))
    dfs = normalize_dates(dfs.assign(announcement_date=dfs.iloc[:, 3]))

    # Process delisted data
    delisted = delisted.assign(
        Ticker=delisted.iloc[:, 0],
        Date=pd.to_datetime(delisted.iloc[:, 1], errors="coerce").dt.normalize(),
        Close=pd.to_numeric(delisted.iloc[:, 5], errors="coerce")
    )
    delisted = clean_ticker_column(delisted)

    d0 = delisted.dropna(subset=["Ticker", "Date", "Close"])
    delisted_map = {
        t: g.drop_duplicates("Date").set_index("Date")["Close"].sort_index().astype(float)
        for t, g in d0.groupby("Ticker")
    }

    if use_cache and cache_inputs_path:
        save_pickle({"excel_fp": excel_fp, "dfs": dfs, "delisted_map": delisted_map}, cache_inputs_path)

    return dfs, delisted_map


# =========================
# FEATURE ENGINEERING
# =========================

def impute_commodity_prices(features_df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing commodity prices using nearest same-metal observation"""
    features_df["commodity_price_missing_flag"] = features_df["commodity_price_aud_per_ton"].isna().astype("Int8")

    known = features_df.loc[
        features_df["commodity_price_aud_per_ton"].notna() &
        features_df["primary_metal_clean"].notna() &
        features_df["announcement_date"].notna(),
        ["primary_metal_clean", "announcement_date", "commodity_price_aud_per_ton"]
    ].copy()

    need = features_df.loc[
        features_df["commodity_price_aud_per_ton"].isna() &
        features_df["primary_metal_clean"].notna() &
        features_df["announcement_date"].notna(),
        ["row_id", "primary_metal_clean", "announcement_date"]
    ].copy()

    imputed_parts = []
    for metal, need_g in need.groupby("primary_metal_clean", sort=False):
        known_g = known.loc[known["primary_metal_clean"] == metal].copy()
        need_g = need_g.sort_values("announcement_date").reset_index(drop=True)
        known_g = known_g.sort_values("announcement_date").reset_index(drop=True)

        if known_g.empty:
            out_g = need_g.copy()
            out_g["commodity_price_aud_per_ton"] = np.nan
        else:
            out_g = pd.merge_asof(
                need_g, known_g[["announcement_date", "commodity_price_aud_per_ton"]],
                on="announcement_date", direction="nearest", tolerance=pd.Timedelta(days=365)
            )
        imputed_parts.append(out_g)

    imputed = pd.concat(imputed_parts, ignore_index=True) if imputed_parts else pd.DataFrame()

    features_df = features_df.merge(
        imputed[["row_id", "commodity_price_aud_per_ton"]], on="row_id", how="left", suffixes=("", "_imputed")
    )

    features_df["commodity_price_aud_per_ton_filled"] = features_df["commodity_price_aud_per_ton"]
    fill_mask = (
            features_df["commodity_price_aud_per_ton_filled"].isna() &
            features_df["commodity_price_aud_per_ton_imputed"].notna()
    )
    features_df.loc[fill_mask, "commodity_price_aud_per_ton_filled"] = features_df.loc[
        fill_mask, "commodity_price_aud_per_ton_imputed"]

    features_df["commodity_price_imputed_flag"] = pd.Series(pd.NA, index=features_df.index, dtype="Int8")
    features_df.loc[features_df["commodity_price_aud_per_ton"].notna(), "commodity_price_imputed_flag"] = 0
    features_df.loc[
        features_df["commodity_price_aud_per_ton"].isna() &
        features_df["commodity_price_aud_per_ton_filled"].notna(),
        "commodity_price_imputed_flag"
    ] = 1

    features_df["annual_revenue_aud"] = features_df["annual_production_tpy"] * features_df[
        "commodity_price_aud_per_ton_filled"]
    features_df.drop(columns=["commodity_price_aud_per_ton_imputed"], inplace=True)

    return features_df


def engineer_features(dfs: pd.DataFrame) -> pd.DataFrame:
    """Create all ML features"""
    features_df = dfs.reset_index(drop=True).copy()
    features_df["row_id"] = features_df.index

    # Feature 1 & 2: Best NPV and IRR with flags
    select_best_value(features_df, "npv_post_tax_aud", "npv_pre_tax_aud", "best_npv_aud", "npv_flag")
    select_best_value(features_df, "irr_post_tax_pct", "irr_pre_tax_pct", "best_irr_pct", "irr_flag")

    # Feature 3: Payback period
    features_df["payback_years"] = pd.to_numeric(features_df["payback_years"], errors="coerce")

    # Features 4-6: Unit normalizations
    normalize_units(features_df, "aisc_aud_per_unit", "aisc_unit", UNIT_CONVERSION, "aisc_aud_per_ton")
    normalize_units(features_df, "annual_production", "annual_production_unit", PROD_CONVERSION,
                    "annual_production_tpy")
    normalize_units(features_df, "base_case_commodity_price_aud_per_unit", "commodity_price_unit",
                    PRICE_CONVERSION, "commodity_price_aud_per_ton")

    # Create primary_metal_clean BEFORE imputation (needed for grouping)
    pm = features_df["primary_metal"].astype("string").str.strip().str.lower()
    features_df["primary_metal_clean"] = pm

    # Feature 7: Commodity price imputation
    features_df = impute_commodity_prices(features_df)

    # Feature 8-11: Simple conversions
    features_df["market_cap_aud"] = pd.to_numeric(features_df["mkt_cap_aud"], errors="coerce")
    features_df["mine_life_years"] = pd.to_numeric(features_df["mine_life_years"], errors="coerce")
    features_df["initial_capex_aud"] = pd.to_numeric(features_df["initial_capex_aud"], errors="coerce")
    features_df["permitting_status"] = pd.to_numeric(features_df["permitting_status"], errors="coerce").astype("Int8")

    # Feature 12: Operating margin
    price = features_df["commodity_price_aud_per_ton_filled"]
    cost = features_df["aisc_aud_per_ton"]
    features_df["operating_margin"] = np.where(
        (price.notna()) & (cost.notna()) & (price != 0),
        (price - cost) / price, np.nan
    )

    # Metal categorization
    features_df["primary_metal_major"] = np.where(pm.isna(), "unknown", pm.map(MAJOR_MAP).fillna("other"))
    features_df["metal_bucket"] = np.where(pm.isna(), "unknown", pm.map(BUCKET_MAP).fillna("other"))

    # Ratio features
    npv = features_df["best_npv_aud"]
    mkt = features_df["market_cap_aud"]
    capex = features_df["initial_capex_aud"]

    features_df["npv_to_mktcap"] = np.where(npv.notna() & mkt.notna() & (mkt != 0), npv / mkt, np.nan)
    features_df["npv_to_capex"] = np.where(npv.notna() & capex.notna() & (capex != 0), npv / capex, np.nan)

    return features_df


# =========================
# MARKET DATA
# =========================

def get_close_series_yf(ticker: str, start: pd.Timestamp, end: pd.Timestamp,
                        use_cache: bool = True, cache_dir: Optional[Path] = None) -> pd.Series:
    """
    Download and cache yfinance data
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    start : pd.Timestamp
        Start date
    end : pd.Timestamp
        End date
    use_cache : bool
        Whether to use cached data
    cache_dir : Path, optional
        Cache directory path
        
    Returns
    -------
    pd.Series
        Price series with dates as index
    """
    if use_cache and cache_dir:
        cache_file = yf_cache_path(ticker, start, end, cache_dir)
        if cache_file.exists():
            try:
                s = load_pickle(cache_file)
                if isinstance(s, pd.Series):
                    return s
            except:
                pass

    try:
        data = yf.download(ticker, start=str(start.date()),
                           end=str((end + pd.Timedelta(days=1)).date()),
                           progress=False, auto_adjust=False, threads=True)

        if data is None or data.empty:
            s = pd.Series(dtype=float)
        else:
            close = (data.xs("Close", axis=1, level=0, drop_level=True).iloc[:, 0]
                     if isinstance(data.columns, pd.MultiIndex) else data["Close"])
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            s = close.dropna().astype(float)
            s.index = pd.to_datetime(s.index).normalize()
            s = s.sort_index()
    except:
        s = pd.Series(dtype=float)

    if use_cache and cache_dir:
        try:
            save_pickle(s, cache_file)
        except:
            pass

    return s


def build_event_window(close_s: pd.Series, ann_date: pd.Timestamp, n: int):
    """
    Build event window around announcement date
    
    Parameters
    ----------
    close_s : pd.Series
        Price series
    ann_date : pd.Timestamp
        Announcement date
    n : int
        Number of trading days before/after
        
    Returns
    -------
    Tuple
        (t0_date, window_series, window_dates, error_reason)
    """
    if close_s is None or close_s.empty or pd.isna(ann_date):
        return None, None, None, "no_price_or_bad_date"

    close_s = close_s.dropna().sort_index()
    dates = close_s.index

    future = dates[dates >= ann_date]
    t0 = future[0] if len(future) else dates[dates < ann_date][-1] if len(dates[dates < ann_date]) else None

    if t0 is None:
        return None, None, None, "no_trading_day_around_announcement"

    pos = dates.get_loc(t0)
    if pos < n or (pos + n) >= len(dates):
        return None, None, None, "insufficient_trading_days"

    win_slice = close_s.iloc[pos - n: pos + n + 1].copy()
    win_dates = win_slice.index.copy()
    win_slice.index = range(-n, n + 1)

    return t0, win_slice, win_dates, None


def download_ticker_data(args):
    """
    Download data for a single ticker (for parallel processing)
    
    Parameters
    ----------
    args : tuple
        (row_id, ticker, announcement_date, delisted_map, n_days, calendar_buffer, 
         use_cache, cache_dir)
        
    Returns
    -------
    dict
        Download results and event window data
    """
    row_id, ticker, ann, delisted_map, n_days, calendar_buffer, use_cache, cache_dir = args

    if pd.isna(ann) or not ticker or str(ticker).strip() == "":
        return None

    ticker_ax = f"{ticker}.AX"
    start = ann - pd.Timedelta(days=calendar_buffer)
    end = ann + pd.Timedelta(days=calendar_buffer)

    close_s = get_close_series_yf(ticker_ax, start, end, use_cache, cache_dir)
    source = "yfinance"

    if close_s.empty:
        close_s = delisted_map.get(ticker, pd.Series(dtype=float))
        source = "delisted"

    t0, win, win_dates, reason = build_event_window(close_s, ann, n_days)

    if win is None and source == "yfinance":
        alt = delisted_map.get(ticker, pd.Series(dtype=float))
        t0_alt, win_alt, win_dates_alt, reason_alt = build_event_window(alt, ann, n_days)
        if win_alt is not None:
            t0, win, win_dates, source, reason = t0_alt, win_alt, win_dates_alt, "delisted", None

    return {
        'row_id': row_id, 'ticker': ticker, 'ann': ann, 't0': t0,
        'win': win, 'win_dates': win_dates, 'source': source, 'reason': reason
    }


def build_event_data(dfs: pd.DataFrame, delisted_map: Dict, index_close_s: pd.Series,
                    n_trading_days: int = 30, calendar_buffer_days: int = 260, 
                    max_workers: int = 10, use_cache: bool = True, 
                    cache_yf_dir: Optional[Path] = None):
    """
    Build event study data with parallel downloads
    
    Parameters
    ----------
    dfs : pd.DataFrame
        DFS data with tickers and announcement dates
    delisted_map : Dict
        Delisted stock price series
    index_close_s : pd.Series
        Market index price series
    n_trading_days : int
        Event window size
    calendar_buffer_days : int
        Days to download before/after announcement
    max_workers : int
        Number of parallel download workers
    use_cache : bool
        Whether to use cached data
    cache_yf_dir : Path, optional
        Yahoo Finance cache directory
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Event data and skipped events
    """
    print("⟳ Downloading market data...")

    dfs_iter = dfs.reset_index(drop=True)
    
    args_list = [(i, r["Ticker"], r["announcement_date"], delisted_map, 
                  n_trading_days, calendar_buffer_days, use_cache, cache_yf_dir)
                 for i, r in dfs_iter.iterrows()]

    rows, skip_log = [], []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_ticker_data, args): args for args in args_list}

        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue

            if result['win'] is None:
                skip_log.append({
                    'row_id': result['row_id'], 'ticker': result['ticker'],
                    'announcement_date': result['ann'], 'reason': result['reason']
                })
                continue

            # Align index to stock window
            xmm_on_dates = index_close_s.reindex(result['win_dates']).ffill()
            xmm_win = pd.Series(xmm_on_dates.values, index=range(-n_trading_days, n_trading_days + 1))

            row = {
                'row_id': result['row_id'], 'ticker': result['ticker'],
                'announcement_date': result['ann'], 't0_trading_date': result['t0'],
                'source': result['source'],
                **{f"close_t{t}": float(v) for t, v in result['win'].items()},
                **{f"xmm_close_t{t}": (float(v) if pd.notna(v) else None) for t, v in xmm_win.items()}
            }
            rows.append(row)

    pivot_df = pd.DataFrame(rows)
    skip_df = pd.DataFrame(skip_log)

    # Reorder columns
    base_cols = ["row_id", "ticker", "announcement_date", "t0_trading_date", "source"]
    t_cols = [f"close_t{t}" for t in range(-n_trading_days, n_trading_days + 1)]
    xmm_cols = [f"xmm_close_t{t}" for t in range(-n_trading_days, n_trading_days + 1)]

    if not pivot_df.empty:
        pivot_df = pivot_df[base_cols +
                            [c for c in t_cols if c in pivot_df.columns] +
                            [c for c in xmm_cols if c in pivot_df.columns]]

    print(f"✓ Events: {len(pivot_df)} | Skipped: {len(skip_df)}")
    return pivot_df, skip_df