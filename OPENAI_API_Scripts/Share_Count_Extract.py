"""
ASX shares outstanding near an announcement date (Excel -> Excel) - PARALLEL VERSION
====================================================================================

What it does
- Reads your Excel file (e.g. C:\\Users\\User\\Desktop\\Book1.xlsx)
- For each row (Company, Ticker, announcement_date), searches ASX sources IN PARALLEL
- Prefers Appendix 2A (Application for quotation) and Appendix 3H (Cessation)
  because they often contain tables with "Total number of securities on issue"
- Falls back to other ASX docs (3B/3C/4A/4E, annual report, etc.)
- Writes results back to a new Excel file with source + as-of date + notes

Install
  pip install openai pandas openpyxl python-dateutil yfinance

Auth
  Set OPENAI_API_KEY as an environment variable (recommended)
  PowerShell:
    setx OPENAI_API_KEY "YOUR_KEY"
  (Open a new terminal after setx)

Important
- If you get error 429 insufficient_quota, you need to enable billing / add credits.
- Mini model is recommended for cost. You can switch to gpt-5.2 if needed.
- Adjust MAX_WORKERS based on your rate limits (start with 3-5)

Expected input columns (case-insensitive):
- Company (or "Comapany" typo supported)
- Ticker
- announcement_date
"""

import os
import re
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple

import pandas as pd
import yfinance as yf
from dateutil import parser as dateparser
from openai import OpenAI


# ------------------------- CONFIG -------------------------

# Change these paths to match your machine
INPUT_XLSX = r"C:\Users\User\Desktop\MoreDFS\downloaded_pdfs\tickershares.xlsx"
OUTPUT_XLSX = r"C:\Users\User\Desktop\MoreDFS\downloaded_pdfs\Book1_with_shares_outstanding.xlsx"

# Cheaper model (recommended)
MODEL = "gpt-5-mini"
# MODEL = "gpt-5.2"  # use if you want a stronger (more expensive) model

# ASX domain filtering (keeps sources relevant)
ALLOWED_DOMAINS = [
    "asx.com.au",
    "www.asx.com.au",
    "www2.asx.com.au",
]

# Parallel processing (tune to your rate limits)
MAX_WORKERS = 4  # Start with 3-5, increase if you have good rate limits

# If the model returns malformed JSON, retry
MAX_JSON_RETRIES = 2

# Retry config for rate limits
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 1.5

# ---------------------------------------------------------


@dataclass
class SharesResult:
    shares_outstanding: Optional[int] = None
    shares_asof_date: Optional[str] = None  # ISO date string
    source_url: Optional[str] = None
    source_title: Optional[str] = None
    source_domain: Optional[str] = None
    confidence: str = "low"  # low/medium/high
    method_notes: str = ""
    yfinance_data_available: str = "Unknown"  # Yes/No/Unknown


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {c.lower().strip(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in colmap:
                return colmap[n]
        return None

    company_col = pick("company", "comapany", "co", "name")
    ticker_col = pick("ticker", "asx ticker", "code", "symbol")
    date_col = pick("announcement_date", "announcement date", "date", "ann_date")

    missing = [("Company", company_col), ("Ticker", ticker_col), ("announcement_date", date_col)]
    missing = [k for k, v in missing if v is None]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    return df.rename(columns={company_col: "Company", ticker_col: "Ticker", date_col: "announcement_date"})


def _parse_date_to_iso(d) -> Optional[str]:
    if pd.isna(d):
        return None
    if isinstance(d, pd.Timestamp):
        return d.date().isoformat()
    try:
        return dateparser.parse(str(d), dayfirst=False).date().isoformat()
    except Exception:
        return None


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        return None


def _coerce_int_shares(x) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)

    s = str(x).strip()
    s = re.sub(r"[,\s]", "", s)
    s = re.sub(r"shares?$", "", s, flags=re.IGNORECASE)

    # handle 1.2b / 450m / 10k
    m = re.match(r"^(\d+(\.\d+)?)([kmb])?$", s, flags=re.IGNORECASE)
    if m:
        num = float(m.group(1))
        suf = (m.group(3) or "").lower()
        mult = {"": 1, "k": 1_000, "m": 1_000_000, "b": 1_000_000_000}[suf]
        return int(round(num * mult))

    digits = re.findall(r"\d+", s)
    if not digits:
        return None
    return int("".join(digits))


def _domain_from_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    m = re.search(r"https?://([^/]+)/?", url)
    return m.group(1).lower() if m else None


def _is_quota_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("insufficient_quota" in msg) or ("exceeded your current quota" in msg)


def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("rate_limit" in msg) or ("429" in msg) or ("too many requests" in msg)


def check_yfinance_data(ticker: str, announcement_date: str) -> str:
    """
    Check if yfinance has price data 30 days before and after announcement date.
    Returns: "Yes", "No", or "Unknown"

    Args:
        ticker: ASX ticker (e.g., "CBA")
        announcement_date: ISO date string (YYYY-MM-DD)
    """
    if not ticker or not announcement_date:
        print(f"  [YF] Skipping - missing ticker or date")
        return "Unknown"

    try:
        # Parse announcement date
        ann_dt = datetime.fromisoformat(announcement_date)

        # Calculate date range (30 days before and after)
        start_date = ann_dt - timedelta(days=30)
        end_date = ann_dt + timedelta(days=30)

        # ASX tickers need .AX suffix for yfinance
        ticker_yf = f"{ticker.upper().strip()}.AX"

        print(f"  [YF] Checking {ticker_yf} from {start_date.date()} to {end_date.date()}")

        # Fetch data from yfinance (minimal parameters for compatibility)
        stock = yf.Ticker(ticker_yf)
        hist = stock.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d")
        )

        # Check if we have sufficient data
        if hist is None or hist.empty:
            print(f"  [YF] No data returned for {ticker_yf}")
            return "No"

        print(f"  [YF] Got {len(hist)} rows of data for {ticker_yf}")

        # Check if we have data both before and after announcement date
        dates = pd.to_datetime(hist.index).date
        has_before = any(d < ann_dt.date() for d in dates)
        has_after = any(d > ann_dt.date() for d in dates)

        print(f"  [YF] Before ann date: {has_before}, After ann date: {has_after}")

        if has_before and has_after:
            return "Yes"
        else:
            return "No"

    except Exception as e:
        # If there's any error (ticker not found, network issue, etc.), return Unknown
        print(f"  [YF] ERROR for {ticker}: {type(e).__name__}: {str(e)[:100]}")
        return "Unknown"


def backoff_sleep(attempt: int) -> None:
    """Exponential backoff with jitter"""
    sleep_s = BASE_BACKOFF_SECONDS * (2 ** attempt) * (0.7 + random.random() * 0.6)
    time.sleep(min(sleep_s, 30))


def call_with_retries(fn, *args, **kwargs):
    """Retry wrapper for transient errors"""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            # Don't retry quota errors
            if _is_quota_error(e):
                raise
            # Retry rate limits and transient errors
            if _is_rate_limit_error(e) or "5" in str(getattr(e, "status_code", "")):
                if attempt < MAX_RETRIES - 1:
                    backoff_sleep(attempt)
                    continue
            # For other errors, raise immediately
            raise
    raise last_err


def _run_responses_json(
    client: OpenAI,
    prompt_text: str,
    tools: list,
    include: list,
) -> Dict[str, Any]:
    """
    Call OpenAI Responses API and parse a JSON object from output_text.
    Retries if response isn't valid JSON.
    """
    last_err = None
    p = prompt_text
    for attempt in range(MAX_JSON_RETRIES + 1):
        try:
            def _call():
                return client.responses.create(
                    model=MODEL,
                    tools=tools,
                    tool_choice="auto",
                    include=include,
                    input=p,
                )

            resp = call_with_retries(_call)
            text = getattr(resp, "output_text", "") or ""
            data = _extract_first_json_object(text)
            if not isinstance(data, dict):
                raise ValueError(f"Model did not return JSON. Output (truncated): {text[:300]}")
            return data
        except Exception as e:
            last_err = e
            # strengthen instruction
            p = p + "\n\nREMINDER: Output ONLY valid JSON. No prose. No markdown. No code fences."
            time.sleep(0.8)
            if _is_quota_error(e):
                # don't retry endlessly if quota is dead
                raise
    raise RuntimeError(f"Failed to extract JSON after retries. Last error: {last_err}")


def fetch_shares_outstanding(client: OpenAI, company: str, ticker: str, ann_date_iso: str) -> SharesResult:
    """
    Two-stage strategy:
      Stage 1: Appendix 2A / Appendix 3H targeted
      Stage 2: broader ASX docs fallback
    """
    ticker_clean = (ticker or "").strip().upper()
    company_clean = (company or "").strip()

    tools = [{
        "type": "web_search",
        "filters": {"allowed_domains": ALLOWED_DOMAINS}
    }]
    include = ["web_search_call.action.sources"]

    # -------------------- STAGE 1 --------------------
    stage1_prompt = f"""
You are a financial data extraction assistant.

Goal:
Find the total number of ORDINARY shares on issue for:

Company: {company_clean}
Ticker: {ticker_clean}
Target date: {ann_date_iso}

PRIORITY:
Search specifically for ASX forms that include a table with the column:
"Total number of securities on issue", especially:

- Appendix 2A: "Application for quotation of securities"
  Often has section 4.1/4.2 "Quoted securities" with "Total number of securities on issue"
- Appendix 3H: "Notification of cessation of securities"
  Often has section 3.1/3.2 with "Total number of securities on issue"

Extract the row for "ORDINARY FULLY PAID" (or closest equivalent).
Date selection:
- Prefer the closest dated document to {ann_date_iso}
- If similarly close, prefer BEFORE the target date

Return ONLY valid JSON:

{{
  "shares_outstanding": <integer or null>,
  "shares_asof_date": "<YYYY-MM-DD or null>",
  "source_url": "<url or null>",
  "source_title": "<title or null>",
  "source_domain": "<domain or null>",
  "confidence": "<low|medium|high>",
  "method_notes": "<e.g. Appendix 2A section 4.1 dated YYYY-MM-DD; why chosen>"
}}

If no Appendix 2A/3H found with a clear dated shares-on-issue value, return shares_outstanding=null.
"""

    try:
        data1 = _run_responses_json(client, stage1_prompt, tools, include)
        shares1 = _coerce_int_shares(data1.get("shares_outstanding"))
        if shares1:
            url1 = data1.get("source_url")
            return SharesResult(
                shares_outstanding=shares1,
                shares_asof_date=_parse_date_to_iso(data1.get("shares_asof_date")) or data1.get("shares_asof_date"),
                source_url=url1,
                source_title=data1.get("source_title"),
                source_domain=data1.get("source_domain") or _domain_from_url(url1),
                confidence=(data1.get("confidence") or "medium").strip().lower(),
                method_notes=(data1.get("method_notes") or "") + " (Stage 1: Appendix 2A/3H)",
            )
    except Exception as e:
        if _is_quota_error(e):
            return SharesResult(
                confidence="low",
                method_notes=f"OpenAI quota/billing error: {e}",
            )
        # otherwise continue to stage 2

    # -------------------- STAGE 2 --------------------
    stage2_prompt = f"""
You are a financial data extraction assistant.

Task:
Find the number of ordinary shares outstanding (shares on issue / issued capital)
for the ASX-listed company below, ON the announcement date or closest available date.

Company: {company_clean}
Ticker: {ticker_clean}
Announcement date: {ann_date_iso}

Source priority:
- ASX announcements or lodged documents that state shares on issue / issued capital
  (Appendix 3B, 3C, 4A, 4E, annual report, results release, investor presentation lodged on ASX)

Rules:
- Prefer document date closest to {ann_date_iso}
- If similarly close, prefer BEFORE the date
- Extract ORDINARY FULLY PAID shares (or equivalent)

Return ONLY valid JSON:

{{
  "shares_outstanding": <integer or null>,
  "shares_asof_date": "<YYYY-MM-DD or null>",
  "source_url": "<url or null>",
  "source_title": "<title or null>",
  "source_domain": "<domain or null>",
  "confidence": "<low|medium|high>",
  "method_notes": "<why this document/date was chosen>"
}}
"""

    try:
        data2 = _run_responses_json(client, stage2_prompt, tools, include)
        shares2 = _coerce_int_shares(data2.get("shares_outstanding"))
        url2 = data2.get("source_url")
        return SharesResult(
            shares_outstanding=shares2,
            shares_asof_date=_parse_date_to_iso(data2.get("shares_asof_date")) or data2.get("shares_asof_date"),
            source_url=url2,
            source_title=data2.get("source_title"),
            source_domain=data2.get("source_domain") or _domain_from_url(url2),
            confidence=(data2.get("confidence") or "low").strip().lower(),
            method_notes=(data2.get("method_notes") or "") + " (Stage 2: fallback)",
        )
    except Exception as e:
        if _is_quota_error(e):
            return SharesResult(
                confidence="low",
                method_notes=f"OpenAI quota/billing error: {e}",
            )
        return SharesResult(
            confidence="low",
            method_notes=f"API failure: {e}",
        )


def process_one_row(client: OpenAI, idx: int, row: pd.Series) -> Tuple[int, SharesResult]:
    """
    Process a single row and return (index, result).
    Returns index so we can map results back to correct rows.
    """
    company = str(row.get("Company") or "")
    ticker = str(row.get("Ticker") or "")
    ann_iso = row.get("announcement_date_iso")

    if not ann_iso or not ticker.strip():
        return idx, SharesResult(
            confidence="low",
            method_notes="Skipped (missing ticker or announcement_date).",
            yfinance_data_available="Unknown"
        )

    try:
        # Check yfinance data availability first (fast, doesn't use API credits)
        yf_status = check_yfinance_data(ticker, ann_iso)

        # Then fetch shares outstanding (uses API)
        result = fetch_shares_outstanding(client, company, ticker, ann_iso)
        result.yfinance_data_available = yf_status

        return idx, result
    except Exception as e:
        return idx, SharesResult(
            confidence="low",
            method_notes=f"Error: {str(e)[:200]}",
            yfinance_data_available="Unknown"
        )


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set.\n"
            "PowerShell: setx OPENAI_API_KEY \"YOUR_KEY\"\n"
            "Then open a NEW terminal and run again."
        )

    client = OpenAI()

    print("=== ASX Shares Outstanding Extraction (Parallel) ===")
    print(f"Model: {MODEL}")
    print(f"Max workers: {MAX_WORKERS}")
    print(f"Input: {INPUT_XLSX}")
    print(f"Output: {OUTPUT_XLSX}")
    print()

    df = pd.read_excel(INPUT_XLSX)
    df = _normalize_columns(df)
    df["announcement_date_iso"] = df["announcement_date"].apply(_parse_date_to_iso)

    out_cols = [
        "shares_outstanding",
        "shares_asof_date",
        "source_url",
        "source_title",
        "source_domain",
        "confidence",
        "method_notes",
        "yfinance_data_available",
    ]
    for c in out_cols:
        if c not in df.columns:
            df[c] = None

    # Store results indexed by row number
    results: Dict[int, SharesResult] = {}

    # Process in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_idx = {}
        for idx, row in df.iterrows():
            ticker = str(row.get("Ticker") or "").strip().upper()
            ann_iso = row.get("announcement_date_iso")

            future = executor.submit(process_one_row, client, idx, row)
            future_to_idx[future] = (idx, ticker, ann_iso)

        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx, ticker, ann_iso = future_to_idx[future]
            row_idx, result = future.result()
            results[row_idx] = result

            status = "✓" if result.shares_outstanding else "✗"
            yf_indicator = f"YF:{result.yfinance_data_available}"
            print(f"{status} [{idx+1}/{len(df)}] {ticker} @ {ann_iso} - {result.confidence} ({yf_indicator})")

    # Write results back to dataframe in original order
    for idx in results:
        result = results[idx]
        df.at[idx, "shares_outstanding"] = result.shares_outstanding
        df.at[idx, "shares_asof_date"] = result.shares_asof_date
        df.at[idx, "source_url"] = result.source_url
        df.at[idx, "source_title"] = result.source_title
        df.at[idx, "source_domain"] = result.source_domain
        df.at[idx, "confidence"] = result.confidence
        df.at[idx, "method_notes"] = result.method_notes
        df.at[idx, "yfinance_data_available"] = result.yfinance_data_available

    df.drop(columns=["announcement_date_iso"], inplace=True, errors="ignore")
    df.to_excel(OUTPUT_XLSX, index=False)

    # Summary
    successful = sum(1 for r in results.values() if r.shares_outstanding is not None)
    yf_yes = sum(1 for r in results.values() if r.yfinance_data_available == "Yes")
    yf_no = sum(1 for r in results.values() if r.yfinance_data_available == "No")

    print(f"\n=== Summary ===")
    print(f"Shares Outstanding Found: {successful}/{len(df)}")
    print(f"Failed: {len(df) - successful}/{len(df)}")
    print(f"\nYahoo Finance Data:")
    print(f"  Available (±30 days): {yf_yes}/{len(df)}")
    print(f"  Not Available: {yf_no}/{len(df)}")
    print(f"\nDone. Wrote: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()