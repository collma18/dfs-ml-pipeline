"""
DFS PDF -> OpenAI extraction -> Excel (ASX / AUD) with FX injection + PARALLEL batch processing

What this version does:
- Same two-pass workflow per PDF:
    PASS 1: extract announcement_date only
    PASS 2: fetch FX rates for that date + inject into prompt (no model web search)
- Runs PDFs in parallel using ThreadPoolExecutor
- Adds retry + backoff for transient API/HTTP errors
- Writes Excel; if locked, writes a timestamped filename instead

Install:
  pip install -U openai pandas openpyxl requests

Env:
  OPENAI_API_KEY=sk-...
  (optional) MODEL=gpt-5.2
"""

import base64
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from openai import OpenAI

# =========================
# CONFIG
# =========================
DFS_FOLDER = Path(r"C:\Users\User\Desktop\MoreDFS\downloaded_pdfs")
OUTPUT_FILE = Path(r"C:\Users\User\Desktop\MoreDFS\downloaded_pdfs\dfs_features.xlsx")

MODEL = os.environ.get("MODEL", "gpt-5.2")  # Main extraction model
DATE_MODEL = os.environ.get("DATE_MODEL", "gpt-5-mini")  # Cheap model for date extraction

# Parallelism: tune this to your rate limits (start with 3–6)
MAX_WORKERS = 4

# FX lookup config
FX_TIMEOUT_SECONDS = 20
CURRENCIES_TO_PROVIDE = ["USD", "CAD", "EUR", "GBP", "NZD"]

# Retries for transient errors (429 / 5xx / network hiccups)
MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 1.5

client = OpenAI()

# =========================
# EXTRACTION PROMPT
# =========================
EXTRACTION_PROMPT = """
You are extracting key metrics from an ASX-listed mining company's Definitive Feasibility Study (DFS).

TASK
Return ONE JSON object that matches the provided JSON schema exactly.
Do not include commentary, explanations, or extra keys.

GENERAL RULES
- Use the DFS document only. Do NOT guess.
- Prefer BASE CASE / Reference Case / Most Likely Case.
- Ignore upside, expansion, and sensitivity cases.
- Return BOTH post-tax and pre-tax values if available.
- ALL currency figures must be converted to AUD using the provided exchange rates below.

SOURCE PRIORITY / TIE-BREAKERS
- Prefer values shown in the "Executive Summary", "Project Summary", "Key Metrics", "Investment Highlights", or "Financial Summary" table.
- If the same metric appears in multiple places, use the value from the main summary table.
- Ignore appendices, sensitivity tables, expansion cases, and upside scenarios.
- If multiple discount rates are presented, use the base-case discount rate used in the summary table.
- If both pre-tax and post-tax values are shown in the same table, extract both from that table.
- If pre-tax and post-tax values appear in different places, prefer the summary table values.

CURRENCY CONVERSION
- All currency figures must be in AUD as absolute numbers (no strings like "A$m").
- If reported as A$m, AUDm, A$M → multiply by 1,000,000
- If reported as A$bn → multiply by 1,000,000,000
- If values are reported in a foreign currency (USD, CAD, etc.):
  1. Identify the currency in the document
  2. Use the FX rates provided below (do NOT search for rates)
  3. Convert ALL currency values to AUD using those rates
  4. Apply this conversion to all fields: NPV, capex, AISC, commodity price, and any other currency values
- Example: If NPV is USD $500M and the rate is 1 USD = 1.55 AUD, then NPV = 500 × 1.55 = 775M AUD

COMPANY INFORMATION
- company_name: The full legal name of the mining company (e.g., "Pilbara Minerals Limited").
- Usually found on the cover page, header, or ASX announcement header.
- stock_ticker: The 3-letter ASX ticker code (e.g., "PLS").
- Usually found near the company name or in the ASX announcement header.
- If ticker cannot be found, return null.

ANNOUNCEMENT DATE
- announcement_date is the ASX announcement / lodgement date of the DFS.
- Usually found in the ASX release header, cover page, or first page footer.
- Ignore report preparation or effective dates.
- Format strictly as YYYY-MM-DD.
- If multiple dates exist, choose the ASX announcement date.

PRIMARY METAL
- primary_metal is the main revenue / headline commodity (e.g., Gold, Copper, Lithium).
- Use a simple single-word or common industry name.

FIELD GUIDANCE
- npv_post_tax_aud: Post-tax NPV in AUD (base case). Only populate if explicitly labeled as "post-tax" or "after-tax".
- irr_post_tax_pct: Post-tax IRR in percent (e.g., 27.5). Only populate if explicitly labeled as "post-tax" or "after-tax".
- npv_pre_tax_aud: Pre-tax NPV in AUD (base case). If only one NPV value is given without tax specification, assume it is pre-tax and populate this field.
- irr_pre_tax_pct: Pre-tax IRR in percent (e.g., 27.5). If only one IRR value is given without tax specification, assume it is pre-tax and populate this field.
- payback_years: Payback period in years. If given in months, divide by 12.
- initial_capex_aud: Initial / pre-production capex in AUD.
- aisc_aud_per_unit: All-In Sustaining Cost (AISC). If AISC is not available, use any operating cost per unit (e.g., C1 cost, operating cost, cash cost).
- aisc_unit: One of ["AUD/oz","AUD/t","AUD/lb","AUD/kg","AUD/MWh","AUD/unit","other"]. Should match the unit used for the cost figure extracted.
- annual_production: Steady-state annual production for the primary metal (life of mine average).
- annual_production_unit: Unit for annual production (e.g., "oz/y", "t/y", "lb/y").
- mine_life_years: Mine life in years.
- base_case_commodity_price_aud_per_unit: Commodity price assumption in AUD per unit. Priority order: (1) Long-term sale price, (2) Base-case price, (3) Any referenced sale price. If a price range is given (e.g., $1,800-$2,000), use the midpoint.
- commodity_price_unit: Unit for commodity price (e.g., "AUD/oz", "AUD/t").
- permitting_status:
    0 = Not permitted
    1 = In progress / partially permitted
    2 = Fully permitted / approved

MISSING DATA
- If a value cannot be found, return null.
""".strip()

# =========================
# STRUCTURED OUTPUTS SCHEMA
# =========================
DFS_SCHEMA_ONLY: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "document_name": {"type": "string"},
        "company_name": {"type": ["string", "null"]},
        "stock_ticker": {"type": ["string", "null"]},
        "announcement_date": {"type": ["string", "null"]},
        "primary_metal": {"type": ["string", "null"]},
        "npv_post_tax_aud": {"type": ["number", "null"]},
        "irr_post_tax_pct": {"type": ["number", "null"]},
        "npv_pre_tax_aud": {"type": ["number", "null"]},
        "irr_pre_tax_pct": {"type": ["number", "null"]},
        "payback_years": {"type": ["number", "null"]},
        "initial_capex_aud": {"type": ["number", "null"]},
        "aisc_aud_per_unit": {"type": ["number", "null"]},
        "aisc_unit": {
            "type": ["string", "null"],
            "enum": ["AUD/oz", "AUD/t", "AUD/lb", "AUD/kg", "AUD/MWh", "AUD/unit", "other", None],
        },
        "annual_production": {"type": ["number", "null"]},
        "annual_production_unit": {"type": ["string", "null"]},
        "mine_life_years": {"type": ["number", "null"]},
        "base_case_commodity_price_aud_per_unit": {"type": ["number", "null"]},
        "commodity_price_unit": {"type": ["string", "null"]},
        "permitting_status": {"type": ["integer", "null"], "enum": [0, 1, 2, None]},
    },
    "required": [
        "document_name",
        "company_name",
        "stock_ticker",
        "announcement_date",
        "primary_metal",
        "npv_post_tax_aud",
        "irr_post_tax_pct",
        "npv_pre_tax_aud",
        "irr_pre_tax_pct",
        "payback_years",
        "initial_capex_aud",
        "aisc_aud_per_unit",
        "aisc_unit",
        "annual_production",
        "annual_production_unit",
        "mine_life_years",
        "base_case_commodity_price_aud_per_unit",
        "commodity_price_unit",
        "permitting_status",
    ],
}

TEXT_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "name": "dfs_key_metrics",
    "strict": True,
    "schema": DFS_SCHEMA_ONLY,
}

DATE_ONLY_TEXT_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "name": "dfs_date_only",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {"announcement_date": {"type": ["string", "null"]}},
        "required": ["announcement_date"],
    },
}


# =========================
# HELPERS
# =========================
def pdf_to_data_url(pdf_path: Path) -> str:
    pdf_bytes = pdf_path.read_bytes()
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    return f"data:application/pdf;base64,{b64}"


def safe_write_excel(df: pd.DataFrame, output_path: Path) -> Path:
    try:
        df.to_excel(output_path, index=False)
        return output_path
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = output_path.with_name(f"{output_path.stem}_{ts}{output_path.suffix}")
        df.to_excel(alt, index=False)
        return alt


def normalize_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    if "announcement_date" in df.columns:
        parsed = pd.to_datetime(df["announcement_date"], errors="coerce", format="%Y-%m-%d")
        df.loc[parsed.notna(), "announcement_date"] = parsed.dt.date.astype(str)
    return df


def backoff_sleep(attempt: int) -> None:
    # Exponential backoff + jitter
    sleep_s = BASE_BACKOFF_SECONDS * (2 ** attempt) * (0.7 + random.random() * 0.6)
    time.sleep(min(sleep_s, 30))


def call_with_retries(fn, *args, **kwargs):
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            # Retry on common transient issues: 429s, 5xx, network timeouts, etc.
            # If it's a hard 400, no point retrying.
            msg = str(e).lower()
            hard_400 = "error code: 400" in msg or "invalid_request_error" in msg
            if hard_400:
                raise
            if attempt < MAX_RETRIES - 1:
                print(f"  Retry {attempt + 1}/{MAX_RETRIES - 1} after error: {str(e)[:100]}")
                backoff_sleep(attempt)
            else:
                raise last_err


def get_aud_per_1_foreign(date_yyyy_mm_dd: str, foreign: str) -> Optional[float]:
    """
    Returns AUD per 1 unit of `foreign` on date_yyyy_mm_dd.
    Uses Frankfurter:
      /YYYY-MM-DD?from=AUD&to=USD returns USD per 1 AUD.
    So AUD per 1 USD = 1 / (USD per 1 AUD)
    """
    foreign = foreign.upper().strip()
    if foreign == "AUD":
        return 1.0

    url = f"https://api.frankfurter.app/{date_yyyy_mm_dd}?from=AUD&to={foreign}"
    try:
        r = requests.get(url, timeout=FX_TIMEOUT_SECONDS)
        r.raise_for_status()
        data = r.json()
        rate_foreign_per_aud = float(data["rates"][foreign])
        if rate_foreign_per_aud == 0:
            return None
        return 1.0 / rate_foreign_per_aud
    except Exception:
        return None


def extract_announcement_date_only(pdf_path: Path) -> Optional[str]:
    prompt = (
        "Extract the ASX announcement / lodgement date from the document header/cover. "
        "Return strictly YYYY-MM-DD format or null if not found. "
        "Look for dates in the ASX release header, cover page, or first page footer. "
        "Ignore report preparation or effective dates."
    )

    def _call():
        return client.responses.create(
            model=DATE_MODEL,  # Use cheaper model for simple date extraction
            reasoning={"effort": "low"},  # Low effort is fine for simple date extraction
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_file", "filename": pdf_path.name, "file_data": pdf_to_data_url(pdf_path)},
                    {"type": "input_text", "text": prompt},
                ],
            }],
            text={"format": DATE_ONLY_TEXT_FORMAT},
        )

    resp = call_with_retries(_call)
    out_text = resp.output_text
    if not out_text:
        return None
    try:
        return json.loads(out_text).get("announcement_date")
    except Exception:
        return None


def build_fx_hint(announcement_date: Optional[str]) -> str:
    if not announcement_date:
        return (
            "\n\nFX RATES NOT PROVIDED:\n"
            "- No announcement date could be extracted.\n"
            "- If foreign-currency values exist in the document, return null for currency fields that cannot be converted.\n"
            "- Do NOT attempt to search for exchange rates.\n"
        )

    lines: List[str] = [
        "\n\nFX RATES PROVIDED (use these rates, do NOT search the web):",
        f"- Announcement date: {announcement_date}",
        "- Exchange rates for conversion to AUD:",
    ]

    any_rate = False
    for ccy in CURRENCIES_TO_PROVIDE:
        aud_per_1 = get_aud_per_1_foreign(announcement_date, ccy)
        if aud_per_1 is not None:
            any_rate = True
            lines.append(f"  - 1 {ccy} = {aud_per_1:.6f} AUD")
        else:
            lines.append(f"  - 1 {ccy} = (rate unavailable)")

    if not any_rate:
        lines.append("\n- WARNING: No FX rates could be retrieved.")
        lines.append("- If foreign currency amounts exist, return null for fields that cannot be converted.")

    lines.append("\n- Multiply all foreign currency values by the appropriate rate to convert to AUD.")
    lines.append("- Example: USD $100M × 1.55 AUD/USD = AUD $155M")

    return "\n".join(lines) + "\n"


def extract_dfs(pdf_path: Path) -> Dict[str, Any]:
    # Pass 1: Get announcement date
    print(f"  Pass 1: Extracting date from {pdf_path.name}...")
    announcement_date = extract_announcement_date_only(pdf_path)

    if announcement_date:
        print(f"  Found date: {announcement_date}")
    else:
        print(f"  No date found, proceeding without FX rates")

    # Pass 2: Get all fields with FX rates injected
    fx_hint = build_fx_hint(announcement_date)
    print(f"  Pass 2: Extracting all fields...")

    def _call():
        return client.responses.create(
            model=MODEL,
            reasoning={"effort": "medium"},
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_file", "filename": pdf_path.name, "file_data": pdf_to_data_url(pdf_path)},
                    {"type": "input_text", "text": EXTRACTION_PROMPT + fx_hint},
                ],
            }],
            text={"format": TEXT_FORMAT},
        )

    resp = call_with_retries(_call)
    return json.loads(resp.output_text)


def make_error_row(pdf_name: str, err: Exception) -> Dict[str, Any]:
    return {
        "document_name": pdf_name,
        "company_name": None,
        "stock_ticker": None,
        "announcement_date": None,
        "primary_metal": None,
        "npv_post_tax_aud": None,
        "irr_post_tax_pct": None,
        "npv_pre_tax_aud": None,
        "irr_pre_tax_pct": None,
        "payback_years": None,
        "initial_capex_aud": None,
        "aisc_aud_per_unit": None,
        "aisc_unit": None,
        "annual_production": None,
        "annual_production_unit": None,
        "mine_life_years": None,
        "base_case_commodity_price_aud_per_unit": None,
        "commodity_price_unit": None,
        "permitting_status": None,
        "error": str(err)[:200],  # Store error message for debugging
    }


def process_one_pdf(pdf: Path) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (pdf_name, row_dict). Always returns a row dict (error row on failure).
    """
    print(f"\nProcessing: {pdf.name}")
    try:
        data = extract_dfs(pdf)
        data["document_name"] = data.get("document_name") or pdf.name
        print(f"  ✓ Success: {data.get('company_name', 'Unknown')} - {data.get('primary_metal', 'N/A')}")
        return pdf.name, data
    except Exception as e:
        print(f"  ✗ Failed: {str(e)[:100]}")
        return pdf.name, make_error_row(pdf.name, e)


# =========================
# MAIN (PARALLEL)
# =========================
def main() -> None:
    if not DFS_FOLDER.exists():
        raise FileNotFoundError(f"Folder not found: {DFS_FOLDER.resolve()}")

    pdfs = sorted(DFS_FOLDER.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in: {DFS_FOLDER.resolve()}")

    print(f"=== DFS Extraction Pipeline ===")
    print(f"Date extraction model: {DATE_MODEL} (cheaper)")
    print(f"Main extraction model: {MODEL}")
    print(f"Parallel workers: {MAX_WORKERS}")
    print(f"Total PDFs: {len(pdfs)}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    results: Dict[str, Dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(process_one_pdf, pdf): pdf for pdf in pdfs}

        for fut in as_completed(future_map):
            pdf = future_map[fut]
            name, row = fut.result()
            results[name] = row

    # Preserve original file order
    rows: List[Dict[str, Any]] = [results[pdf.name] for pdf in pdfs]

    # Remove error column if it exists (just for debugging)
    for row in rows:
        row.pop("error", None)

    df = pd.DataFrame(rows)
    df = normalize_for_excel(df)

    saved_path = safe_write_excel(df, OUTPUT_FILE)

    # Summary
    successful = sum(1 for r in rows if r.get("company_name") is not None)
    print(f"\n=== Summary ===")
    print(f"Successful: {successful}/{len(pdfs)}")
    print(f"Failed: {len(pdfs) - successful}/{len(pdfs)}")
    print(f"\nSaved to: {saved_path.resolve()}")


if __name__ == "__main__":
    main()