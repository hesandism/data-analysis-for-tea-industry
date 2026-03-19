"""
Tea Auction Weather Data Pipeline
==================================
Extracts weather data from Forbes & Walker weekly PDF reports AND
fetches actual historical meteorological data from Open-Meteo API
for all four Sri Lanka tea-growing regions.

Output: weather_features.csv — one row per (sale_id, region)
        ready to JOIN with your price CSVs on sale_id.

Usage:
    pip install pdfplumber requests pandas
    python weather_pipeline.py --pdf_dir ./pdfs --output weather_features.csv
"""

import re
import json
import time
import argparse
import requests
import pdfplumber
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


# ── Region coordinates (centroid of each tea-growing elevation) ──────────────
REGIONS = {
    "western_high": {
        "lat": 6.9271,
        "lon": 80.5350,
        "label": "Western High Grown (Maskeliya / Dickoya)",
    },
    "nuwara_eliya": {
        "lat": 6.9497,
        "lon": 80.7891,
        "label": "Nuwara Eliya",
    },
    "uva_udapussellawa": {
        "lat": 6.8700,
        "lon": 81.0600,
        "label": "Uva / Uda Pussellawa",
    },
    "low_grown": {
        "lat": 6.2500,
        "lon": 80.3000,
        "label": "Low Grown (Matara / Galle belt)",
    },
}

# Open-Meteo free historical endpoint — no API key needed
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# Weather variables to fetch — all daily
METEO_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "windspeed_10m_max",
    "sunshine_duration",           # seconds of bright sunshine per day
    "et0_fao_evapotranspiration",  # useful proxy for plant water stress
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
]

# ── Sale date lookup ─────────────────────────────────────────────────────────
# Map sale IDs to their auction date range so we can fetch the correct week.
# Extend this dict as you add more sales.
SALE_DATES = {
    "SALE_01_2026": ("2026-01-06", "2026-01-07"),
    "SALE_02_2026": ("2026-01-12", "2026-01-13"),
    "SALE_03_2026": ("2026-01-20", "2026-01-21"),
    "SALE_04_2026": ("2026-01-27", "2026-01-28"),
    "SALE_05_2026": ("2026-02-03", "2026-02-04"),
    "SALE_06_2026": ("2026-02-10", "2026-02-11"),
    "SALE_07_2026": ("2026-02-17", "2026-02-18"),
    "SALE_08_2026": ("2026-02-24", "2026-02-25"),
    "SALE_09_2026": ("2026-03-03", "2026-03-04"),
    "SALE_10_2026": ("2026-03-10", "2026-03-11"),
}

# For each sale we fetch the 7-day window BEFORE the auction date
# (that's the crop week that influenced the teas on offer).
LOOKBACK_DAYS = 7


# ── PDF text extraction ──────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: Path) -> str:
    """Return all text from a PDF as a single string."""
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text.append(t)
    return "\n".join(text)


def extract_page1_text(pdf_path: Path) -> str:
    """Return text from page 1 only (cover page)."""
    with pdfplumber.open(pdf_path) as pdf:
        t = pdf.pages[0].extract_text()
        return t or ""


def parse_sale_id_from_cover(pdf_path: Path) -> str | None:
    """
    Parse sale_id from the cover page of a Forbes & Walker PDF.

    The cover page text always follows this exact layout (confirmed from PDFs):

        FORBES & WALKER TEA BROKERS PVT LTD
        WEEKLY TEA
        MARKET REPORT
        SALE NO
        06                   ← sale number on its own line
        10TH/11TH
        FEBRUARY 2026        ← month + year on its own line

    Returns e.g. "SALE_06_2026".
    """
    cover = extract_page1_text(pdf_path)

    # Match "SALE NO" followed by the sale number on the very next non-empty line
    m = re.search(
        r"SALE\s+NO\s*\n\s*(\d{1,3})\s*\n",
        cover,
        re.IGNORECASE,
    )
    if not m:
        # Fallback: "SALE NO : 6" or "SALE NO: 06" on the same line
        m = re.search(r"SALE\s+NO\s*[:\-]?\s*(\d{1,3})", cover, re.IGNORECASE)

    if not m:
        return None

    sale_num = int(m.group(1))

    # Year: last 4-digit number that looks like a year (20xx)
    years = re.findall(r"\b(20\d{2})\b", cover)
    if not years:
        return None
    year = years[-1]   # take the last occurrence — avoids matching body text

    return f"SALE_{sale_num:02d}_{year}"


# ── Weather text parsing ─────────────────────────────────────────────────────

CONDITION_KEYWORDS = {
    "bright":      1,
    "sunny":       1,
    "clear":       1,
    "dry":         1,
    "showers":    -1,
    "rain":       -1,
    "rainfall":   -1,
    "misty":      -1,
    "mist":       -1,
    "dull":       -1,
    "cloud":      -1,
    "thunder":    -2,
    "heavy":      -2,
    "flood":      -3,
}


def parse_region_weather_text(text: str) -> dict:
    """
    Parse the 'CROP AND WEATHER' section of a report.
    Returns a dict with one sub-dict per region containing:
        - raw_text : the original paragraph
        - condition_score : sentiment-like integer (-3 → +1)
        - keywords_found : list of matched condition words
        - crop_change : "increase" | "decrease" | "maintained" | "unknown"
    """
    results = {}

    # Extract just the crop & weather section
    section_match = re.search(
        r"CROP AND WEATHER(.+?)(?=FORBES & WALKER WEEKLY|HIGH GROWN TEAS|$)",
        text, re.DOTALL | re.IGNORECASE
    )
    if not section_match:
        return results

    section = section_match.group(1)

    # Region anchors — map to our canonical keys
    region_patterns = {
        "western_high": r"(Western[/ ]Nuwara Eliya[^\n]*\n.{10,600}?)(?=Uva|Udapu|Low Grown|Crop\n|$)",
        "nuwara_eliya": r"(Nuwara Eliya[^\n]*\n.{10,400}?)(?=Uva|Udapu|Low Grown|Crop\n|$)",
        "uva_udapussellawa": r"(Uva[/ ]Udapu[^\n]*\n.{10,600}?)(?=Low Grown|Western|Crop\n|$)",
        "low_grown": r"(Low Grown[^\n]*\n.{10,600}?)(?=Crop\n|Western|Uva|$)",
    }

    for region_key, pattern in region_patterns.items():
        m = re.search(pattern, section, re.DOTALL | re.IGNORECASE)
        raw = m.group(1).strip() if m else ""

        # Score weather conditions
        score = 0
        found = []
        lower = raw.lower()
        for kw, weight in CONDITION_KEYWORDS.items():
            if kw in lower:
                score += weight
                found.append(kw)

        # Detect crop change for the summary section
        crop_change = "unknown"
        crop_m = re.search(
            r"(increase|decrease|maintained|decline|improved|similar to last)",
            lower
        )
        if crop_m:
            word = crop_m.group(1)
            if word in ("increase", "improved"):
                crop_change = "increase"
            elif word in ("decrease", "decline"):
                crop_change = "decrease"
            elif word == "maintained":
                crop_change = "maintained"
            elif "similar" in word:
                crop_change = "similar"

        results[region_key] = {
            "raw_text": raw[:500],   # cap to avoid huge CSV cells
            "condition_score": score,
            "keywords_found": "|".join(found),
            "crop_change": crop_change,
            # Binary flags for modelling convenience
            "has_rain": int("rain" in lower or "shower" in lower),
            "has_mist": int("mist" in lower),
            "has_bright": int("bright" in lower or "sunny" in lower),
            "has_thunder": int("thunder" in lower),
        }

    return results


# ── Open-Meteo API fetcher ───────────────────────────────────────────────────

def fetch_weekly_weather(lat: float, lon: float,
                          auction_date_str: str) -> dict:
    """
    Fetch 7-day averages for the week BEFORE the auction date.
    Returns a flat dict of aggregated meteorological variables.
    """
    auction_dt = datetime.strptime(auction_date_str, "%Y-%m-%d")
    end_dt = auction_dt - timedelta(days=1)
    start_dt = end_dt - timedelta(days=LOOKBACK_DAYS - 1)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        "daily": ",".join(METEO_VARIABLES),
        "timezone": "Asia/Colombo",
    }

    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"    [WARN] API error for ({lat},{lon}): {e}")
        return {}

    daily = data.get("daily", {})
    result = {}

    for var in METEO_VARIABLES:
        values = [v for v in daily.get(var, []) if v is not None]
        if not values:
            result[f"{var}_mean"] = None
            result[f"{var}_max"] = None
            result[f"{var}_min"] = None
            continue

        if "precipitation" in var or "rain" in var or "sunshine" in var or "et0" in var:
            # Accumulate totals for flow/energy variables
            result[f"{var}_total"] = round(sum(values), 2)
            result[f"{var}_max_day"] = round(max(values), 2)
        else:
            result[f"{var}_mean"] = round(sum(values) / len(values), 2)
            result[f"{var}_max"] = round(max(values), 2)
            result[f"{var}_min"] = round(min(values), 2)

    result["fetch_start"] = start_dt.strftime("%Y-%m-%d")
    result["fetch_end"] = end_dt.strftime("%Y-%m-%d")
    return result


# ── Lag feature generator ────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame,
                      lag_cols: list[str],
                      lags: list[int] = [1, 2, 3]) -> pd.DataFrame:
    """
    For each (region, variable) add lag-N columns aligned by sale order.
    This is the core of your 'lag deviation of time' idea.
    """
    df = df.sort_values(["region", "sale_id"]).copy()

    for region, grp in df.groupby("region"):
        idx = grp.index
        for col in lag_cols:
            if col not in df.columns:
                continue
            for lag in lags:
                lag_col = f"{col}_lag{lag}"
                df.loc[idx, lag_col] = grp[col].shift(lag).values

    return df


# ── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(pdf_dir: str, output_path: str):
    pdf_dir = Path(pdf_dir)
    rows = []

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {pdf_dir}")
        return

    print(f"Found {len(pdfs)} PDF(s) in {pdf_dir}\n")

    for pdf_path in pdfs:
        print(f"Processing: {pdf_path.name}")

        # Sale ID is read directly from the cover page number, e.g. 'SALE NO\n06'
        sale_id = parse_sale_id_from_cover(pdf_path)
        if not sale_id:
            print(f"  [SKIP] Could not parse sale_id from cover of {pdf_path.name}")
            continue
        print(f"  Sale ID: {sale_id}")

        text = extract_pdf_text(pdf_path)

        # Get auction date
        if sale_id not in SALE_DATES:
            print(f"  [WARN] No date mapping for {sale_id} — add to SALE_DATES dict")
            continue

        auction_date = SALE_DATES[sale_id][0]  # use first day

        # Parse text weather summaries
        text_weather = parse_region_weather_text(text)

        # For each region, combine text parse + API fetch
        for region_key, region_info in REGIONS.items():
            print(f"    Region: {region_key}")

            row = {
                "sale_id": sale_id,
                "auction_date": auction_date,
                "region": region_key,
                "region_label": region_info["label"],
                "lat": region_info["lat"],
                "lon": region_info["lon"],
                "pdf_source": pdf_path.name,
            }

            # Add text-parsed fields
            tw = text_weather.get(region_key, {})
            row.update({
                "text_condition_score": tw.get("condition_score"),
                "text_crop_change": tw.get("crop_change", "unknown"),
                "text_has_rain": tw.get("has_rain"),
                "text_has_mist": tw.get("has_mist"),
                "text_has_bright": tw.get("has_bright"),
                "text_has_thunder": tw.get("has_thunder"),
                "text_keywords": tw.get("keywords_found", ""),
                "text_raw_summary": tw.get("raw_text", ""),
            })

            # Add API meteorological data
            api_data = fetch_weekly_weather(
                region_info["lat"], region_info["lon"], auction_date
            )
            row.update(api_data)

            rows.append(row)
            time.sleep(0.3)   # be polite to the free API

    if not rows:
        print("\nNo rows collected — check PDF parsing.")
        return

    df = pd.DataFrame(rows)

    # Generate lag features for key meteorological columns
    lag_targets = [
        "precipitation_sum_total",
        "rain_sum_total",
        "temperature_2m_mean_mean",
        "sunshine_duration_total",
        "relative_humidity_2m_max_max",
        "text_condition_score",
    ]
    df = add_lag_features(df, lag_targets, lags=[1, 2, 3])

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n✓ Saved {len(df)} rows to {out}")
    print(f"  Columns: {list(df.columns)}\n")

    return df


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    _root = Path(__file__).parent.parent.parent

    parser = argparse.ArgumentParser(
        description="Extract weather data from F&W tea auction PDFs + Open-Meteo API"
    )
    parser.add_argument(
        "--pdf_dir", default=str(_root / "data" / "raw"),
        help="Directory containing the weekly PDF reports"
    )
    parser.add_argument(
        "--output", default=str(_root / "data" / "interim" / "09_weather_features.csv"),
        help="Output CSV path"
    )
    args = parser.parse_args()
    run_pipeline(args.pdf_dir, args.output)
