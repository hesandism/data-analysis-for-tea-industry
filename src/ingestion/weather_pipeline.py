# flake8: noqa
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
# Auto-generated from PDFs in data/Raw folder. Updates dynamically.
SALE_DATES = {}  # Will be populated by build_sale_dates()

# For each sale we fetch the 7-day window BEFORE the auction date
# (that's the crop week that influenced the teas on offer).
LOOKBACK_DAYS = 7


# ── Auto-generate SALE_DATES from PDF folder ─────────────────────────────────

def extract_dates_from_filename(filename: str) -> tuple[str, str] | None:
    """
    Extract auction dates from PDF filename pattern:
    'Sale of DD & DD Month YYYY.pdf' → ('YYYY-MM-DD', 'YYYY-MM-DD')
    """
    name = filename.strip()

    # Match "Sale of DD & DD Month YYYY" (allow extra spaces)
    m = re.search(
        r"Sale\s+of\s+(\d{1,2})\s*&\s*(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})",
        name,
        re.IGNORECASE,
    )

    # Fallback for one-day filenames, e.g. "Sale of 30 December 2025.pdf"
    if not m:
        m = re.search(
            r"Sale\s+of\s+(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})",
            name,
            re.IGNORECASE,
        )
        if not m:
            return None
        day1, month_str, year = m.groups()
        day2 = day1
    else:
        day1, day2, month_str, year = m.groups()
    
    month_map = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
    }
    month = month_map.get(month_str.lower())
    if not month:
        return None
    
    date1 = f"{year}-{month}-{int(day1):02d}"
    date2 = f"{year}-{month}-{int(day2):02d}"
    return (date1, date2)


def build_sale_dates(data_raw_dir: Path) -> dict[str, tuple[str, str]]:
    """
    Scan data/Raw folder and auto-generate SALE_DATES dictionary.
    Reads sale number from PDF cover page, dates from filename.
    
    Returns: {"SALE_NN_YYYY": ("YYYY-MM-DD", "YYYY-MM-DD"), ...}
    """
    sale_dates = {}
    pdfs = sorted(data_raw_dir.glob("*.pdf"))
    
    for pdf_path in pdfs:
        # Extract dates from filename
        dates = extract_dates_from_filename(pdf_path.name)
        if not dates:
            continue
        
        # Extract sale number from PDF cover page
        sale_id = parse_sale_id_from_cover(pdf_path)
        if sale_id:
            sale_dates[sale_id] = dates
    
    return dict(sorted(sale_dates.items()))


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

    return f"SALE_{year}_{sale_num:02d}"


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
    For each row, build lag-N features using true calendar-week offsets.

    lag1 = value at (auction_date - 7 days)
    lag2 = value at (auction_date - 14 days)
    lag3 = value at (auction_date - 21 days)

    Weather columns are fetched directly from Open-Meteo API using each row's
    lat/lon and lagged auction_date. Non-API columns (for example text features)
    are derived from in-data historical values by region.
    """
    df = df.copy()
    if "auction_date" not in df.columns:
        return df

    df["auction_date"] = pd.to_datetime(df["auction_date"], errors="coerce")
    df = df.sort_values(["region", "auction_date", "sale_id"]).copy()

    # Detect API-derived weather columns from the known Open-Meteo variable prefixes.
    api_cols = {
        col for col in lag_cols
        if any(col.startswith(f"{var}_") for var in METEO_VARIABLES)
    }
    non_api_cols = [col for col in lag_cols if col not in api_cols]

    # Build historical lookups for non-API lag columns by region and auction_date.
    non_api_lookup = {}
    if non_api_cols:
        for region, grp in df.groupby("region", sort=False):
            region_series = {
                col: grp.set_index("auction_date")[col].dropna().sort_index()
                for col in non_api_cols
                if col in grp.columns
            }
            non_api_lookup[region] = region_series

    # Cache Open-Meteo responses to avoid duplicate API calls.
    api_cache = {}

    def get_api_weather(lat: float, lon: float, lag_auction_dt: pd.Timestamp) -> dict:
        if pd.isna(lat) or pd.isna(lon) or pd.isna(lag_auction_dt):
            return {}
        date_str = lag_auction_dt.strftime("%Y-%m-%d")
        key = (round(float(lat), 4), round(float(lon), 4), date_str)
        if key not in api_cache:
            api_cache[key] = fetch_weekly_weather(float(lat), float(lon), date_str)
        return api_cache[key]

    for idx, row in df.iterrows():
        sale_id = row.get("sale_id")
        region = row.get("region")
        auction_dt = row.get("auction_date")
        lat = row.get("lat")
        lon = row.get("lon")

        if pd.isna(auction_dt):
            continue

        for lag in lags:
            lag_dt = auction_dt - pd.to_timedelta(7 * lag, unit="D")
            auction_date_str = auction_dt.strftime("%Y-%m-%d")
            lag_date_str = lag_dt.strftime("%Y-%m-%d")

            # API-driven lag values (directly fetched from Open-Meteo).
            api_data = get_api_weather(lat, lon, lag_dt) if api_cols else {}
            for col in api_cols:
                value = api_data.get(col)
                df.at[idx, f"{col}_lag{lag}"] = value
                print(
                    f"[LAG][API] sale_id={sale_id} region={region} "
                    f"col={col} lag{lag} auction_date={auction_date_str} "
                    f"lag_date={lag_date_str} value={value}"
                )

            # Non-API lag values from historical in-data series by region.
            region_series = non_api_lookup.get(region, {})
            for col in non_api_cols:
                series = region_series.get(col)
                if series is None or series.empty:
                    continue
                value = series.get(lag_dt)
                if pd.isna(value):
                    hist = series.loc[:lag_dt]
                    if not hist.empty:
                        value = hist.iloc[-1]
                df.at[idx, f"{col}_lag{lag}"] = value
                print(
                    f"[LAG][HIST] sale_id={sale_id} region={region} "
                    f"col={col} lag{lag} auction_date={auction_date_str} "
                    f"lag_date={lag_date_str} value={value}"
                )

    # Final safeguard so lag features stay complete for modelling.
    for region, grp in df.groupby("region", sort=False):
        idx = grp.index
        for col in lag_cols:
            for lag in lags:
                lag_col = f"{col}_lag{lag}"
                if lag_col in df.columns:
                    df.loc[idx, lag_col] = df.loc[idx, lag_col].bfill().ffill()

    df["auction_date"] = df["auction_date"].dt.strftime("%Y-%m-%d")
    return df


# ── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline_weather(pdf_dir: str, output_path: str):
    global SALE_DATES
    pdf_dir = Path(pdf_dir)
    
    # Auto-populate SALE_DATES from PDFs in the directory
    print("Building SALE_DATES from PDFs...")
    SALE_DATES = build_sale_dates(pdf_dir)
    print(f"   Found {len(SALE_DATES)} sales\n")
    
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
        "temperature_2m_mean_mean",
        "sunshine_duration_total",
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
    run_pipeline_weather(args.pdf_dir, args.output)
