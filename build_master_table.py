"""
build_master_table.py
=====================
Builds two master datasets from raw tea_output CSVs, per RESEARCH_PLAN.md.

  datasets/ds_analysis.csv    — Dataset A: one row per sale_id (comprehensive EDA)
  datasets/ds_prediction.csv  — Dataset B: one row per (sale_id × region) for ML

Run:  python build_master_table.py
"""

import csv
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).parent
INPUT  = ROOT / "tea_output"
OUTPUT = ROOT / "datasets"
OUTPUT.mkdir(exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

def read_csv(name: str) -> list[dict]:
    with open(INPUT / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_csv(name: str, rows: list[dict], fieldnames: list[str]) -> None:
    path = OUTPUT / name
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  OK  {path.name}  --  {len(rows)} rows x {len(fieldnames)} columns")

def safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

def mid(lo, hi):
    l, h = safe_float(lo), safe_float(hi)
    if l is not None and h is not None:
        return round((l + h) / 2, 2)
    return None

def price_range(lo, hi):
    l, h = safe_float(lo), safe_float(hi)
    if l is not None and h is not None:
        return round(h - l, 2)
    return None

def avg_nonempty(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None

def fmt(val):
    """Return value as string, or empty string for None."""
    return "" if val is None else val

def pct_change(new, old):
    n, o = safe_float(new), safe_float(old)
    if n is not None and o is not None and o != 0:
        return round((n - o) / o * 100, 2)
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD ALL SOURCE TABLES
# ══════════════════════════════════════════════════════════════════════════════

print("Loading source tables…")
sales    = read_csv("01_sales_index.csv")
offers   = read_csv("02_auction_offerings.csv")
hg       = read_csv("04_high_grown_prices.csv")
lg       = read_csv("05_low_grown_prices.csv")
offgrade = read_csv("06_offgrade_dust_prices.csv")
top      = read_csv("07_top_prices.csv")
weather  = read_csv("09_weather_features.csv")

# Sort sales by sale_number
sales.sort(key=lambda r: int(r["sale_number"]))

# Collect all sale_ids in order
SALE_IDS = [r["sale_id"] for r in sales]

# ── Lookup dicts keyed by sale_id ──────────────────────────────────────────────
sales_by_id = {r["sale_id"]: r for r in sales}

# offers → {sale_id: {category: row}}
offers_by_id: dict[str, dict] = {}
for r in offers:
    offers_by_id.setdefault(r["sale_id"], {})[r["category"]] = r

# hg → {sale_id: {(grade, segment): row}}
hg_by_id: dict[str, dict] = {}
for r in hg:
    hg_by_id.setdefault(r["sale_id"], {})[(r["grade"], r["segment"])] = r

# lg → {sale_id: {(grade, tier): row}}
lg_by_id: dict[str, dict] = {}
for r in lg:
    lg_by_id.setdefault(r["sale_id"], {})[(r["grade"], r["tier"])] = r

# offgrade → {sale_id: {(cat_type, category, elevation): row}}
od_by_id: dict[str, dict] = {}
for r in offgrade:
    od_by_id.setdefault(r["sale_id"], {})[(r["category_type"], r["category"], r["elevation"])] = r

# top prices → {sale_id: [rows]}
top_by_id: dict[str, list] = {}
for r in top:
    top_by_id.setdefault(r["sale_id"], []).append(r)

# weather → {sale_id: {region: row}}
wx_by_id: dict[str, dict] = {}
for r in weather:
    wx_by_id.setdefault(r["sale_id"], {})[r["region"]] = r


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET A  —  ds_analysis.csv  (one row per sale_id)
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding Dataset A (ds_analysis.csv)…")

# ── Enumerate all HG / LG / OD combos present in the data ────────────────────
HG_GRADES    = ["bop", "bopf", "op", "pekoe_fbop"]
HG_SEGMENTS  = [
    "best_western", "below_best_western", "plainer_western", "nuwara_eliya",
    "brighter_udapussellawa", "other_udapussellawa", "best_uva", "other_uva",
]
LG_GRADES    = [
    "bop", "bop1", "bopf", "fbop", "fbop1", "fbopf", "fbopf1", "fbopf_tippy",
    "op", "op1", "opa", "pek1", "pekoe",
]
LG_TIERS     = ["select_best", "best", "below_best", "others"]
OD_CATS      = [
    ("off_grade", "fannings_orthodox_better"), ("off_grade", "fannings_orthodox_other"),
    ("off_grade", "fannings_ctc_better"),      ("off_grade", "fannings_ctc_other"),
    ("off_grade", "brokens_good"),             ("off_grade", "brokens_other"),
    ("off_grade", "bop1a_better"),             ("off_grade", "bop1a_other"),
    ("dust",      "primary_orth_better"),      ("dust", "primary_orth_below_best"),
    ("dust",      "primary_orth_other"),       ("dust", "primary_ctc_better"),
    ("dust",      "primary_ctc_other"),        ("dust", "secondary_better"),
    ("dust",      "secondary_other"),
]
OD_ELEVS     = ["high", "medium", "low"]
OFFER_CATS   = [
    "ex_estate", "high_medium", "leafy", "semi_leafy",
    "tippy", "premium_flowery", "off_grade", "dust", "total",
]

# Focal grade sub-region mappings for summary averages
BOPF_WESTERN_SEGS  = ["best_western", "below_best_western", "plainer_western", "nuwara_eliya"]
BOPF_UVA_SEGS      = ["brighter_udapussellawa", "other_udapussellawa", "best_uva", "other_uva"]
FBOP_WESTERN_SEGS  = BOPF_WESTERN_SEGS
FBOP_UVA_SEGS      = BOPF_UVA_SEGS
OP1_TIERS          = ["select_best", "best", "below_best", "others"]

# ── Columns from 01_sales_index to include directly ──────────────────────────
SALES_KEEP = [
    "sale_id", "sale_number", "sale_year", "sale_month",
    # Auction metadata
    "total_lots", "total_kgs", "reprint_lots", "reprint_quantity",
    # Sentiment
    "sentiment_overall", "sentiment_ex_estate", "sentiment_low_grown",
    # Weather scores (numeric + text)
    "western_nuwara_eliya_weather_score", "uva_udapussellawa_weather_score",
    "low_grown_weather_score", "avg_weather_severity",
    "western_nuwara_eliya_weather_desc", "uva_udapussellawa_weather_desc",
    "low_grown_weather_desc",
    # Crop trends
    "crop_nuwara_eliya_trend", "crop_western_trend",
    "crop_uva_trend", "crop_low_grown_trend",
    # Weekly volumes — current year
    "private_sales_weekly_2026", "public_auction_weekly_2026",
    "forward_contracts_weekly_2026", "total_sold_weekly_2026",
    # Weekly volumes — prior year (YoY)
    "private_sales_weekly_2025", "public_auction_weekly_2025",
    "total_sold_weekly_2025",
    # FX — current year only
    "fx_usd_2026", "fx_gbp_2026", "fx_eur_2026", "fx_jpy_2026",
    # Production
    "sl_production_mkgs", "sl_production_yoy_variance",
    # Gross LKR — weekly, current year (not todate)
    "gross_lkr_weekly_uva_high_2026",     "gross_lkr_weekly_uva_high_2025",
    "gross_lkr_weekly_western_high_2026", "gross_lkr_weekly_western_high_2025",
    "gross_lkr_weekly_ctc_high_2026",
    "gross_lkr_weekly_high_summary_2026", "gross_lkr_weekly_high_summary_2025",
    "gross_lkr_weekly_uva_medium_2026",   "gross_lkr_weekly_uva_medium_2025",
    "gross_lkr_weekly_western_medium_2026","gross_lkr_weekly_western_medium_2025",
    "gross_lkr_weekly_ctc_medium_2026",
    "gross_lkr_weekly_medium_summary_2026","gross_lkr_weekly_medium_summary_2025",
    "gross_lkr_weekly_orthodox_low_2026", "gross_lkr_weekly_orthodox_low_2025",
    "gross_lkr_weekly_ctc_low_2026",
    "gross_lkr_weekly_low_summary_2026",  "gross_lkr_weekly_low_summary_2025",
    "gross_lkr_weekly_total_2026",        "gross_lkr_weekly_total_2025",
]

# Build column name lists for pivot blocks
offer_cols = []
for cat in OFFER_CATS:
    for metric in ["demand_score", "demand_label", "qty_mkgs", "lots", "kgs"]:
        offer_cols.append(f"off_{cat}_{metric}")

hg_price_cols = []
for grade in HG_GRADES:
    for seg in HG_SEGMENTS:
        for metric in ["price_lo", "price_hi", "price_mid", "price_range"]:
            hg_price_cols.append(f"hg_{grade}_{seg}_{metric}")

lg_price_cols = []
for grade in LG_GRADES:
    for tier in LG_TIERS:
        for metric in ["price_lo", "price_hi", "price_mid", "price_range"]:
            lg_price_cols.append(f"lg_{grade}_{tier}_{metric}")

od_price_cols = []
for cat_type, cat in OD_CATS:
    prefix = f"od_{cat_type}_{cat}"
    for elev in OD_ELEVS:
        for metric in ["price_mid", "price_range"]:
            od_price_cols.append(f"{prefix}_{elev}_{metric}")

# Top price summary cols
top_cols = [
    "top_bopf_max_price", "top_bopf_avg_price", "top_bopf_count",
    "top_pekoe_fbop_max_price", "top_pekoe_fbop_avg_price",
    "top_op1_max_price", "top_op1_avg_price",
    "top_any_max_price", "top_any_estate", "top_any_grade", "top_any_region",
]

# Focal grade summary cols (derived)
focal_cols = [
    # BOPF
    "bopf_western_avg_price_mid", "bopf_uva_avg_price_mid", "bopf_all_avg_price_mid",
    "bopf_best_western_price_mid", "bopf_nuwara_eliya_price_mid",
    "bopf_best_uva_price_mid", "bopf_brighter_udapussellawa_price_mid",
    # FBOP
    "fbop_western_avg_price_mid", "fbop_uva_avg_price_mid", "fbop_all_avg_price_mid",
    "fbop_best_western_price_mid", "fbop_brighter_udapussellawa_price_mid",
    "fbop_best_uva_price_mid",
    # OP1
    "op1_select_best_price_mid", "op1_best_price_mid",
    "op1_below_best_price_mid",  "op1_others_price_mid",
    "op1_all_avg_price_mid",     "op1_premium_spread",
    # Derived market metrics
    "sold_to_offered_ratio", "yoy_volume_change_pct",
]

# Full column order for Dataset A
DS_A_COLS = SALES_KEEP + offer_cols + hg_price_cols + lg_price_cols + od_price_cols + top_cols + focal_cols

# ── Build rows ────────────────────────────────────────────────────────────────
ds_a_rows = []

for sid in SALE_IDS:
    s = sales_by_id[sid]
    row: dict = {}

    # ── 01_sales_index core columns
    for col in SALES_KEEP:
        row[col] = s.get(col, "")

    # ── 02_auction_offerings — pivot
    off = offers_by_id.get(sid, {})
    for cat in OFFER_CATS:
        o = off.get(cat, {})
        row[f"off_{cat}_demand_score"] = o.get("demand_score", "")
        row[f"off_{cat}_demand_label"] = o.get("demand_label", "")
        row[f"off_{cat}_qty_mkgs"]     = o.get("qty_mkgs", "")
        row[f"off_{cat}_lots"]         = o.get("lots", "")
        row[f"off_{cat}_kgs"]          = o.get("kgs", "")

    # ── 04_high_grown_prices — pivot
    hg_s = hg_by_id.get(sid, {})
    for grade in HG_GRADES:
        for seg in HG_SEGMENTS:
            key = (grade, seg)
            p = hg_s.get(key, {})
            lo = p.get("price_lo_lkr")
            hi = p.get("price_hi_lkr")
            row[f"hg_{grade}_{seg}_price_lo"]    = fmt(safe_float(lo))
            row[f"hg_{grade}_{seg}_price_hi"]    = fmt(safe_float(hi))
            row[f"hg_{grade}_{seg}_price_mid"]   = fmt(mid(lo, hi))
            row[f"hg_{grade}_{seg}_price_range"] = fmt(price_range(lo, hi))

    # ── 05_low_grown_prices — pivot
    lg_s = lg_by_id.get(sid, {})
    for grade in LG_GRADES:
        for tier in LG_TIERS:
            key = (grade, tier)
            p = lg_s.get(key, {})
            lo = p.get("price_lo_lkr")
            hi = p.get("price_hi_lkr")
            row[f"lg_{grade}_{tier}_price_lo"]    = fmt(safe_float(lo))
            row[f"lg_{grade}_{tier}_price_hi"]    = fmt(safe_float(hi))
            row[f"lg_{grade}_{tier}_price_mid"]   = fmt(mid(lo, hi))
            row[f"lg_{grade}_{tier}_price_range"] = fmt(price_range(lo, hi))

    # ── 06_offgrade_dust_prices — pivot (mid + range only, keep compact)
    od_s = od_by_id.get(sid, {})
    for cat_type, cat in OD_CATS:
        prefix = f"od_{cat_type}_{cat}"
        for elev in OD_ELEVS:
            p = od_s.get((cat_type, cat, elev), {})
            lo = p.get("price_lo_lkr")
            hi = p.get("price_hi_lkr")
            row[f"{prefix}_{elev}_price_mid"]   = fmt(mid(lo, hi))
            row[f"{prefix}_{elev}_price_range"] = fmt(price_range(lo, hi))

    # ── 07_top_prices — summary stats per sale
    top_rows = top_by_id.get(sid, [])

    def top_stats(rows, grade_filter=None):
        filtered = [r for r in rows if grade_filter is None or r.get("grade","").lower() == grade_filter]
        prices = [safe_float(r["price_lkr"]) for r in filtered if safe_float(r["price_lkr"]) is not None]
        return (
            fmt(max(prices) if prices else None),
            fmt(round(sum(prices) / len(prices), 2) if prices else None),
            str(len(prices)),
        )

    bopf_max, bopf_avg, bopf_cnt   = top_stats(top_rows, "bopf")
    pfbop_max, pfbop_avg, _         = top_stats(top_rows, "fbop")
    op1_max, op1_avg, _             = top_stats(top_rows, "op1")

    row["top_bopf_max_price"]       = bopf_max
    row["top_bopf_avg_price"]       = bopf_avg
    row["top_bopf_count"]           = bopf_cnt
    row["top_pekoe_fbop_max_price"] = pfbop_max
    row["top_pekoe_fbop_avg_price"] = pfbop_avg
    row["top_op1_max_price"]        = op1_max
    row["top_op1_avg_price"]        = op1_avg

    # Overall top price for the sale
    all_prices = [(safe_float(r["price_lkr"]), r) for r in top_rows if safe_float(r["price_lkr"])]
    if all_prices:
        best_price, best_row = max(all_prices, key=lambda x: x[0])
        row["top_any_max_price"] = fmt(best_price)
        row["top_any_estate"]    = best_row.get("estate", "")
        row["top_any_grade"]     = best_row.get("grade", "")
        row["top_any_region"]    = best_row.get("region", "")
    else:
        row["top_any_max_price"] = ""
        row["top_any_estate"]    = ""
        row["top_any_grade"]     = ""
        row["top_any_region"]    = ""

    # ── Focal grade summary prices ────────────────────────────────────────────

    def hg_mid(grade, seg):
        p = hg_s.get((grade, seg), {})
        return mid(p.get("price_lo_lkr"), p.get("price_hi_lkr"))

    def lg_mid_tier(grade, tier):
        p = lg_s.get((grade, tier), {})
        return mid(p.get("price_lo_lkr"), p.get("price_hi_lkr"))

    # BOPF focal points
    bopf_w_mids  = [hg_mid("bopf", seg) for seg in BOPF_WESTERN_SEGS]
    bopf_u_mids  = [hg_mid("bopf", seg) for seg in BOPF_UVA_SEGS]
    bopf_all_mid = bopf_w_mids + bopf_u_mids

    row["bopf_western_avg_price_mid"]       = fmt(avg_nonempty(bopf_w_mids))
    row["bopf_uva_avg_price_mid"]           = fmt(avg_nonempty(bopf_u_mids))
    row["bopf_all_avg_price_mid"]           = fmt(avg_nonempty(bopf_all_mid))
    row["bopf_best_western_price_mid"]      = fmt(hg_mid("bopf", "best_western"))
    row["bopf_nuwara_eliya_price_mid"]      = fmt(hg_mid("bopf", "nuwara_eliya"))
    row["bopf_best_uva_price_mid"]          = fmt(hg_mid("bopf", "best_uva"))
    row["bopf_brighter_udapussellawa_price_mid"] = fmt(hg_mid("bopf", "brighter_udapussellawa"))

    # FBOP (pekoe_fbop) focal points
    fbop_w_mids  = [hg_mid("pekoe_fbop", seg) for seg in FBOP_WESTERN_SEGS]
    fbop_u_mids  = [hg_mid("pekoe_fbop", seg) for seg in FBOP_UVA_SEGS]
    fbop_all_mid = fbop_w_mids + fbop_u_mids

    row["fbop_western_avg_price_mid"]       = fmt(avg_nonempty(fbop_w_mids))
    row["fbop_uva_avg_price_mid"]           = fmt(avg_nonempty(fbop_u_mids))
    row["fbop_all_avg_price_mid"]           = fmt(avg_nonempty(fbop_all_mid))
    row["fbop_best_western_price_mid"]      = fmt(hg_mid("pekoe_fbop", "best_western"))
    row["fbop_brighter_udapussellawa_price_mid"] = fmt(hg_mid("pekoe_fbop", "brighter_udapussellawa"))
    row["fbop_best_uva_price_mid"]          = fmt(hg_mid("pekoe_fbop", "best_uva"))

    # OP1 focal points
    op1_mids = [lg_mid_tier("op1", tier) for tier in OP1_TIERS]

    row["op1_select_best_price_mid"] = fmt(lg_mid_tier("op1", "select_best"))
    row["op1_best_price_mid"]        = fmt(lg_mid_tier("op1", "best"))
    row["op1_below_best_price_mid"]  = fmt(lg_mid_tier("op1", "below_best"))
    row["op1_others_price_mid"]      = fmt(lg_mid_tier("op1", "others"))
    row["op1_all_avg_price_mid"]     = fmt(avg_nonempty(op1_mids))

    sb = safe_float(row["op1_select_best_price_mid"])
    ot = safe_float(row["op1_others_price_mid"])
    row["op1_premium_spread"] = fmt(round(sb - ot, 2) if sb and ot else None)

    # ── Derived market metrics
    total_kgs_val  = safe_float(s.get("total_kgs"))
    sold_26        = safe_float(s.get("total_sold_weekly_2026"))
    sold_25        = safe_float(s.get("total_sold_weekly_2025"))

    row["sold_to_offered_ratio"]   = fmt(round(sold_26 / total_kgs_val, 4) if sold_26 and total_kgs_val else None)
    row["yoy_volume_change_pct"]   = fmt(pct_change(sold_26, sold_25))

    ds_a_rows.append(row)

write_csv("ds_analysis.csv", ds_a_rows, DS_A_COLS)


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET B  —  ds_prediction.csv  (one row per sale_id × region)
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding Dataset B (ds_prediction.csv)…")

# Four weather regions matching the actual keys in 09_weather_features.csv:
#   western_high, nuwara_eliya, uva_udapussellawa, low_grown
# Dataset B has 4 rows per sale (40 rows total).
REGIONS = {
    "western_high": {
        # Western slopes BOPF / FBOP (excl. Nuwara Eliya sub-segment)
        "focal_grades": ["bopf", "pekoe_fbop"],
        "offer_cats":   ["ex_estate", "high_medium"],
        "gross_lkr_bopf": "gross_lkr_weekly_western_high_2026",
        "gross_lkr_fbop": "gross_lkr_weekly_western_medium_2026",
        "bopf_segments":  ["best_western", "below_best_western", "plainer_western"],
        "fbop_segments":  ["best_western", "below_best_western", "plainer_western"],
    },
    "nuwara_eliya": {
        # Nuwara Eliya BOPF / FBOP sub-segment only
        "focal_grades": ["bopf", "pekoe_fbop"],
        "offer_cats":   ["ex_estate", "high_medium"],
        "gross_lkr_bopf": "gross_lkr_weekly_western_high_2026",   # uses western proxy
        "gross_lkr_fbop": "gross_lkr_weekly_western_medium_2026",
        "bopf_segments":  ["nuwara_eliya"],
        "fbop_segments":  ["nuwara_eliya"],
    },
    "uva_udapussellawa": {
        "focal_grades": ["bopf", "pekoe_fbop"],
        "offer_cats":   ["ex_estate", "high_medium"],
        "gross_lkr_bopf": "gross_lkr_weekly_uva_high_2026",
        "gross_lkr_fbop": "gross_lkr_weekly_uva_medium_2026",
        "bopf_segments":  BOPF_UVA_SEGS,
        "fbop_segments":  FBOP_UVA_SEGS,
    },
    "low_grown": {
        "focal_grades": ["op1"],
        "offer_cats":   ["leafy", "semi_leafy"],
        "gross_lkr_bopf": None,
        "gross_lkr_fbop": None,
        "gross_lkr_op1":  "gross_lkr_weekly_orthodox_low_2026",
        "bopf_segments":  [],
        "fbop_segments":  [],
    },
}

# Weather columns to keep in Dataset B (per research plan — minimal but complete)
WX_KEEP = [
    # Text-derived crop and condition signals
    "text_condition_score",
    "text_crop_change",
    "text_has_rain",
    "text_has_mist",
    "text_has_bright",
    "text_raw_summary",        # crop narrative text — qualitative reference
    # Numeric weather — one per dimension
    "precipitation_sum_total",
    "sunshine_duration_total",
    "temperature_2m_mean_mean",
    "relative_humidity_2m_max_mean",
    # Lag features — 1 and 2 weeks prior (lag3 excluded per plan)
    "precipitation_sum_total_lag1",
    "precipitation_sum_total_lag2",
    "sunshine_duration_total_lag1",
    "sunshine_duration_total_lag2",
    "text_condition_score_lag1",
    "text_condition_score_lag2",
]

# Target + gross LKR reference cols
TARGET_COLS = [
    # Focal grade price targets
    "target_bopf_price_mid",     # BOPF avg mid across this region's segments
    "target_fbop_price_mid",     # FBOP (pekoe_fbop) avg mid across this region's segments
    "target_op1_price_mid",      # OP1 avg mid across all tiers (low_grown only)
    "target_op1_select_best",    # OP1 select_best tier
    "target_op1_others",         # OP1 others tier
    "target_price_direction",    # WoW direction of the region's gross_lkr: 1=up, -1=down, 0=flat
    # Gross LKR reference (the most reliable auction-level average)
    "gross_lkr_bopf_region",
    "gross_lkr_fbop_region",
    "gross_lkr_op1_region",
    # Previous week price (autoregressive feature)
    "bopf_price_mid_lag1",
    "fbop_price_mid_lag1",
    "op1_price_mid_lag1",
]

# Demand score columns (grade-matched)
DEMAND_COLS = [
    "demand_score_cat1",    # primary auction category for this region
    "demand_score_cat2",    # secondary auction category for this region
    "demand_label_cat1",
]

# Macro cols (from 01_sales_index)
MACRO_COLS = [
    "sale_number", "sale_month",
    "fx_usd_2026", "sl_production_mkgs",
    "sentiment_overall",
]

# Full column order for Dataset B
DS_B_COLS = (
    ["sale_id", "region", "auction_date"]
    + MACRO_COLS
    + WX_KEEP
    + DEMAND_COLS
    + TARGET_COLS
)

# ── Build rows ────────────────────────────────────────────────────────────────
ds_b_rows = []

# We need previous-week price for lag1 autoregressive feature
# Build price series per region first, then look up prev sale
bopf_price_series: dict[str, dict[str, float | None]] = {}  # {region: {sale_id: price}}
fbop_price_series: dict[str, dict[str, float | None]] = {}
op1_price_series:  dict[str, dict[str, float | None]] = {}

for region, cfg in REGIONS.items():
    bopf_price_series[region] = {}
    fbop_price_series[region] = {}
    op1_price_series[region]  = {}
    for sid in SALE_IDS:
        hg_s = hg_by_id.get(sid, {})
        lg_s = lg_by_id.get(sid, {})

        bopf_mids = [mid(hg_s.get(("bopf", seg), {}).get("price_lo_lkr"),
                         hg_s.get(("bopf", seg), {}).get("price_hi_lkr"))
                     for seg in cfg["bopf_segments"]]
        fbop_mids = [mid(hg_s.get(("pekoe_fbop", seg), {}).get("price_lo_lkr"),
                         hg_s.get(("pekoe_fbop", seg), {}).get("price_hi_lkr"))
                     for seg in cfg["fbop_segments"]]
        op1_mids  = [mid(lg_s.get(("op1", tier), {}).get("price_lo_lkr"),
                         lg_s.get(("op1", tier), {}).get("price_hi_lkr"))
                     for tier in OP1_TIERS] if region == "low_grown" else []

        bopf_price_series[region][sid] = avg_nonempty(bopf_mids)
        fbop_price_series[region][sid] = avg_nonempty(fbop_mids)
        op1_price_series[region][sid]  = avg_nonempty(op1_mids)

# Now build Dataset B rows
for idx, sid in enumerate(SALE_IDS):
    s   = sales_by_id[sid]
    off = offers_by_id.get(sid, {})
    hg_s = hg_by_id.get(sid, {})
    lg_s = lg_by_id.get(sid, {})

    prev_sid = SALE_IDS[idx - 1] if idx > 0 else None

    for region, cfg in REGIONS.items():
        wx = wx_by_id.get(sid, {}).get(region, {})
        row: dict = {}

        # Core identifiers
        row["sale_id"]     = sid
        row["region"]      = region
        row["auction_date"] = wx.get("auction_date", s.get("sale_date_raw", ""))

        # Macro context from 01_sales_index
        for col in MACRO_COLS:
            row[col] = s.get(col, "")

        # Weather columns
        for col in WX_KEEP:
            row[col] = wx.get(col, "")

        # Demand scores — matched to region's auction categories
        cats = cfg["offer_cats"]
        o1 = off.get(cats[0], {}) if len(cats) > 0 else {}
        o2 = off.get(cats[1], {}) if len(cats) > 1 else {}
        row["demand_score_cat1"]  = o1.get("demand_score", "")
        row["demand_label_cat1"]  = o1.get("demand_label", "")
        row["demand_score_cat2"]  = o2.get("demand_score", "")

        # ── Target variables ──────────────────────────────────────────────────

        # BOPF target — avg mid across this region's segments
        bopf_mids = [mid(hg_s.get(("bopf", seg), {}).get("price_lo_lkr"),
                         hg_s.get(("bopf", seg), {}).get("price_hi_lkr"))
                     for seg in cfg["bopf_segments"]]
        bopf_avg = avg_nonempty(bopf_mids)
        row["target_bopf_price_mid"] = fmt(bopf_avg)

        # FBOP target
        fbop_mids = [mid(hg_s.get(("pekoe_fbop", seg), {}).get("price_lo_lkr"),
                         hg_s.get(("pekoe_fbop", seg), {}).get("price_hi_lkr"))
                     for seg in cfg["fbop_segments"]]
        fbop_avg = avg_nonempty(fbop_mids)
        row["target_fbop_price_mid"] = fmt(fbop_avg)

        # OP1 targets (low_grown only)
        if region == "low_grown":
            op1_tier_mids = {
                tier: mid(lg_s.get(("op1", tier), {}).get("price_lo_lkr"),
                          lg_s.get(("op1", tier), {}).get("price_hi_lkr"))
                for tier in OP1_TIERS
            }
            row["target_op1_price_mid"]   = fmt(avg_nonempty(list(op1_tier_mids.values())))
            row["target_op1_select_best"] = fmt(op1_tier_mids.get("select_best"))
            row["target_op1_others"]      = fmt(op1_tier_mids.get("others"))
        else:
            row["target_op1_price_mid"]   = ""
            row["target_op1_select_best"] = ""
            row["target_op1_others"]      = ""

        # Gross LKR reference columns (the auction-level summary averages)
        bopf_col = cfg.get("gross_lkr_bopf")
        fbop_col = cfg.get("gross_lkr_fbop")
        op1_col  = cfg.get("gross_lkr_op1") if region == "low_grown" else None

        row["gross_lkr_bopf_region"] = s.get(bopf_col, "") if bopf_col else ""
        row["gross_lkr_fbop_region"] = s.get(fbop_col, "") if fbop_col else ""
        row["gross_lkr_op1_region"]  = s.get(op1_col,  "") if op1_col  else ""

        # Price direction — WoW change in focal price for this region.
        # Prefer gross_lkr (more reliable auction summary); fall back to computed target price.
        primary_gross_col = bopf_col if bopf_col else op1_col
        curr_gross = safe_float(s.get(primary_gross_col, "")) if primary_gross_col else None
        if prev_sid:
            prev_s_row = sales_by_id[prev_sid]
            prev_gross = safe_float(prev_s_row.get(primary_gross_col, "")) if primary_gross_col else None
        else:
            prev_gross = None

        if curr_gross is not None and prev_gross is not None:
            direction = 1 if curr_gross > prev_gross else (-1 if curr_gross < prev_gross else 0)
        else:
            # Fall back: use computed focal price vs previous sale's price from the series cache
            curr_bopf  = bopf_price_series[region].get(sid)
            curr_fbop  = fbop_price_series[region].get(sid)
            curr_op1   = op1_price_series[region].get(sid)
            curr_target = curr_bopf or curr_fbop or curr_op1

            prev_bopf_  = bopf_price_series[region].get(prev_sid) if prev_sid else None
            prev_fbop_  = fbop_price_series[region].get(prev_sid) if prev_sid else None
            prev_op1_   = op1_price_series[region].get(prev_sid)  if prev_sid else None
            prev_target = prev_bopf_ or prev_fbop_ or prev_op1_

            if curr_target is not None and prev_target is not None:
                direction = 1 if curr_target > prev_target else (-1 if curr_target < prev_target else 0)
            else:
                direction = ""
        row["target_price_direction"] = direction

        # Autoregressive lag 1 price (previous sale's focal price)
        row["bopf_price_mid_lag1"] = fmt(bopf_price_series[region].get(prev_sid) if prev_sid else None)
        row["fbop_price_mid_lag1"] = fmt(fbop_price_series[region].get(prev_sid) if prev_sid else None)
        row["op1_price_mid_lag1"]  = fmt(op1_price_series[region].get(prev_sid)  if prev_sid else None)

        ds_b_rows.append(row)

write_csv("ds_prediction.csv", ds_b_rows, DS_B_COLS)

print("\nDone. Datasets written to datasets/")
