"""
Forbes & Walker Tea Market Report – Multi-Table Pipeline
=========================================================
Produces 8 focused CSVs instead of one wide flat file.
Every table shares sale_id = "SALE_{number}_{year}" as the join key.

OUTPUT FILES
------------
01_sales_index.csv          – One row per sale: meta, totals, sentiment, weather
02_auction_offerings.csv    – One row per category per sale (lots + kgs on offer)
03_quantity_sold.csv        – Weekly/YTD sold volumes and average prices
04_high_grown_prices.csv    – One row per segment×grade per sale (HG price ranges)
05_low_grown_prices.csv     – One row per grade×tier per sale  (LG price ranges)
06_offgrade_dust_prices.csv – One row per category×elevation per sale
07_top_prices.csv           – One row per estate record (long/tidy format)
08_column_dictionary.csv    – Data dictionary: one row per column across all tables

USAGE
-----
    python tea_pipeline_v2.py reports/          # whole folder
    python tea_pipeline_v2.py report1.pdf report2.pdf
    python tea_pipeline_v2.py                   # uses default sample path
"""

import pdfplumber
import pandas as pd
import re
import os
import sys
import time
import requests
from pathlib import Path
from datetime import datetime
from weather_pipeline import (
    parse_region_weather_text, run_pipeline_weather,
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def clean(text):
    return re.sub(r'\s+', ' ', text or '').strip()

def parse_price_range(raw):
    raw = str(raw).strip().replace(' ', '')
    if not raw or raw.upper() in ('N/A', 'NIL', '-', ''):
        return None, None
    m = re.match(r'^(\d+)-(\d+)$', raw)
    if m:  return float(m.group(1)), float(m.group(2))
    m = re.match(r'^(\d+)-$', raw)
    if m:  return float(m.group(1)), None
    m = re.match(r'^-(\d+)$', raw)
    if m:  return None, float(m.group(1))
    m = re.match(r'^(\d+)$', raw)
    if m:  v = float(m.group(1)); return v, v
    return None, None

def weather_score(text):
    t = text.lower()
    if 'heavy rain' in t or 'very wet' in t: return 5
    if 'rain' in t and 'bright' not in t:    return 4
    if 'shower' in t:                        return 3
    if 'occasional shower' in t:             return 2
    if 'bright' in t or 'sunny' in t:        return 1
    return 3

def sentiment_from_text(text):
    t = text.lower()
    pos = ['dearer', 'improved', 'appreciated', 'gained', 'firm',
           'good demand', 'premium', 'strong', 'better demand',
           'well-made', 'active', 'support']
    neg = ['easier', 'bearish', 'declined', 'lower', 'irregular',
           'less demand', 'withdrawn', 'withdrawal', 'limited',
           'neglected', 'discounted']
    p = sum(t.count(w) for w in pos)
    n = sum(t.count(w) for w in neg)
    total = p + n
    return round((p - n) / total, 4) if total else 0.0

def sale_id(sale_number, sale_year):
    return f"SALE_{sale_number:02d}_{sale_year}"


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION MODULES  (return plain Python dicts / lists)
# ─────────────────────────────────────────────────────────────────────────────

# ── MODULE 1 : Header ────────────────────────────────────────────────────────
def extract_header(full_text):
    meta = {}
    m = re.search(r'SALE\s+NO\s*[:\-]?\s*(\d+)', full_text, re.I)
    meta['sale_number'] = int(m.group(1)) if m else None

    m = re.search(
        r'(\d+\w*/\d+\w*)\s+(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|'
        r'JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+(\d{4})',
        full_text, re.I)
    if m:
        meta['sale_date_raw'] = f"{m.group(1)} {m.group(2).capitalize()} {m.group(3)}"
        meta['sale_year']  = int(m.group(3))
        meta['sale_month'] = m.group(2).capitalize()
    else:
        meta['sale_date_raw'] = meta['sale_year'] = meta['sale_month'] = None
    return meta


# ── MODULE 2 : Overall Market + Auction Details (Page 2) ────────────────────
CATEGORY_MAP = {
    'ExEstate': 'ex_estate', 'Ex Estate': 'ex_estate',
    'High&Medium': 'high_medium', 'High & Medium': 'high_medium',
    'Leafy': 'leafy', 'SemiLeafy': 'semi_leafy', 'Semi Leafy': 'semi_leafy',
    'Tippy/SmallLeaf': 'tippy', 'Tippy/Small Leaf': 'tippy',
    'PremiumFlowery': 'premium_flowery', 'Premium Flowery': 'premium_flowery',
    'OffGrade': 'off_grade', 'Off Grade': 'off_grade',
    'Dust': 'dust', 'Total': 'total',
}
DEMAND_MAP = {
    'Good': 'good', 'Fair': 'fair',
    'Fairgeneral': 'fair_general', 'Fair general': 'fair_general',
    'Less': 'less', 'Poor': 'poor',
}
DEMAND_NUM = {'good': 4, 'fair': 3, 'fair_general': 3, 'less': 2, 'poor': 1}

def extract_overall_market(page2_text):
    """Returns list of dicts – one per category row."""
    rows = []
    pattern = re.compile(
        r'(ExEstate|Ex Estate|High&Medium|High & Medium|Leafy|SemiLeafy|'
        r'Semi Leafy|Tippy/SmallLeaf|Tippy/Small Leaf|PremiumFlowery|'
        r'Premium Flowery|OffGrade|Off Grade|Dust|Total)\s+'
        r'(\d+\.\d+)\s+(\w[\w ]*)', re.I)
    for m in pattern.finditer(page2_text):
        cat    = CATEGORY_MAP.get(m.group(1).strip(), m.group(1).lower().replace(' ', '_'))
        demand = DEMAND_MAP.get(clean(m.group(3)), clean(m.group(3)).lower())
        rows.append({
            'category':         cat,
            'qty_mkgs':         float(m.group(2)),
            'demand_label':     demand,
            'demand_score':     DEMAND_NUM.get(demand, 3),
        })
    return rows

def extract_auction_details(page2_text):
    """Returns dict of totals + list of per-category lot/quantity rows."""
    totals = {}
    m = re.search(r'([\d,]+)\s*LOTS\s*TOTALLING\s*([\d,]+)', page2_text, re.I)
    if m:
        totals['total_lots'] = int(m.group(1).replace(',', ''))
        totals['total_kgs']  = int(m.group(2).replace(',', ''))

    m = re.search(r'Re-?Prints?\s+([\d,]+)\s+([\d,]+)', page2_text, re.I)
    if m:
        totals['reprint_lots']     = int(m.group(1).replace(',', ''))
        totals['reprint_quantity'] = int(m.group(2).replace(',', ''))

    m = re.search(
        r'(\d{2}/\d{2}/\d{4})\s+(\d{2}/\d{2}/\d{4})\s+(\d{2}/\d{2}/\d{4})',
        page2_text)
    if m:
        totals['settlement_10pct']   = m.group(1)
        totals['settlement_buyers']  = m.group(2)
        totals['settlement_sellers'] = m.group(3)

    label_map = {
        'ExEstate': 'ex_estate', 'High&Medium': 'high_medium',
        'LowGrown-Leafy': 'leafy', 'LowGrown-SemiLeafy': 'semi_leafy',
        'LowGrown-Tippy': 'tippy', 'PremiumFlowery': 'premium_flowery',
        'OffGrades': 'off_grade', 'Dust': 'dust',
    }
    cat_rows = []
    cat_pattern = re.compile(
        r'(ExEstate|High&Medium|LowGrown-Leafy|LowGrown-SemiLeafy|'
        r'LowGrown-Tippy|PremiumFlowery|OffGrades|Dust)\s+([\d,]+)\s+([\d,]+)',
        re.I)
    for m in cat_pattern.finditer(page2_text):
        key = label_map.get(m.group(1).strip(), m.group(1).lower())
        cat_rows.append({
            'category': key,
            'lots':     int(m.group(2).replace(',', '')),
            'kgs':      int(m.group(3).replace(',', '')),
        })
    return totals, cat_rows


# ── MODULE 3 : Commentary + Sentiment ───────────────────────────────────────
def extract_commentary(full_text):
    m = re.search(r'COMMENTS\s*(.*?)(?:NATIONAL TEA|WORLD TEA|CROP AND|$)',
                  full_text, re.S | re.I)
    if not m:
        return {'commentary': '', 'sentiment_overall': 0.0,
                'sentiment_ex_estate': 0.0, 'sentiment_low_grown': 0.0}
    comments = clean(m.group(1))
    lg_start = comments.lower().find('low grown')
    return {
        'commentary':           comments[:1500],
        'sentiment_overall':    sentiment_from_text(comments),
        'sentiment_ex_estate':  sentiment_from_text(
            re.sub(r'(?i)low grown.*', '', comments)),
        'sentiment_low_grown':  sentiment_from_text(
            comments[lg_start:] if lg_start >= 0 else ''),
    }


# ── MODULE 4 : Weather + Crop ────────────────────────────────────────────────
def extract_weather(full_text):
    regions = {
        'western_nuwara_eliya':
            r'Western/Nuwara Eliya Regions?\s*(.*?)(?:Uva/Uda|Low Grown|Crop|$)',
        'uva_udapussellawa':
            r'Uva/Uda\w* Regions?\s*(.*?)(?:Low Grown|Crop|$)',
        'low_grown':
            r'Low Growns?\s*(.*?)(?:FORBES|Crop|$)',
    }
    result = {}
    for region, pat in regions.items():
        m = re.search(pat, full_text, re.S | re.I)
        if m:
            desc = clean(m.group(1))
            result[f'{region}_weather_desc']  = desc[:250]
            result[f'{region}_weather_score'] = weather_score(desc)

    m = re.search(r'Crop\s*(.*?)(?:FORBES|$)', full_text, re.S | re.I)
    crop = clean(m.group(1)) if m else ''
    result['crop_notes'] = crop[:300]

    scores = [v for k, v in result.items() if k.endswith('_score')]
    result['avg_weather_severity'] = round(sum(scores)/len(scores), 2) if scores else None

    # Crop trend encoding per region
    for region in ['nuwara_eliya', 'western', 'uva', 'low_grown']:
        cl = crop.lower()
        if region.replace('_', ' ') in cl:
            if 'decrease' in cl:   trend = -1
            elif 'increase' in cl: trend = 1
            else:                  trend = 0
            result[f'crop_{region}_trend'] = trend

    return result


# ── MODULE 5 : Production ────────────────────────────────────────────────────
def extract_production(full_text):
    result = {}
    m = re.search(r'totalled\s+(?:at\s+)?([\d.]+)\s+M/Kgs', full_text, re.I)
    if m: result['sl_production_mkgs'] = float(m.group(1))

    m = re.search(r'([\d.]+)\s+M/Kgs\s+(?:decrease|decline)', full_text, re.I)
    if m:   result['sl_production_yoy_variance'] = -float(m.group(1))
    else:
        m = re.search(r'([\d.]+)\s+M/Kgs\s+increase', full_text, re.I)
        if m: result['sl_production_yoy_variance'] = float(m.group(1))

    for elev in ['HIGH', 'MEDIUM', 'LOW']:
        m = re.search(
            rf'{elev}\s+([\d,]+)\s+([\d,]+)\s+([+-]?[\d.]+)\s+([+-]?[\d.]+)%',
            full_text, re.I)
        if m:
            k = elev.lower()
            result[f'prod_{k}_2026'] = int(m.group(1).replace(',', ''))
            result[f'prod_{k}_2025'] = int(m.group(2).replace(',', ''))
            result[f'prod_{k}_var']  = float(m.group(3))
            result[f'prod_{k}_pct']  = float(m.group(4))
    return result


# ── MODULE 6 : Quantity Sold + Exchange Rates ────────────────────────────────
def extract_quantity_sold(full_text):
    result = {}
    channel_map = {
        'PRIVATESALES':    'private_sales',
        'PUBLICAUCTION':   'public_auction',
        'FORWARDCONTRACTS':'forward_contracts',
        'TOTAL':           'total_sold',
    }
    for raw, key in channel_map.items():
        m = re.search(raw + r'\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)',
                      full_text, re.I)
        if m:
            result[f'{key}_weekly_2026']  = int(m.group(1).replace(',',''))
            result[f'{key}_weekly_2025']  = int(m.group(2).replace(',',''))
            result[f'{key}_todate_2026']  = int(m.group(3).replace(',',''))
            result[f'{key}_todate_2025']  = int(m.group(4).replace(',',''))

    # Average price per auction (most recent row)
    for m in re.finditer(
        r'(\d+\w+FEBRUARY|\d+\w+MARCH|\d+\w+JANUARY|\d+\w+APRIL)\s+(\d{4})\s+'
        r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+'
        r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+'
        r'([\d.]+)\s+([\d.]+)\s+([\d.]+)',
        full_text, re.I):
        label = f"{m.group(1).lower()}_{m.group(2)}"
        result[f'avg_qty_mkgs_{label}_2026']  = float(m.group(3))
        result[f'avg_lkr_{label}_2026']       = float(m.group(6))
        result[f'avg_usd_{label}_2026']       = float(m.group(9))

    for currency, key in [('USD','usd'),('STG.PD','gbp'),('EURO','eur'),('YEN','jpy')]:
        m = re.search(
            currency.replace('.', r'\.') + r'\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',
            full_text, re.I)
        if m:
            result[f'fx_{key}_2026'] = float(m.group(1))
            result[f'fx_{key}_2025'] = float(m.group(2))
            result[f'fx_{key}_2024'] = float(m.group(3))
    return result


# ── MODULE 7 : Gross Sales Average ──────────────────────────────────────────
def extract_gross_averages(full_text):
    result = {}
    seg_map = {
        'UvaHighGrown':             'uva_high',
        'WesternHighGrown':         'western_high',
        'CTCHighGrown':             'ctc_high',
        r'HighGrown\(Summary\)':    'high_summary',
        'UvaMediumGrown':           'uva_medium',
        'WesternMediumGrown':       'western_medium',
        'CTCMediumGrown':           'ctc_medium',
        r'MediumGrown\(Summary\)':  'medium_summary',
        'OrthodoxLowGrown':         'orthodox_low',
        'CTCLowGrown':              'ctc_low',
        r'LowGrown\(Summary\)':     'low_summary',
        'Total':                    'total',
    }
    for pat, key in seg_map.items():
        m = re.search(
            pat + r'\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
                  r'\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',
            full_text, re.I)
        if m:
            result[f'gross_lkr_weekly_{key}_2026'] = float(m.group(1))
            result[f'gross_lkr_weekly_{key}_2025'] = float(m.group(2))
            result[f'gross_lkr_todate_{key}_2026'] = float(m.group(4))
            result[f'gross_lkr_todate_{key}_2025'] = float(m.group(5))
    return result


# ── MODULE 8 : Price Quotations – High Grown ─────────────────────────────────
HG_SEGMENTS = {
    'BestWesterns':           'best_western',
    'BelowBestWesterns':      'below_best_western',
    'PlainerWesterns':        'plainer_western',
    'NuwaraEliyas':           'nuwara_eliya',
    'BrighterUdapussellawas': 'brighter_udapussellawa',
    'OtherUdapussellawas':    'other_udapussellawa',
    'BestUvas':               'best_uva',
    'OtherUvas':              'other_uva',
}
HG_GRADES = ['bop', 'bopf', 'pekoe_fbop', 'op']

def extract_hg_prices(price_pages_text):
    """Returns list of dicts – one per (segment × grade) with lo/hi prices."""
    rows = []
    for raw_seg, seg in HG_SEGMENTS.items():
        label_re = raw_seg.replace(' ', r'\s*')
        m = re.search(
            label_re + r'\s+([\d\-]+)\s+([\d\-]+)\s+([\d\-]+)\s+([\d\-]+)'
                       r'\s+([\d\-]+)\s+([\d\-]+)\s+([\d\-]+)\s+([\d\-]+)',
            price_pages_text, re.I)
        if not m:
            continue
        # 8 groups: prev_bop curr_bop prev_bopf curr_bopf prev_pekoe curr_pekoe prev_op curr_op
        for i, grade in enumerate(HG_GRADES):
            lo, hi = parse_price_range(m.group(2 + i * 2))
            rows.append({
                'elevation':    'high_grown',
                'segment':      seg,
                'grade':        grade,
                'price_lo_lkr': lo,
                'price_hi_lkr': hi,
            })
    return rows


# ── MODULE 9 : Price Quotations – Low Grown ──────────────────────────────────
LG_GRADES = {
    r'FBOPF\s*\(TIPPY\)/FBOPFSP':  'fbopf_tippy',
    r'FBOPF1':                      'fbopf1',
    r'(?<![A-Z])FBOPF(?!1)':        'fbopf',
    r'(?<![A-Z])FBOP1':             'fbop1',
    r'(?<![A-Z])FBOP(?!F|1)':       'fbop',
    r'(?<![A-Z])BOP1':              'bop1',
    r'(?<![A-Z])BOPF':              'bopf',
    r'(?<![A-Z])BOP(?!F|1)':        'bop',
    r'(?<![A-Z])OP1':               'op1',
    r'(?<![A-Z])OPA':               'opa',
    r'(?<![A-Z])OP(?!A|1)':         'op',
    r'(?<![A-Z])PEKOE':             'pekoe',
    r'(?<![A-Z])PEK1':              'pek1',
}
LG_TIERS = ['select_best', 'best', 'below_best', 'others']

def extract_lg_prices(price_pages_text):
    """Returns list of dicts – one per (grade × tier) with lo/hi prices."""
    rows = []
    for row_re, grade in LG_GRADES.items():
        m = re.search(
            row_re + r'\s+([\d\-]+)\s+([\d\-]+)\s+([\d\-]+)\s+([\d\-]+)'
                     r'\s+([\d\-]+)\s+([\d\-]+)\s+([\d\-]+)\s+([\d\-]+)',
            price_pages_text, re.I)
        if not m:
            continue
        for i, tier in enumerate(LG_TIERS):
            lo, hi = parse_price_range(m.group(2 + i * 2))
            rows.append({
                'elevation':    'low_grown',
                'grade':        grade,
                'tier':         tier,
                'price_lo_lkr': lo,
                'price_hi_lkr': hi,
            })
    return rows


# ── MODULE 10 : Off-Grade + Dust Prices ──────────────────────────────────────
OG_ROWS = {
    r'BetterFannings\(Orthodox\)':  'fannings_orthodox_better',
    r'BetterFannings\(CTC\)':       'fannings_ctc_better',
    r'OtherFannings\(Orthodox\)':   'fannings_orthodox_other',
    r'OtherFannings\(CTC\)':        'fannings_ctc_other',
    r'GoodBrokens':                 'brokens_good',
    r'OtherBrokens':                'brokens_other',
    r'BetterBOP1As':                'bop1a_better',
    r'OtherBOP1As':                 'bop1a_other',
}
DUST_ROWS = {
    r'BetterPrimaryDust\(Orthodox\)':    'primary_orth_better',
    r'BetterPrimaryDust\(CTC\)P\.Dust':  'primary_ctc_better',
    r'BelowBestPrimaryDust\(Orthodox\)': 'primary_orth_below_best',
    r'OtherPrimaryDust\(CTC\)P\.Dust':   'primary_ctc_other',
    r'OtherPrimaryDust\(Orthodox\)':     'primary_orth_other',
    r'BetterSecondaryDust':              'secondary_better',
    r'OtherSecondaryDust':               'secondary_other',
}
ELEVATIONS = ['high', 'medium', 'low']

def extract_offgrade_dust_prices(price_pages_text):
    """Returns list of dicts – one per (category × elevation)."""
    rows = []
    for category_type, row_dict in [('off_grade', OG_ROWS), ('dust', DUST_ROWS)]:
        for row_re, label in row_dict.items():
            m = re.search(
                row_re + r'\s+([\d\-N/Aa]+)\s+([\d\-N/Aa]+)'
                         r'\s+([\d\-N/Aa]+)\s+([\d\-N/Aa]+)'
                         r'\s+([\d\-N/Aa]+)\s+([\d\-N/Aa]+)',
                price_pages_text, re.I)
            if not m:
                continue
            for i, elev in enumerate(ELEVATIONS):
                lo, hi = parse_price_range(m.group(2 + i * 2))
                rows.append({
                    'category_type': category_type,
                    'category':      label,
                    'elevation':     elev,
                    'price_lo_lkr':  lo,
                    'price_hi_lkr':  hi,
                })
    return rows


# ── MODULE 11 : Top Prices ────────────────────────────────────────────────────
TOP_PRICE_REGIONS = [
    'WESTERN MEDIUM', 'WESTERN HIGH', 'NUWARA ELIYAS', 'UDAPUSSELLAWAS',
    'UVA HIGH', 'UVA MEDIUM', 'LOW GROWNS', 'PREMIUM FLOWERY',
    'DUSTS', 'OFF GRADES', 'OTHERS',
    'UNORTHODOX HIGH', 'UNORTHODOX MEDIUM', 'UNORTHODOX LOW',
]
TOP_PAT = re.compile(
    r'([A-Z][A-Za-z\s]+?)\s+'
    r'(FBOPF?ExSp\d?|FBOPFSp|BOPFSp|BOPSp|FBOPF\d?|FBOP\d?|'
    r'BOP\d?A?|BOPF\d?|OP\d?A?|OPA|PEKOE|PEK\d?|FGS\d?|BM|'
    r'BP\d?|CTC\s*PF\d|CTC\s*BP\d|CTC\s*BPS|DUST\d?|PD)\s*'
    r'(@)?\s*([\d,]+)'
)

def extract_top_prices(pages_text):
    records = []
    current_region = 'Unknown'
    for line in pages_text.split('\n'):
        for rh in TOP_PRICE_REGIONS:
            if rh.lower() in line.lower():
                current_region = rh
        m = TOP_PAT.search(line)
        if m:
            records.append({
                'estate':    clean(m.group(1)),
                'grade':     m.group(2).replace(' ', ''),
                'fw_sold':   1 if m.group(3) == '@' else 0,
                'price_lkr': int(m.group(4).replace(',', '')),
                'region':    current_region,
            })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# MASTER EXTRACTION: reads one PDF, returns all table fragments
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf(pdf_path):
    print(f"  Processing: {pdf_path}")
    with pdfplumber.open(pdf_path) as pdf:
        pages = [p.extract_text() or '' for p in pdf.pages]
    full   = '\n'.join(pages)
    page2  = pages[1] if len(pages) > 1 else full
    price_pages = '\n'.join(pages[5:12])
    top_pages   = '\n'.join(pages[9:12])

    header   = extract_header(full)
    sn       = header.get('sale_number')
    sy       = header.get('sale_year')
    sid      = sale_id(sn, sy) if sn and sy else f"UNKNOWN_{os.path.basename(str(pdf_path))}"

    market_rows         = extract_overall_market(page2)
    totals, cat_rows    = extract_auction_details(page2)
    commentary          = extract_commentary(full)
    weather             = extract_weather(full)
    text_weather_regions = parse_region_weather_text(full)
    production          = extract_production(full)
    qty_sold      = extract_quantity_sold(full)
    gross_avg     = extract_gross_averages(full)
    hg_prices     = extract_hg_prices(price_pages)
    lg_prices     = extract_lg_prices(price_pages)
    og_prices     = extract_offgrade_dust_prices(price_pages)
    top_prices    = extract_top_prices(top_pages)

    return {
        'sale_id':     sid,
        'header':      header,
        'market_rows': market_rows,
        'totals':      totals,
        'cat_rows':    cat_rows,
        'commentary':  commentary,
        'weather':              weather,
        'text_weather_regions': text_weather_regions,
        'production':           production,
        'qty_sold':    qty_sold,
        'gross_avg':   gross_avg,
        'hg_prices':   hg_prices,
        'lg_prices':   lg_prices,
        'og_prices':   og_prices,
        'top_prices':  top_prices,
        'source_file': os.path.basename(str(pdf_path)),
        'extracted_at': datetime.now().isoformat(timespec='seconds'),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TABLE BUILDERS  (one per output CSV)
# ─────────────────────────────────────────────────────────────────────────────

def build_sales_index(extractions):
    """
    01_sales_index.csv
    One row per sale. Joins: meta + totals + commentary + weather + production
    + qty_sold + gross averages + fx rates.
    """
    rows = []
    for e in extractions:
        row = {
            'sale_id':    e['sale_id'],
            'sale_number': e['header'].get('sale_number'),
            'sale_date_raw': e['header'].get('sale_date_raw'),
            'sale_year':  e['header'].get('sale_year'),
            'sale_month': e['header'].get('sale_month'),
            'source_file': e['source_file'],
            'extracted_at': e['extracted_at'],
        }
        row.update(e['totals'])
        row.update(e['commentary'])
        row.update(e['weather'])
        row.update(e['production'])
        row.update(e['qty_sold'])
        row.update(e['gross_avg'])
        rows.append(row)
    return pd.DataFrame(rows)


def build_auction_offerings(extractions):
    """
    02_auction_offerings.csv
    One row per (sale × category).
    Columns: sale_id, category, qty_mkgs, demand_label, demand_score, lots, kgs
    """
    rows = []
    for e in extractions:
        sid = e['sale_id']
        # Build lookup from cat_rows (lots + kgs)
        lots_lookup = {r['category']: r for r in e['cat_rows']}
        for mrow in e['market_rows']:
            cat = mrow['category']
            lot_info = lots_lookup.get(cat, {})
            rows.append({
                'sale_id':      sid,
                'category':     cat,
                'qty_mkgs':     mrow.get('qty_mkgs'),
                'demand_label': mrow.get('demand_label'),
                'demand_score': mrow.get('demand_score'),
                'lots':         lot_info.get('lots'),
                'kgs':          lot_info.get('kgs'),
            })
    return pd.DataFrame(rows)


def build_quantity_sold(extractions):
    """
    03_quantity_sold.csv
    One row per sale. Weekly/YTD volumes, average prices, FX rates.
    Subset of sales_index for clarity.
    """
    keep_prefixes = (
        'sale_id', 'sale_number', 'sale_year', 'sale_month',
        'private_sales_', 'public_auction_', 'forward_contracts_',
        'total_sold_', 'avg_qty_', 'avg_lkr_', 'avg_usd_',
        'fx_usd_', 'fx_gbp_', 'fx_eur_', 'fx_jpy_',
    )
    rows = []
    for e in extractions:
        row = {
            'sale_id':     e['sale_id'],
            'sale_number': e['header'].get('sale_number'),
            'sale_year':   e['header'].get('sale_year'),
            'sale_month':  e['header'].get('sale_month'),
        }
        row.update(e['qty_sold'])
        rows.append(row)
    return pd.DataFrame(rows)


def build_hg_prices(extractions):
    """
    04_high_grown_prices.csv
    One row per (sale × segment × grade).
    """
    rows = []
    for e in extractions:
        for r in e['hg_prices']:
            rows.append({'sale_id': e['sale_id'], **r})
    return pd.DataFrame(rows)


def build_lg_prices(extractions):
    """
    05_low_grown_prices.csv
    One row per (sale × grade × tier).
    """
    rows = []
    for e in extractions:
        for r in e['lg_prices']:
            rows.append({'sale_id': e['sale_id'], **r})
    return pd.DataFrame(rows)


def build_offgrade_dust_prices(extractions):
    """
    06_offgrade_dust_prices.csv
    One row per (sale × category_type × category × elevation).
    """
    rows = []
    for e in extractions:
        for r in e['og_prices']:
            rows.append({'sale_id': e['sale_id'], **r})
    return pd.DataFrame(rows)


def build_top_prices(extractions):
    """
    07_top_prices.csv
    One row per estate-grade record (long format, many rows per sale).
    """
    rows = []
    for e in extractions:
        for r in e['top_prices']:
            rows.append({'sale_id': e['sale_id'], **r})
    return pd.DataFrame(rows)





# ─────────────────────────────────────────────────────────────────────────────
# COLUMN DICTIONARY BUILDER
# ─────────────────────────────────────────────────────────────────────────────

COLUMN_DEFINITIONS = {
    # ── 01_sales_index ──────────────────────────────────────────────────────
    'sale_id':                       ('01_sales_index', 'Primary key: SALE_{number}_{year}', 'text'),
    'sale_number':                   ('01_sales_index', 'Auction sale number within the year', 'integer'),
    'sale_date_raw':                 ('01_sales_index', 'Date string as printed in PDF header', 'text'),
    'sale_year':                     ('01_sales_index', 'Calendar year of the sale', 'integer'),
    'sale_month':                    ('01_sales_index', 'Month name of the sale', 'text'),
    'source_file':                   ('01_sales_index', 'PDF filename this row was extracted from', 'text'),
    'extracted_at':                  ('01_sales_index', 'ISO timestamp of extraction run', 'datetime'),
    'total_lots':                    ('01_sales_index', 'Total lots offered across all categories', 'integer'),
    'total_kgs':                     ('01_sales_index', 'Total kg offered across all categories', 'integer'),
    'reprint_lots':                  ('01_sales_index', 'Number of re-print lots', 'integer'),
    'reprint_quantity':              ('01_sales_index', 'Re-print quantity in kg', 'integer'),
    'settlement_10pct':              ('01_sales_index', '10% payment date (DD/MM/YYYY)', 'date'),
    'settlement_buyers':             ('01_sales_index', "Buyers' prompt date (DD/MM/YYYY)", 'date'),
    'settlement_sellers':            ('01_sales_index', "Sellers' prompt date (DD/MM/YYYY)", 'date'),
    'commentary':                    ('01_sales_index', 'Full market commentary text (up to 1500 chars)', 'text'),
    'sentiment_overall':             ('01_sales_index', 'Rule-based sentiment of full commentary: −1 (bearish) to +1 (bullish)', 'float'),
    'sentiment_ex_estate':           ('01_sales_index', 'Sentiment score for Ex-Estate commentary segment', 'float'),
    'sentiment_low_grown':           ('01_sales_index', 'Sentiment score for Low Grown commentary segment', 'float'),
    'western_nuwara_eliya_weather_desc': ('01_sales_index', 'Raw weather text for Western/Nuwara Eliya region', 'text'),
    'western_nuwara_eliya_weather_score':('01_sales_index', 'Weather severity 1=sunny … 5=heavy rain', 'integer'),
    'uva_udapussellawa_weather_desc':    ('01_sales_index', 'Raw weather text for Uva/Udapussellawa region', 'text'),
    'uva_udapussellawa_weather_score':   ('01_sales_index', 'Weather severity 1=sunny … 5=heavy rain', 'integer'),
    'low_grown_weather_desc':        ('01_sales_index', 'Raw weather text for Low Grown region', 'text'),
    'low_grown_weather_score':       ('01_sales_index', 'Weather severity 1=sunny … 5=heavy rain', 'integer'),
    'avg_weather_severity':          ('01_sales_index', 'Mean of the three regional weather scores', 'float'),
    'crop_notes':                    ('01_sales_index', 'Crop intake summary text', 'text'),
    'crop_nuwara_eliya_trend':       ('01_sales_index', 'Crop intake trend: +1 increase / 0 maintained / -1 decrease', 'integer'),
    'crop_western_trend':            ('01_sales_index', 'Crop intake trend for Western region', 'integer'),
    'crop_uva_trend':                ('01_sales_index', 'Crop intake trend for Uva region', 'integer'),
    'crop_low_grown_trend':          ('01_sales_index', 'Crop intake trend for Low Grown region', 'integer'),
    'sl_production_mkgs':            ('01_sales_index', 'Sri Lanka monthly tea production (M/Kgs)', 'float'),
    'sl_production_yoy_variance':    ('01_sales_index', 'Year-on-year production variance M/Kgs (negative = decline)', 'float'),
    'prod_high_2026':                ('01_sales_index', 'High grown production Jan 2026 (M/Kgs)', 'float'),
    'prod_high_2025':                ('01_sales_index', 'High grown production Jan 2025 (M/Kgs)', 'float'),
    'prod_medium_2026':              ('01_sales_index', 'Medium grown production Jan 2026 (M/Kgs)', 'float'),
    'prod_medium_2025':              ('01_sales_index', 'Medium grown production Jan 2025 (M/Kgs)', 'float'),
    'prod_low_2026':                 ('01_sales_index', 'Low grown production Jan 2026 (M/Kgs)', 'float'),
    'prod_low_2025':                 ('01_sales_index', 'Low grown production Jan 2025 (M/Kgs)', 'float'),
    'public_auction_weekly_2026':    ('01_sales_index', 'Public auction volume sold this week 2026 (kgs)', 'integer'),
    'public_auction_weekly_2025':    ('01_sales_index', 'Public auction volume sold this week 2025 (kgs)', 'integer'),
    'public_auction_todate_2026':    ('01_sales_index', 'Public auction year-to-date volume 2026 (kgs)', 'integer'),
    'public_auction_todate_2025':    ('01_sales_index', 'Public auction year-to-date volume 2025 (kgs)', 'integer'),
    'private_sales_weekly_2026':     ('01_sales_index', 'Private sales volume this week 2026 (kgs)', 'integer'),
    'private_sales_weekly_2025':     ('01_sales_index', 'Private sales volume this week 2025 (kgs)', 'integer'),
    'total_sold_weekly_2026':        ('01_sales_index', 'Total all-channel volume sold this week 2026 (kgs)', 'integer'),
    'total_sold_weekly_2025':        ('01_sales_index', 'Total all-channel volume sold this week 2025 (kgs)', 'integer'),
    'fx_usd_2026':                   ('01_sales_index', 'LKR per USD (current year)', 'float'),
    'fx_usd_2025':                   ('01_sales_index', 'LKR per USD (prior year)', 'float'),
    'fx_usd_2024':                   ('01_sales_index', 'LKR per USD (two years prior)', 'float'),
    'fx_gbp_2026':                   ('01_sales_index', 'LKR per GBP (current year)', 'float'),
    'fx_eur_2026':                   ('01_sales_index', 'LKR per EUR (current year)', 'float'),
    'fx_jpy_2026':                   ('01_sales_index', 'LKR per JPY (current year)', 'float'),
    # ── 02_auction_offerings ────────────────────────────────────────────────
    'category':                      ('02_auction_offerings', 'Tea category: ex_estate / high_medium / lg_leafy / lg_semi_leafy / lg_tippy / premium_flowery / off_grades / dust / total', 'text'),
    'qty_mkgs':                      ('02_auction_offerings', 'Quantity on offer in Million Kilograms', 'float'),
    'demand_label':                  ('02_auction_offerings', 'Demand description: good / fair / fair_general / less / poor', 'text'),
    'demand_score':                  ('02_auction_offerings', 'Demand encoded numerically: good=4 fair=3 less=2 poor=1', 'integer'),
    'lots':                          ('02_auction_offerings', 'Number of lots in this category', 'integer'),
    'kgs':                           ('02_auction_offerings', 'Kilograms offered in this category', 'integer'),
    # ── 03_quantity_sold ────────────────────────────────────────────────────
    'forward_contracts_weekly_2026': ('03_quantity_sold', 'Forward contract volume this week 2026 (kgs)', 'integer'),
    'forward_contracts_todate_2026': ('03_quantity_sold', 'Forward contract year-to-date volume 2026 (kgs)', 'integer'),
    # ── 04_high_grown_prices ────────────────────────────────────────────────
    'elevation':                     ('04_high_grown_prices', 'high_grown / low_grown / off_grade / dust', 'text'),
    'segment':                       ('04_high_grown_prices', 'Price segment: best_western / below_best_western / plainer_western / nuwara_eliya / brighter_udapussellawa / other_udapussellawa / best_uva / other_uva', 'text'),
    'grade':                         ('04_high_grown_prices', 'Tea grade: bop / bopf / pekoe_fbop / op  (HG) or fbop1 / fbop / bop1 / bopf / bop / fbopf1 / fbopf / fbopf_tippy / op1 / opa / op / pekoe / pek1 (LG)', 'text'),
    'price_lo_lkr':                  ('04_high_grown_prices', 'Lower bound of quoted price range (LKR per kg)', 'float'),
    'price_hi_lkr':                  ('04_high_grown_prices', 'Upper bound of quoted price range (LKR per kg). NULL if open-ended.', 'float'),
    # ── 05_low_grown_prices ─────────────────────────────────────────────────
    'tier':                          ('05_low_grown_prices', 'Quality tier: select_best / best / below_best / others', 'text'),
    # ── 06_offgrade_dust_prices ─────────────────────────────────────────────
    'category_type':                 ('06_offgrade_dust_prices', 'off_grade or dust', 'text'),
    # ── 07_top_prices ───────────────────────────────────────────────────────
    'estate':                        ('07_top_prices', 'Name of the tea estate', 'text'),
    'fw_sold':                       ('07_top_prices', '1 if the lot was brokered by Forbes & Walker, else 0', 'integer'),
    'price_lkr':                     ('07_top_prices', 'Top price achieved (LKR per kg)', 'integer'),
    'region':                        ('07_top_prices', 'Region header this record appeared under in the PDF', 'text'),
}

def build_column_dictionary(dataframes: dict) -> pd.DataFrame:
    """
    08_column_dictionary.csv
    One row per (table × column).
    Scans actual DataFrames so every column that was produced is captured.
    """
    rows = []
    for table_name, df in dataframes.items():
        for col in df.columns:
            defn = COLUMN_DEFINITIONS.get(col, (table_name, '', ''))
            rows.append({
                'table':       table_name,
                'column':      col,
                'description': defn[1] if len(defn) > 1 else '',
                'data_type':   defn[2] if len(defn) > 2 else '',
                'example_value': str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else '',
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(pdf_paths, output_dir='.'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Extract all PDFs ──
    extractions = []
    for path in pdf_paths:
        try:
            extractions.append(extract_pdf(path))
        except Exception as e:
            print(f"  ERROR: {path}: {e}")
            import traceback; traceback.print_exc()

    if not extractions:
        print("No data extracted.")
        return

    # ── Build tables ──
    tables = {
        '01_sales_index':          build_sales_index(extractions),
        '02_auction_offerings':    build_auction_offerings(extractions),
        '03_quantity_sold':        build_quantity_sold(extractions),
        '04_high_grown_prices':    build_hg_prices(extractions),
        '05_low_grown_prices':     build_lg_prices(extractions),
        '06_offgrade_dust_prices': build_offgrade_dust_prices(extractions),
        '07_top_prices':           build_top_prices(extractions),
    }
    
    # ── Run weather pipeline ──
    pdf_dir = Path(pdf_paths[0]).parent if pdf_paths else Path('.')
    weather_output_path = output_dir / '09_weather_features.csv'
    print("\n  Generating weather features from PDFs...")
    run_pipeline_weather(str(pdf_dir), str(weather_output_path))
    tables['09_weather_features'] = pd.read_csv(weather_output_path, low_memory=False)
    
    tables['08_column_dictionary'] = build_column_dictionary(tables)

    # ── Write CSVs (append + deduplicate if files already exist) ──
    dedup_keys = {
        '01_sales_index':          ['sale_id'],
        '02_auction_offerings':    ['sale_id', 'category'],
        '03_quantity_sold':        ['sale_id'],
        '04_high_grown_prices':    ['sale_id', 'segment', 'grade'],
        '05_low_grown_prices':     ['sale_id', 'grade', 'tier'],
        '06_offgrade_dust_prices': ['sale_id', 'category_type', 'category', 'elevation'],
        '07_top_prices':           ['sale_id', 'estate', 'grade'],
        '08_column_dictionary':    ['table', 'column'],
        '09_weather_features':     ['sale_id', 'region'],
    }

    print(f"\n{'─'*55}")
    print(f"  OUTPUT DIRECTORY: {output_dir.resolve()}")
    print(f"{'─'*55}")
    for name, df_new in tables.items():
        fpath = output_dir / f"{name}.csv"
        if fpath.exists():
            df_old = pd.read_csv(fpath, low_memory=False)
            df_out = pd.concat([df_old, df_new], ignore_index=True, sort=False)
        else:
            df_out = df_new
        # Deduplicate
        keys = [k for k in dedup_keys.get(name, []) if k in df_out.columns]
        if keys:
            df_out = df_out.drop_duplicates(subset=keys, keep='last')
        df_out.to_csv(fpath, index=False)
        print(f"  {name}.csv  →  {df_out.shape[0]:5d} rows × {df_out.shape[1]:3d} cols")

    print(f"\n  Processed {len(extractions)} PDF(s) → 9 CSV files written.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        paths = [Path(__file__).parent.parent.parent / 'data' / 'raw']

    pdf_files = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            pdf_files.extend(sorted(pp.glob('*.pdf')))
        elif pp.suffix.lower() == '.pdf':
            pdf_files.append(pp)

    if not pdf_files:
        print("No PDF files found.")
        sys.exit(1)

    print(f"\nForbes & Walker Tea Pipeline  –  {len(pdf_files)} PDF(s) found\n")
    run_pipeline(pdf_files, output_dir=Path(__file__).parent.parent.parent / 'data' / 'interim')
