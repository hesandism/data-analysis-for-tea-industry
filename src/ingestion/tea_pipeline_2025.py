import pdfplumber, pandas as pd, re, os, sys
from pathlib import Path
from datetime import datetime
from weather_pipeline import (
    REGIONS,
    add_lag_features,
    extract_dates_from_filename,
    fetch_weekly_weather,
    parse_region_weather_text,
    extract_pdf_text,
)

PTOK = r'(?:\d+(?:-{1,2}\d+)?(?:-)?|N/A)'

def clean(t): return re.sub(r'\s+',' ',t or '').strip()

def parse_price_range(raw):
    raw=str(raw).strip(); raw=re.sub(r'--','-',raw); raw=re.sub(r'\s+','',raw)
    if not raw or raw.upper() in ('N/A','NIL','-',''): return None,None
    m=re.match(r'^(\d+)-(\d+)$',raw)
    if m: return float(m.group(1)),float(m.group(2))
    m=re.match(r'^(\d+)-$',raw)
    if m: return float(m.group(1)),None
    m=re.match(r'^(\d+)$',raw)
    if m: v=float(m.group(1)); return v,v
    return None,None

def _etok(n, label_re, text):
    pat=re.compile(label_re+r'\s+('+r')\s+('.join([PTOK]*n)+r')', re.I|re.MULTILINE)
    m=pat.search(text); return list(m.groups()) if m else None

def weather_score(t):
    t=t.lower()
    if 'heavy rain' in t: return 5
    if 'rain' in t and 'bright' not in t: return 4
    if 'shower' in t: return 3
    if 'occasional shower' in t: return 2
    if 'bright' in t or 'sunny' in t: return 1
    return 3

def sentiment(t):
    t=t.lower()
    pos=['dearer','improved','appreciated','gained','firm','good demand','premium','strong','active']
    neg=['easier','bearish','declined','lower','irregular','less demand','withdrawn','withdrawal']
    p=sum(t.count(w) for w in pos); n=sum(t.count(w) for w in neg); tot=p+n
    return round((p-n)/tot,4) if tot else 0.0

def make_sale_id(n,y): return f"SALE_{y}_{int(n):02d}"

MONTHS='JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER'

def extract_header(full):
    meta={}
    m=re.search(r'SALENO[:\-]?\s*(\d+)',full,re.I) or re.search(r'SALE\s*NO\s*[:\-]?\s*(\d+)',full,re.I)
    meta['sale_number']=int(m.group(1)) if m else None
    m=re.search(rf'(\d+\w*(?:/\d+\w*)?)\s+({MONTHS})\s+(\d{{4}})',full,re.I)
    if m:
        meta['sale_date_raw']=f"{m.group(1)} {m.group(2).capitalize()} {m.group(3)}"
        meta['sale_year']=int(m.group(3)); meta['sale_month']=m.group(2).capitalize()
    else: meta['sale_date_raw']=meta['sale_year']=meta['sale_month']=None
    return meta

CMAP={'ExEstate':'ex_estate','High&Medium':'high_medium','Leafy':'leafy',
      'SemiLeafy':'semi_leafy','Tippy/SmallLeaf':'tippy','PremiumFlowery':'premium_flowery',
      'OffGrade':'off_grade','Dust':'dust','Total':'total'}
DMAP={'easier':'easier','fair':'fair','irregular':'irregular','less':'less',
      'bearish':'bearish','good':'good','firm':'firm'}
DNUM={'good':5,'firm':4,'fair':3,'easier':2,'irregular':2,'less':2,'bearish':1}

def extract_overall_market(pg):
    rows=[]
    pat=re.compile(r'(ExEstate|High&Medium|Leafy|SemiLeafy|Tippy/SmallLeaf|'
                   r'PremiumFlowery|OffGrade|Dust|Total)\s+([\d.]+)\s+([A-Za-z]+)',re.I)
    for m in pat.finditer(pg):
        cat=CMAP.get(m.group(1),m.group(1).lower()); qty=float(m.group(2))
        d=DMAP.get(m.group(3).lower(),m.group(3).lower())
        rows.append({'category':cat,'qty_mkgs':qty,'demand_label':d,'demand_score':DNUM.get(d,3)})
    return rows

LMAP={'ExEstate':'ex_estate','High&Medium':'high_medium','LowGrown-Leafy':'leafy',
      'LowGrown-SemiLeafy':'semi_leafy','LowGrown-Tippy':'tippy',
      'PremiumFlowery':'premium_flowery','OffGrades':'off_grade','Dust':'dust'}

def extract_auction_details(pg):
    totals={}
    m=re.search(r'([\d,]+)\s*LOTS\s*TOTALLING\s*([\d,]+)',pg,re.I)
    if m: totals['total_lots']=int(m.group(1).replace(',','')); totals['total_kgs']=int(m.group(2).replace(',',''))
    m=re.search(r'Re[-]?Prints?\s+([\d,]+)\s+([\d,]+)',pg,re.I)
    if m: totals['reprint_lots']=int(m.group(1).replace(',','')); totals['reprint_quantity']=int(m.group(2).replace(',',''))
    m=re.search(r'(\d{2}/\d{2}/\d{4})\s+(\d{2}/\d{2}/\d{4})\s+(\d{2}/\d{2}/\d{4})',pg)
    if m: totals['settlement_10pct']=m.group(1); totals['settlement_buyers']=m.group(2); totals['settlement_sellers']=m.group(3)
    cat_rows=[]
    cp=re.compile(r'(ExEstate|High&Medium|LowGrown-Leafy|LowGrown-SemiLeafy|LowGrown-Tippy|'
                  r'PremiumFlowery|OffGrades|Dust)\s+([\d,]+)\s+([\d,]+)',re.I)
    for m in cp.finditer(pg):
        cat_rows.append({'category':LMAP.get(m.group(1),m.group(1).lower()),
                         'lots':int(m.group(2).replace(',','')),
                         'kgs':int(m.group(3).replace(',',''))})
    return totals, cat_rows

def extract_commentary(full):
    m=re.search(r'COMMENTS\s*(.*?)(?:NOTE\b|WORLD\s*TEA|CROP\s*AND|$)',full,re.S|re.I)
    if not m: return {'commentary':'','sentiment_overall':0.0,'sentiment_ex_estate':0.0,'sentiment_low_grown':0.0}
    text=clean(m.group(1)); lg=text.lower().find('low grown')
    return {'commentary':text[:1500],'sentiment_overall':sentiment(text),
            'sentiment_ex_estate':sentiment(re.sub(r'(?i)low grown.*','',text)),
            'sentiment_low_grown':sentiment(text[lg:] if lg>=0 else '')}

def extract_weather(full):
    regs={'western_nuwara_eliya':(r'Western/Nuwara\s+Eliya\s+Regions?\s*(.*?)(?:Uva/Uda|Low\s*Grown|Crop\b|$)'),
          'uva_udapussellawa':(r'Uva/Uda\w*\s+Regions?\s*(.*?)(?:Low\s*Grown|Crop\b|$)'),
          'low_grown':(r'Low\s*Growns?\s*(.*?)(?:FORBES|CROP|$)')}
    res={}
    for reg,pat in regs.items():
        m=re.search(pat,full,re.S|re.I)
        if m:
            d=clean(m.group(1)); res[f'{reg}_weather_desc']=d[:250]; res[f'{reg}_weather_score']=weather_score(d)
    m=re.search(r'Crop\s*(.*?)(?:FORBES|HIGH\s*GROWN\s*TEAS|$)',full,re.S|re.I)
    crop=clean(m.group(1)) if m else ''; res['crop_notes']=crop[:300]
    sc=[v for k,v in res.items() if k.endswith('_score')]
    res['avg_weather_severity']=round(sum(sc)/len(sc),2) if sc else None
    cl=crop.lower(); all_r='all regions' in cl
    gt=(-1 if 'decrease' in cl else 1 if 'increase' in cl else 0) if all_r else None
    for reg in ['nuwara_eliya','western','uva','low_grown']:
        if gt is not None: res[f'crop_{reg}_trend']=gt
        elif reg.replace('_',' ') in cl:
            res[f'crop_{reg}_trend']=-1 if 'decrease' in cl else 1 if 'increase' in cl else 0
    return res

def build_weather_features(pdf_paths, extractions):
    rows = []
    extraction_by_name = {Path(item['source_file']).name: item for item in extractions}

    for pdf_path in pdf_paths:
        pdf_path = Path(pdf_path)
        extraction = extraction_by_name.get(pdf_path.name)
        if not extraction:
            continue

        dates = extract_dates_from_filename(pdf_path.name)
        if not dates:
            continue
        print(f"Extracted dates from filename '{pdf_path.name}': {dates}")
        auction_date = dates[0]
        sale_id = extraction['sale_id']
        text = extract_pdf_text(pdf_path)
        text_weather = parse_region_weather_text(text)

        for region_key, region_info in REGIONS.items():
            row = {
                'sale_id': sale_id,
                'auction_date': auction_date,
                'region': region_key,
                'region_label': region_info['label'],
                'lat': region_info['lat'],
                'lon': region_info['lon'],
                'pdf_source': pdf_path.name,
            }

            region_weather = text_weather.get(region_key, {})
            row.update({
                'text_condition_score': region_weather.get('condition_score'),
                'text_crop_change': region_weather.get('crop_change', 'unknown'),
                'text_has_rain': region_weather.get('has_rain'),
                'text_has_mist': region_weather.get('has_mist'),
                'text_has_bright': region_weather.get('has_bright'),
                'text_has_thunder': region_weather.get('has_thunder'),
                'text_keywords': region_weather.get('keywords_found', ''),
                'text_raw_summary': region_weather.get('raw_text', ''),
            })

            row.update(fetch_weekly_weather(region_info['lat'], region_info['lon'], auction_date))
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    lag_targets = [
        'precipitation_sum_total',
        'temperature_2m_mean_mean',
        'sunshine_duration_total',
    ]
    return add_lag_features(df, lag_targets, lags=[1, 2, 3])

def extract_quantity_sold(full):
    res={}
    for raw,key in [(r'PRIVATESALES','private_sales'),(r'PUBLICAUCTION','public_auction'),
                    (r'FORWARDCONTRACTS','forward_contracts'),(r'TOTAL(?!LOTS)','total_sold')]:
        m=re.search(raw+r'\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)',full,re.I)
        if m:
            res[f'{key}_weekly_2026']=int(m.group(1).replace(',',''))
            res[f'{key}_weekly_2025']=int(m.group(2).replace(',',''))
            res[f'{key}_todate_2026']=int(m.group(3).replace(',',''))
            res[f'{key}_todate_2025']=int(m.group(4).replace(',',''))
    for m in re.finditer(rf'(\d+\w*)\s*({MONTHS})\s*(\d{{4}})\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',full,re.I):
        lbl=f"{m.group(1).lower()}_{m.group(2).lower()}_{m.group(3)}"
        res[f'avg_qty_mkgs_{lbl}']=float(m.group(4)); res[f'avg_lkr_{lbl}']=float(m.group(7)); res[f'avg_usd_{lbl}']=float(m.group(10))
    for cur,key in [('USD','usd'),(r'STG\.PD','gbp'),('EURO','eur'),('YEN','jpy')]:
        m=re.search(cur+r'\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',full,re.I)
        if m: res[f'fx_{key}_2026']=float(m.group(1)); res[f'fx_{key}_2025']=float(m.group(2)); res[f'fx_{key}_2024']=float(m.group(3))
    return res

def extract_gross_averages(full):
    res={}
    for pat,key in [(r'UvaHighGrown','uva_high'),(r'WesternHighGrown','western_high'),(r'CTCHighGrown','ctc_high'),
                    (r'HighGrown\(Summary\)','high_summary'),(r'UvaMediumGrown','uva_medium'),
                    (r'WesternMediumGrown','western_medium'),(r'CTCMediumGrown','ctc_medium'),
                    (r'MediumGrown\(Summary\)','medium_summary'),(r'OrthodoxLowGrown','orthodox_low'),
                    (r'CTCLowGrown','ctc_low'),(r'LowGrown\(Summary\)','low_summary'),(r'Total(?!\s*LOTS)','total')]:
        m=re.search(pat+r'\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',full,re.I)
        if m:
            res[f'gross_lkr_weekly_{key}_2026']=float(m.group(1)); res[f'gross_lkr_weekly_{key}_2025']=float(m.group(2))
            res[f'gross_lkr_todate_{key}_2026']=float(m.group(4)); res[f'gross_lkr_todate_{key}_2025']=float(m.group(5))
    return res

HG_SEGS=[('BestWesterns','best_western'),('BelowBestWesterns','below_best_western'),
          ('PlainerWesterns','plainer_western'),('NuwaraEliyas','nuwara_eliya'),
          ('BrighterUdapussellawas','brighter_udapussellawa'),('OtherUdapussellawas','other_udapussellawa'),
          ('BestUvas','best_uva'),('OtherUvas','other_uva')]
HG_GRADES=['bop','bopf','pekoe_fbop','op']

def extract_hg_prices(txt):
    rows=[]
    for sr,seg in HG_SEGS:
        tok=_etok(8,sr,txt)
        if tok is None: continue
        for i,g in enumerate(HG_GRADES):
            lo,hi=parse_price_range(tok[1+i*2])
            rows.append({'elevation':'high_grown','segment':seg,'grade':g,'price_lo_lkr':lo,'price_hi_lkr':hi})
    return rows

MG_SEGS=[('GoodMediums','good_mediums'),('OtherMediums','other_mediums')]

def extract_mg_prices(txt):
    rows=[]
    for sr,seg in MG_SEGS:
        tok=_etok(8,sr,txt)
        if tok is None: continue
        for i,g in enumerate(HG_GRADES):
            lo,hi=parse_price_range(tok[1+i*2])
            rows.append({'elevation':'medium_grown','segment':seg,'grade':g,'price_lo_lkr':lo,'price_hi_lkr':hi})
    return rows

def extract_ctc_prices(txt):
    ms=re.search(r'UNORTHODOX\s*/\s*CTCTEAS(.*?)(?:OFFGRADES|DUSTS|$)',txt,re.S|re.I)
    if not ms: return []
    sec=ms.group(1); rows=[]
    for er,elev in [('HighGrown','high'),('MediumGrown','medium'),('LowGrown','low')]:
        tok=_etok(4,er,sec)
        if tok is None: continue
        for i,g in enumerate(['bp1','pf1']):
            lo,hi=parse_price_range(tok[1+i*2])
            rows.append({'elevation':elev,'segment':f'ctc_{elev}','grade':g,'price_lo_lkr':lo,'price_hi_lkr':hi})
    return rows

LG_GRADES=[
    (r'^FBOPF\(TIPPY\)/FBOPFSP','fbopf_tippy'),
    (r'^FBOPF1','fbopf1'),
    (r'^FBOPF(?!1|\()','fbopf'),
    (r'^FBOP1','fbop1'),
    (r'^FBOP(?!F|1)','fbop'),
    (r'^BOP1','bop1'),
    (r'^BOPF','bopf'),
    (r'^BOP(?!F|1)','bop'),
    (r'^OP1','op1'),
    (r'^OPA','opa'),
    (r'^OP(?!A|1)','op'),
    (r'^PEKOE','pekoe'),
    (r'^PEK1','pek1'),
]
LG_TIERS=['select_best','best','below_best','others']

def extract_lg_prices(txt):
    rows=[]
    for gr,grade in LG_GRADES:
        pat=re.compile(gr+r'\s+('+r')\s+('.join([PTOK]*8)+r')',re.I|re.MULTILINE)
        m=pat.search(txt)
        if not m: continue
        tok=list(m.groups())
        for i,tier in enumerate(LG_TIERS):
            lo,hi=parse_price_range(tok[1+i*2])
            rows.append({'elevation':'low_grown','grade':grade,'tier':tier,'price_lo_lkr':lo,'price_hi_lkr':hi})
    return rows

OG_ROWS=[
    (r'BetterFannings\(Orthodox\)','fannings_orthodox_better'),
    (r'BetterFannings\(CTC\)','fannings_ctc_better'),
    (r'OtherFannings\(Orthodox\)','fannings_orthodox_other'),
    (r'OtherFannings\(CTC\)','fannings_ctc_other'),
    (r'GoodBrokens','brokens_good'),
    (r'OtherBrokens','brokens_other'),
    (r'BetterBOP1As','bop1a_better'),
    (r'OtherBOP1As','bop1a_other'),
]
DUST_ROWS=[
    (r'BetterPrimaryDust\(Orthodox\)','primary_orth_better'),
    (r'BetterPrimaryDust\(CTC\)P\.Dust','primary_ctc_better'),
    (r'BelowBestPrimaryDust\(Orthodox\)','primary_orth_below_best'),
    (r'OtherPrimaryDust\(CTC\)P\.Dust','primary_ctc_other'),
    (r'OtherPrimaryDust\(Orthodox\)','primary_orth_other'),
    (r'BetterSecondaryDust','secondary_better'),
    (r'OtherSecondaryDust','secondary_other'),
]
ELEVS=['high','medium','low']

def extract_offgrade_dust_prices(txt):
    rows=[]
    for ct,rl in [('off_grade',OG_ROWS),('dust',DUST_ROWS)]:
        for rr,label in rl:
            tok=_etok(6,rr,txt)
            if tok is None: continue
            for i,elev in enumerate(ELEVS):
                lo,hi=parse_price_range(tok[1+i*2])
                rows.append({'category_type':ct,'category':label,'elevation':elev,'price_lo_lkr':lo,'price_hi_lkr':hi})
    return rows

TOP_REGIONS=['WESTERN MEDIUM','WESTERN HIGH','NUWARA ELIYAS','UDAPUSSELLAWAS',
             'UVA HIGH','UVA MEDIUM','LOW GROWNS','PREMIUM FLOWERY',
             'DUSTS','OFF GRADES','UNORTHODOX HIGH','UNORTHODOX MEDIUM','UNORTHODOX LOW']
TOP_PAT=re.compile(
    r'([A-Z][A-Za-z\s\-]+?)\s+'
    r'(FBOPF?ExSp\d?|FBOPFSp|FBOPFExSp\d?|BOPFSp|BOPSp|'
    r'FBOPF\d?(?:/FBOPF\d?)?|FBOP\d?|BOP\d?A?|BOPF\d?|'
    r'OP\d?A?|OPA|PEKOE|PEK\d?|FGS\d?|BM|BP\d?|'
    r'CTC\s*PF\d?|CTC\s*BP\d?|CTC\s*BPS|DUST\d?|PD)\s*'
    r'(@)?\s*([\d]{3,})')

def extract_top_prices(txt):
    recs=[]; cr='Unknown'
    for line in txt.splitlines():
        for rh in TOP_REGIONS:
            if rh.lower() in line.lower(): cr=rh; break
        m=TOP_PAT.search(line)
        if m: recs.append({'estate':clean(m.group(1)),'grade':m.group(2).replace(' ',''),
                           'fw_sold':1 if m.group(3)=='@' else 0,'price_lkr':int(m.group(4)),'region':cr})
    return recs

def extract_pdf(pdf_path):
    print(f"  Processing: {pdf_path}")
    with pdfplumber.open(pdf_path) as pdf:
        pages=[pg.extract_text() or '' for pg in pdf.pages]
    full='\n'.join(pages)
    page2=pages[1] if len(pages)>1 else full
    header=extract_header(full)
    sn,sy=header.get('sale_number'),header.get('sale_year')
    sid=make_sale_id(sn,sy) if sn and sy else f"UNKNOWN_{os.path.basename(str(pdf_path))}"
    market_rows=extract_overall_market(page2)
    totals,cat_rows=extract_auction_details(page2)
    commentary=extract_commentary(full)
    weather=extract_weather(full)
    qty_sold=extract_quantity_sold(full)
    gross_avg=extract_gross_averages(full)
    p5=pages[5] if len(pages)>5 else ''
    p6=pages[6] if len(pages)>6 else ''
    p7=pages[7] if len(pages)>7 else ''
    p8=pages[8] if len(pages)>8 else ''
    top_txt='\n'.join(pages[9:11])
    hg=extract_hg_prices(p5); mg=extract_mg_prices(p6); ctc=extract_ctc_prices(p6)
    og=extract_offgrade_dust_prices(p7); lg=extract_lg_prices(p8)
    tp=extract_top_prices(top_txt)
    return {'sale_id':sid,'header':header,'market_rows':market_rows,'totals':totals,
            'cat_rows':cat_rows,'commentary':commentary,'weather':weather,
            'qty_sold':qty_sold,'gross_avg':gross_avg,
            'hg_prices':hg+mg+ctc,'lg_prices':lg,'og_prices':og,'top_prices':tp,
            'source_file':os.path.basename(str(pdf_path)),
            'extracted_at':datetime.now().isoformat(timespec='seconds')}

def build_sales_index(ex):
    rows=[]
    for e in ex:
        r={'sale_id':e['sale_id'],'sale_number':e['header'].get('sale_number'),
           'sale_date_raw':e['header'].get('sale_date_raw'),'sale_year':e['header'].get('sale_year'),
           'sale_month':e['header'].get('sale_month'),'source_file':e['source_file'],'extracted_at':e['extracted_at']}
        r.update(e['totals']); r.update(e['commentary']); r.update(e['weather'])
        r.update(e['qty_sold']); r.update(e['gross_avg']); rows.append(r)
    return pd.DataFrame(rows)

def build_auction_offerings(ex):
    rows=[]
    for e in ex:
        ll={r['category']:r for r in e['cat_rows']}
        for mr in e['market_rows']:
            cat=mr['category']; li=ll.get(cat,{})
            rows.append({'sale_id':e['sale_id'],'category':cat,'qty_mkgs':mr.get('qty_mkgs'),
                         'demand_label':mr.get('demand_label'),'demand_score':mr.get('demand_score'),
                         'lots':li.get('lots'),'kgs':li.get('kgs')})
    return pd.DataFrame(rows)

def build_quantity_sold(ex):
    rows=[]
    for e in ex:
        r={'sale_id':e['sale_id'],'sale_number':e['header'].get('sale_number'),
           'sale_year':e['header'].get('sale_year'),'sale_month':e['header'].get('sale_month')}
        r.update(e['qty_sold']); rows.append(r)
    return pd.DataFrame(rows)

def build_hg_prices(ex):
    rows=[]
    for e in ex:
        for r in e['hg_prices']: rows.append({'sale_id':e['sale_id'],**r})
    return pd.DataFrame(rows)

def build_lg_prices(ex):
    rows=[]
    for e in ex:
        for r in e['lg_prices']: rows.append({'sale_id':e['sale_id'],**r})
    return pd.DataFrame(rows)

def build_offgrade_dust_prices(ex):
    rows=[]
    for e in ex:
        for r in e['og_prices']: rows.append({'sale_id':e['sale_id'],**r})
    return pd.DataFrame(rows)

def build_top_prices(ex):
    rows=[]
    for e in ex:
        for r in e['top_prices']: rows.append({'sale_id':e['sale_id'],**r})
    return pd.DataFrame(rows)

def build_column_dictionary(dfs):
    rows=[]
    for tn,df in dfs.items():
        for col in df.columns:
            rows.append({
                'table': tn,
                'column': col,
                'description': '',
                'data_type': '',
                'example_value': str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else ''
            })
    return pd.DataFrame(rows)

def run_pipeline(pdf_paths, output_dir='.'):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    ex=[]
    for path in pdf_paths:
        try: ex.append(extract_pdf(path))
        except Exception as e: import traceback; print(f"  ERROR: {path}: {e}"); traceback.print_exc()
    if not ex: print("No data extracted."); return
    tables={'01_sales_index':build_sales_index(ex),'02_auction_offerings':build_auction_offerings(ex),
            '03_quantity_sold':build_quantity_sold(ex),'04_high_grown_prices':build_hg_prices(ex),
            '05_low_grown_prices':build_lg_prices(ex),'06_offgrade_dust_prices':build_offgrade_dust_prices(ex),
            '07_top_prices':build_top_prices(ex)}
    weather_df = build_weather_features(pdf_paths, ex)
    if not weather_df.empty:
        tables['09_weather_features'] = weather_df
    tables['08_column_dictionary']=build_column_dictionary(tables)
    dedup={'01_sales_index':['sale_id'],'02_auction_offerings':['sale_id','category'],
           '03_quantity_sold':['sale_id'],'04_high_grown_prices':['sale_id','elevation','segment','grade'],
           '05_low_grown_prices':['sale_id','grade','tier'],
           '06_offgrade_dust_prices':['sale_id','category_type','category','elevation'],
           '07_top_prices':['sale_id','estate','grade'],'08_column_dictionary':['table','column'],
           '09_weather_features':['sale_id','region']}
    print(f"\n{'─'*55}\n  OUTPUT: {output_dir.resolve()}\n{'─'*55}")
    for name,df_new in tables.items():
        fpath=output_dir/f"{name}.csv"
        if fpath.exists():
            df_out=pd.concat([pd.read_csv(fpath,low_memory=False),df_new],ignore_index=True,sort=False)
        else: df_out=df_new
        ks=[k for k in dedup.get(name,[]) if k in df_out.columns]
        if ks: df_out=df_out.drop_duplicates(subset=ks,keep='last')
        df_out.to_csv(fpath,index=False)
        print(f"  {name}.csv  →  {df_out.shape[0]:5d} rows × {df_out.shape[1]:3d} cols")
    print(f"\n  Processed {len(ex)} PDF(s) → {len(tables)} CSV files.\n")

if __name__=='__main__':
    root = Path(__file__).resolve().parents[2]
    # paths=sys.argv[1:] if len(sys.argv)>1 else [Path(__file__).parent/'data'/'raw']
    paths = [root/'data'/'raw']
    print(paths)
    pdf_files=[]
    for p in paths:
        pp=Path(p)
        if pp.is_dir(): pdf_files.extend(sorted(pp.glob('*.pdf')))
        elif pp.suffix.lower()=='.pdf': pdf_files.append(pp)
    if not pdf_files: print("No PDF files found."); sys.exit(1)
    print(f"\nForbes & Walker Tea Pipeline (FW Edition) – {len(pdf_files)} PDF(s)\n")
    run_pipeline(pdf_files, output_dir=root/'data'/'Interim'/'interim_combined')