# build_china_long.py
# Produces long-form China SALES from "China Registrations Data.xlsx"
# Uses your 2025 mapping + unmapped_names_new.xlsx; drops totals; blanks in HV/PHV/EV -> ICE.

from pathlib import Path
import pandas as pd
import re

YEARS = range(2020, 2026)

def _pick_col(df, candidates):
    cols = {re.sub(r"\W+", "", str(c)).lower(): c for c in df.columns}
    for name in candidates:
        k = re.sub(r"\W+", "", name.lower())
        if k in cols:
            return cols[k]
    return None

def _is_total_text(s: str) -> bool:
    if not isinstance(s, str): return False
    t = s.strip().lower()
    return (
        "total" in t or "grand total" in t or "subtotal" in t
        or t.startswith("total") or t.endswith(" total")
    )

def _normalize_months(df, year):
    month_order = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
    ren = {}
    for col in df.columns:
        s = str(col).strip()
        if s.lower() in month_order:
            idx = month_order.index(s.lower()) + 1
            ren[col] = f"{year}{idx:02d}"
        elif re.fullmatch(r"\d{6}", s):
            ren[col] = s
    out_cols = sorted(set(ren.values()))
    return ren, out_cols

def _build_maps(xls, extra_map_path: Path|None):
    df_2025 = xls.parse("2025", header=3)

    c_company = _pick_col(df_2025, ["Company"])
    c_brand   = _pick_col(df_2025, ["Maker/Brand"])
    c_new_company = _pick_col(df_2025, ["NEW Company Name","New Company Name","New Company"])
    c_new_brand   = _pick_col(df_2025, ["NEW Maker/Brand Name","New Maker/Brand Name","NEW Brand Name"])

    company_map, brand_map, brand_to_newcompany = {}, {}, {}

    if c_company and c_new_company:
        tmp = (df_2025[[c_company, c_new_company]]
               .dropna(subset=[c_company]).astype(str).apply(lambda s: s.str.strip()))
        tmp = tmp[tmp[c_new_company] != ""].drop_duplicates(subset=[c_company])
        company_map = dict(tmp.values)

    if c_brand and c_new_brand:
        tmp = (df_2025[[c_brand, c_new_brand]]
               .dropna(subset=[c_brand]).astype(str).apply(lambda s: s.str.strip()))
        tmp = tmp[tmp[c_new_brand] != ""].drop_duplicates(subset=[c_brand])
        brand_map = dict(tmp.values)

    # also map brand -> company (needed for 2020/2021 with no Company col)
    if c_brand and c_new_company:
        tmp = (df_2025[[c_brand, c_new_company]]
               .dropna(subset=[c_brand]).astype(str).apply(lambda s: s.str.strip()))
        tmp = tmp[tmp[c_new_company] != ""].drop_duplicates(subset=[c_brand])
        brand_to_newcompany = dict(tmp.values)

    # merge extra mapping if present
    if extra_map_path and extra_map_path.exists():
        mC = pd.read_excel(extra_map_path, sheet_name="Unmapped Companies")
        mB = pd.read_excel(extra_map_path, sheet_name="Unmapped MakerBrands")
        if {"Old Company Name","NEW Company Name"} <= set(mC.columns):
            add = (mC[["Old Company Name","NEW Company Name"]]
                   .dropna().astype(str).apply(lambda s: s.str.strip()))
            company_map.update(dict(zip(add["Old Company Name"], add["NEW Company Name"])))
        if {"Old Maker/Brand Name","NEW Maker/Brand Name"} <= set(mB.columns):
            add = (mB[["Old Maker/Brand Name","NEW Maker/Brand Name"]]
                   .dropna().astype(str).apply(lambda s: s.str.strip()))
            brand_map.update(dict(zip(add["Old Maker/Brand Name"], add["NEW Maker/Brand Name"])))

    # hard fix typo everywhere
    def _fix(s): return "Small and Medium OEM" if s == "Small Medium OEM" else s
    company_map = {k: _fix(v) for k,v in company_map.items()}
    brand_map   = {k: _fix(v) for k,v in brand_map.items()}
    brand_to_newcompany = {k: _fix(v) for k,v in brand_to_newcompany.items()}

    return company_map, brand_map, brand_to_newcompany

def load_china_sales_long(
    registrations_path: str|Path,
    extra_mapping_path: str|Path|None = None,
) -> pd.DataFrame:
    """
    Returns long-form China SALES dataframe with columns:
      ['Region','Country','Group','Maker/Brand','Type','Segment','Model','Powertrain','yyyymm','units']
    """
    registrations_path = Path(registrations_path)
    xls = pd.ExcelFile(registrations_path)
    company_map, brand_map, brand_to_newcompany = _build_maps(xls, Path(extra_mapping_path) if extra_mapping_path else None)

    frames = []
    for year in YEARS:
        if str(year) not in xls.sheet_names:
            continue
        df = xls.parse(str(year), header=3)

        col_company = _pick_col(df, ["Company"])
        col_brand   = _pick_col(df, ["Maker/Brand"])
        col_type    = _pick_col(df, ["Type"])
        col_model   = _pick_col(df, ["Model"])
        col_pt      = _pick_col(df, ["HV/PHV/EV","Powertrain"])

        mrename, month_cols = _normalize_months(df, year)
        month_src = [c for c in df.columns if c in mrename]

        out = pd.DataFrame()

        # Group (Company) with mapping; fallback from brand for 2020/21
        if col_company:
            grp = df[col_company].astype(str).str.strip().map(lambda x: company_map.get(x, x))
        else:
            b = df[col_brand].astype(str).str.strip() if col_brand else ""
            grp = b.map(lambda x: brand_to_newcompany.get(x, company_map.get(x, ""))) if isinstance(b, pd.Series) else ""

        out["Group"] = grp
        out["Maker/Brand"] = df[col_brand].astype(str).str.strip().map(lambda x: brand_map.get(x, x)) if col_brand else ""
        out["Type"] = df[col_type] if col_type else ""
        out["Segment"] = 0
        out["Model"] = df[col_model] if col_model else ""
        if col_pt:
            pt = df[col_pt].astype(str).replace({"nan":"", "None":""}).str.strip()
            out["Powertrain"] = pt.replace("", "ICE")
        else:
            out["Powertrain"] = "ICE"

        # add month columns then melt
        for src in month_src:
            out[mrename[src]] = pd.to_numeric(df[src], errors="coerce").fillna(0).astype(int)

        # drop totals and blank labels
        mask_total = (
            out["Group"].apply(_is_total_text)
            | out["Maker/Brand"].apply(_is_total_text)
            | out["Model"].apply(_is_total_text)
        )
        nan_words = {"", "nan", "none", "null"}
        mask_blank = (
            out["Group"].astype(str).str.strip().str.lower().isin(nan_words) |
            out["Maker/Brand"].astype(str).str.strip().str.lower().isin(nan_words)
        )
        out = out[~(mask_total | mask_blank)].copy()

        # reshape to long
        long = out.melt(
            id_vars=["Group","Maker/Brand","Type","Segment","Model","Powertrain"],
            value_vars=month_cols,
            var_name="yyyymm",
            value_name="units",
        )
        long["yyyymm"] = long["yyyymm"].astype(int)
        long["Region"] = "China"
        long["Country"] = "China"
        frames.append(long)

    if not frames:
        return pd.DataFrame(columns=["Region","Country","Group","Maker/Brand","Type","Segment","Model","Powertrain","yyyymm","units"])

    china_long = pd.concat(frames, ignore_index=True)
    # ensure ints
    china_long["units"] = pd.to_numeric(china_long["units"], errors="coerce").fillna(0).astype(int)
    return china_long