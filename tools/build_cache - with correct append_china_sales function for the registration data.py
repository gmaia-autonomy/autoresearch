# src/build_cache.py

import re, gc
from pathlib import Path
import pandas as pd
import numpy as np
import json

# NEW: import the China sales loader you created
from build_china_long import load_china_sales_long

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
CACHE    = Path(__file__).resolve().parents[1] / ".cache" / "long.parquet"
RAW_DIR  = DATA_DIR
MANIFEST = CACHE.with_suffix(".manifest.json")

# NEW: where the China Registration + optional extra mapping live
CHINA_REG_PATH = DATA_DIR / "China Auto Registration.xlsx"
CHINA_EXTRA_MAP = DATA_DIR / "unmapped_names_new.xlsx"  # ok if missing

def _parse_years(path: Path):
    """
    Supports 'Autos_23_seg.xlsx' and 'Autos_23_25_seg.xlsx' (case-insensitive,
    handles 'Auto' vs 'Autos'). Returns (start, end) as two-digit ints,
    e.g. (23, 25) or (23, 23). Unrecognized names sort first.
    """
    m = re.search(r'autos?_(\d{2})(?:_(\d{2}))?_seg\.xlsx$', path.name, flags=re.IGNORECASE)
    if not m:
        return (-1, -1)
    a = int(m.group(1))
    b = int(m.group(2)) if m.group(2) else a
    return (a, b)

def simplify_powertrain(p):
    if p is None or (isinstance(p,float) and pd.isna(p)): return "ICE"
    s = str(p).strip().upper()
    if s in {"", "-", "—", "N/A","NA","N.A.","N ⁄ A","N / A","NAN"}: return "ICE"
    if s == "EV": return "BEV"
    if s in {"ICE","HV","HV/EV","MILD HV","HV/EV/PHV","HV/PHV","48V MILD HV","HV/MHV","MHV","ICE/EV","MHV/PHV"}: return "ICE"
    if s in {"FCV","EV/FCV/PHV"}: return "FCEV"
    if s in {"PHV","EV/PHV"}: return "PHEV"
    return s

def process_file(path: Path):
    xls = pd.ExcelFile(path)
    chunks = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, header=1)
        if df is None or df.empty:
            continue
        df.columns = (df.columns.astype(str).str.strip().str.lower()
                      .str.replace(" ", "_").str.replace("/", "_"))
        date_cols = [c for c in df.columns if re.fullmatch(r"\d{6}", c)]
        base_cols = [c for c in ["group","maker_brand","country","powertrain"] if c in df.columns]
        if not date_cols:
            continue
        df = df[base_cols + date_cols].copy()
        if "powertrain" in df.columns:
            df["powertrain_simplified"] = df["powertrain"].map(simplify_powertrain)
        else:
            df["powertrain_simplified"] = np.nan
        id_cols = [c for c in df.columns if c not in date_cols]
        long = df.melt(id_vars=id_cols, value_vars=date_cols,
                       var_name="yyyymm", value_name="total_sales")
        long["total_sales"] = (long["total_sales"].astype(str).str.strip()
                               .replace(r"^-+$","0", regex=True)
                               .str.replace(r"[^\d\.-]", "", regex=True)
                               .replace("", "0").astype(float))
        long["year"]  = long["yyyymm"].astype(str).str[:4].astype(int)
        long["month"] = long["yyyymm"].astype(str).str[4:].astype(int)
        long["month_dt"] = pd.to_datetime(long["yyyymm"].astype(str), format="%Y%m")
        long["month_label"] = long["month_dt"].dt.strftime("%m/%Y")
        for col in ["group","maker_brand","country"]:
            if col in long: long[col] = long[col].astype(str)
        chunks.append(long)
        del df, long; gc.collect()
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)

def _append_china_sales(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove China rows from Autos_*_Seg and append China SALES from the
    Registration workbook (via build_china_long.load_china_sales_long).
    """
    # 1) Drop any existing China rows coming from Autos_*_Seg
    if "country" in long_df.columns:
        not_china = long_df["country"].str.strip().str.lower() != "china"
        long_df = long_df.loc[not_china].copy()

    # 2) Load China sales (Registration workbook) and harmonize columns
    extra_map_arg = str(CHINA_EXTRA_MAP) if CHINA_EXTRA_MAP.exists() else None
    china_sales = load_china_sales_long(str(CHINA_REG_PATH), extra_mapping_path=extra_map_arg)
    # china_sales columns: Group, Maker/Brand, Type, Segment, Model, Powertrain, yyyymm, units (+ Region/Country)

    china_part = china_sales.rename(columns={
        "Group": "group",
        "Maker/Brand": "maker_brand",
        "Powertrain": "powertrain",
        "yyyymm": "yyyymm",
        "units": "total_sales",
        "Country": "country",
    }).copy()

    # Normalize fields used downstream
    china_part["powertrain_simplified"] = china_part["powertrain"].map(simplify_powertrain)
    china_part["yyyymm"] = china_part["yyyymm"].astype(int).astype(str)
    china_part["total_sales"] = pd.to_numeric(china_part["total_sales"], errors="coerce").fillna(0.0)

    china_part["year"]  = china_part["yyyymm"].str[:4].astype(int)
    china_part["month"] = china_part["yyyymm"].str[4:].astype(int)
    china_part["month_dt"] = pd.to_datetime(china_part["yyyymm"], format="%Y%m")
    china_part["month_label"] = china_part["month_dt"].dt.strftime("%m/%Y")

    # keep only columns that exist in main df, fill any missing
    for col in long_df.columns:
        if col not in china_part.columns:
            if col in ("group","maker_brand","country","powertrain","powertrain_simplified"):
                china_part[col] = None
            elif col in ("total_sales",):
                china_part[col] = 0.0
            else:
                china_part[col] = pd.NA

    china_part = china_part[long_df.columns]

def main():
    files = sorted(DATA_DIR.glob("*.xlsx"), key=_parse_years)  # lexical order
    if not files:
        raise FileNotFoundError(f"No .xlsx in {DATA_DIR}")

    all_parts = []
    for f in files:
        print("Reading:", f.name, flush=True)
        part = process_file(f)
        print("  rows:", len(part), flush=True)
        if not part.empty:
            all_parts.append(part)

    if not all_parts:
        raise ValueError("No usable sheets (no YYYYMM columns).")

    long = pd.concat(all_parts, ignore_index=True)

    # >>> NEW: swap in China SALES from Registration workbook
    try:
        long = _append_china_sales(long)
        print("China override applied from:", CHINA_REG_PATH)
    except Exception as e:
        print("WARNING: China override failed; keeping Autos_*_Seg China rows.", e)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(CACHE, index=False)
    print("Wrote cache:", CACHE, "rows:", len(long))

    # ---- write manifest (files currently in data/raw) ----
    snapshot = [
        {"name": f.name, "size": f.stat().st_size, "mtime": int(f.stat().st_mtime)}
        for f in files
    ]
    MANIFEST.write_text(json.dumps({"files": snapshot}, indent=2))
    print("Wrote manifest:", MANIFEST)

if __name__ == "__main__":
    main()