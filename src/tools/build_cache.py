import re, gc
from pathlib import Path
import pandas as pd
import numpy as np
import json


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
CACHE    = Path(__file__).resolve().parents[1] / ".cache" / "long.parquet"
RAW_DIR  = DATA_DIR
MANIFEST = CACHE.with_suffix(".manifest.json")

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
