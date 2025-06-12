import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
DB_PATH = Path(__file__).parent.parent / "data" / "car_market.db"

def load_raw():
    all_dfs = []
    for file in RAW_DIR.glob("*.xlsx"):
        sheets = pd.read_excel(file, sheet_name=None, header=1)
        for name, df in sheets.items():
            df.columns = (
                df.columns
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .str.replace(" ", "_")
                    .str.replace("/", "_")
    )
            df["source_file"] = file.name
            df["sheet_name"]  = name
            all_dfs.append(df)


    combined = pd.concat(all_dfs, ignore_index=True).fillna(0)
    import re
    
    date_cols = [c for c in combined.columns if re.match(r"^\d{6}$", c)]
    
    id_cols   = [c for c in combined.columns if c not in date_cols]
    
    combined = (
        combined
          .melt(
             id_vars=id_cols,
             value_vars=date_cols,
             var_name="yyyymm",
             value_name="sales"
          )
    )
    
    combined["year"]  = combined["yyyymm"].str[:4].astype(int)
    combined["month"] = combined["yyyymm"].str[4:].astype(int)
    
    combined = combined.rename(columns={"sales": "total_sales"})
    return combined

def save_to_sql(df):
    engine = create_engine(f"sqlite:///{DB_PATH}")
    df.to_sql("sales", engine, if_exists="replace", index=False)

if __name__ == "__main__":
    df = load_raw()
    save_to_sql(df)
    print(f"Loaded {len(df)} rows into {DB_PATH}")
