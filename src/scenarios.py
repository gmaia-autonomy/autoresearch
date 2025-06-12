import pandas as pd
import numpy as np

def apply_ev_policy(df: pd.DataFrame,
                    country: str,
                    mandate_year: int,
                    adoption_pct: float) -> pd.DataFrame:
    df2 = df.copy()
    df2['total_sales'] = pd.to_numeric(df2['total_sales'],
                                       errors='coerce')\
                          .fillna(0).astype(float)

    mask = (
        df2['country'].str.lower() == country.lower()
    ) & (
        df2['year'] >= mandate_year
    )

    df2['ev_sales']  = np.where(mask,
                                df2['total_sales'] * adoption_pct,
                                0.0)
    df2['ice_sales'] = df2['total_sales'] - df2['ev_sales']
    return df2
