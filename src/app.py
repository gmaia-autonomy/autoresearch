# # File: src/app.py

# import streamlit as st
# import pandas as pd
# import numpy as np
# import re
# from pathlib import Path

# @st.cache_data
# def load_and_prepare():
#     # 1) Load all Excel sheets
#     RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
#     all_dfs = []
#     for file in RAW_DIR.glob("*.xlsx"):
#         sheets = pd.read_excel(file, sheet_name=None, header=1)
#         for name, df in sheets.items():
#             # normalize headers
#             df.columns = (
#                 df.columns
#                   .astype(str)
#                   .str.strip()
#                   .str.lower()
#                   .str.replace(" ", "_")
#                   .str.replace("/", "_")
#             )
#             df["source_file"] = file.name
#             df["sheet_name"]  = name
#             all_dfs.append(df)

#     # 2) Combine & fill
#     combined = pd.concat(all_dfs, ignore_index=True).fillna(0)

#     # 3) Pivot monthly columns into long form
#     date_cols = [c for c in combined.columns if re.match(r"^\d{6}$", c)]
#     id_cols   = [c for c in combined.columns if c not in date_cols]

#     long = combined.melt(
#         id_vars=id_cols,
#         value_vars=date_cols,
#         var_name="yyyymm",
#         value_name="total_sales"
#     )

#     # 4) Clean and convert total_sales to float, treating dashes as zero
#     long["total_sales"] = (
#         long["total_sales"]
#           .astype(str)
#           .str.strip()
#           .replace(r"^-+$", "0", regex=True)           # dash-only cells → "0"
#           .str.replace(r"[^\d\.-]", "", regex=True)     # drop non-numeric chars
#           .replace("", "0")                             # empty → "0"
#           .astype(float)
#     )

#     # 5) Extract year and month
#     long["year"]  = long["yyyymm"].str[:4].astype(int)
#     long["month"] = long["yyyymm"].str[4:].astype(int)

#     return long

# def apply_ev_policy(df, country, mandate_year, adoption_pct):
#     ts = df["total_sales"]
#     mask = (
#         df["country"].str.lower() == country.lower()
#     ) & (
#         df["year"] >= mandate_year
#     )
#     ev = np.where(mask, ts * adoption_pct, 0.0)
#     ice = ts - ev
#     return df.assign(ev_sales=ev, ice_sales=ice)

# # ——— Streamlit UI ———————————————————————————————————

# df = load_and_prepare()
# st.title("🚗 Global Car-Market Explorer")

# oem     = st.sidebar.selectbox("OEM / Brand", sorted(df["maker_brand"].unique()))
# country = st.sidebar.selectbox("Country",      sorted(df["country"].unique()))
# year    = st.sidebar.slider("Year",
#                             int(df.year.min()),
#                             int(df.year.max()),
#                             int(df.year.max()))
# ev_pct  = st.sidebar.slider("EV adoption (%)", 0, 100, 0) / 100.0

# sub = df[(df.maker_brand == oem) &
#          (df.country     == country) &
#          (df.year        == year)]

# if sub.empty:
#     st.warning("No data for that selection.")
# else:
#     base    = sub["total_sales"].sum()
#     sub_adj = apply_ev_policy(sub.copy(), country, year, ev_pct)
#     adj     = sub_adj["ev_sales"].sum() + sub_adj["ice_sales"].sum()

#     col1, col2 = st.columns(2)
#     col1.metric("Base Total Sales",     int(base))
#     col2.metric("Adjusted Total Sales", int(adj))

#     st.dataframe(sub_adj[[
#         "maker_brand", "country", "year",
#         "ice_sales", "ev_sales", "total_sales"
#     ]])

#     trend = sub_adj.groupby("month")[["ice_sales","ev_sales"]].sum()
#     st.line_chart(trend)

#     ms = trend.div(trend.sum(axis=1), axis=0).iloc[-1]
#     st.bar_chart(ms)

#     st.download_button(
#         "Download adjusted CSV",
#         sub_adj.to_csv(index=False),
#         "adjusted.csv"
#     )

# File: src/app.py

import streamlit as st
# 1) Set Streamlit’s page configuration BEFORE any other Streamlit calls:
st.set_page_config(
    page_title=" Auto Research ",
       page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2) Inject custom CSS for a Bloomberg‐like dark interface
st.markdown(
    """
    <style>
    /*──────────────────────────────────────────────────────────────*/
    /* 1) Load a clean, modern font: “Space Grotesk” */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

    /* 2) Apply that font and set a dark gradient background */
    html, body, .block-container, .sidebar .sidebar-content {
      font-family: 'Space Grotesk', sans-serif !important;
      background: linear-gradient(135deg, #0A0F1C 0%, #1A2332 100%) !important;
      color: #E0E0E0 !important;
    }

    /*──────────────────────────────────────────────────────────────*/
    /* 3) Glass-morphism panels for metrics and charts */
    /*    - Semi-transparent white overlay + backdrop blur */
    [data-testid="stMetric"] {
      background: rgba(255, 255, 255, 0.08) !important;
      backdrop-filter: blur(8px) !important;
      border-radius: 10px !important;
      box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
      padding: 1rem !important;
    }
    [data-testid="stLineChart"] svg,
    [data-testid="stPlotlyChart"] svg,
    [data-testid="stGraphViz"] svg,
    [data-testid="stAgGrid"] .ag-root-wrapper {
      background: rgba(255, 255, 255, 0.05) !important;
      border-radius: 10px !important;
      box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
    }

    /*──────────────────────────────────────────────────────────────*/
    /* 4) Sidebar styling: glass look + small accent on focus */
    .sidebar .sidebar-content {
      background: rgba(255, 255, 255, 0.06) !important;
      backdrop-filter: blur(8px) !important;
      border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
      padding: 1rem !important;
    }
    .sidebar .stSelectbox > label,
    .sidebar .stRadio > label,
    .sidebar .stButton > button {
      color: #E0E0E0 !important;
      font-weight: 500 !important;
    }

    /*──────────────────────────────────────────────────────────────*/
    /* 5) Inputs / dropdowns / radio buttons: glass look */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div[role="combobox"],
    .stRadio > label {
      background: rgba(255, 255, 255, 0.06) !important;
      backdrop-filter: blur(6px) !important;
      border: 1px solid rgba(255, 255, 255, 0.1) !important;
      border-radius: 8px !important;
      color: #E0E0E0 !important;
    }

    /* Focus/Hover: soft teal accent (#00bfae) */
    :root {
      --accent-color: #00bfae;
    }
    .stButton > button:hover,
    .stSelectbox > div > div > div[role="combobox"]:hover,
    .stTextInput > div > div > input:hover {
      border: 1px solid var(--accent-color) !important;
      box-shadow: 0 0 6px rgba(0, 191, 174, 0.4) !important;
    }

    /* When a dropdown is open (aria-expanded="true"), show accent */
    .stSelectbox > div > div > div[aria-expanded="true"] {
      border: 1px solid var(--accent-color) !important;
      box-shadow: 0 0 8px rgba(0, 191, 174, 0.6) !important;
    }

    /*──────────────────────────────────────────────────────────────*/
    /* 6) Headings: simple white, slightly bold */
    h1, h2, h3, h4, h5, h6 {
      color: #FFFFFF !important;
      font-weight: 600 !important;
      letter-spacing: 0.4px !important;
      margin-bottom: 0.5rem !important;
    }

    /*──────────────────────────────────────────────────────────────*/
    /* 7) Style tabs underline for active tab (generic selector) */
    [role="tablist"] [role="tab"]:focus,
    [role="tablist"] [aria-selected="true"] {
      border-bottom: 2px solid var(--accent-color) !important;
    }

    /*──────────────────────────────────────────────────────────────*/
    /* 8) Hide Streamlit’s default header & footer for a clean look */
    header, footer, .css-18e3th9 {
      visibility: hidden !important;
    }
    /*──────────────────────────────────────────────────────────────*/
    </style>
    """,
    unsafe_allow_html=True
)

import pandas as pd
import numpy as np
import re
from pathlib import Path
import plotly.express as px
import io
import plotly.graph_objects as go


@st.cache_data


def plot_consolidated_sales_by_region(df_filtered: pd.DataFrame, region_map: dict, title: str):
    """
    Given df_filtered (only rows for a single OEM or a single Brand),
    build a stacked‐bar “Unit Sales by Region” chart using Plotly.
    """

    # 1) Build reverse lookup: country → region
    country_to_region = {}
    for region_name, countries in region_map.items():
        for c in countries:
            country_to_region[c] = region_name

    # 2) Copy DataFrame & map each row’s country → region
    df = df_filtered.copy()
    df["region"] = df["country"].map(country_to_region)
    df = df.dropna(subset=["region"])  # drop any row whose country isn’t in region_map

    # 3) Aggregate total_sales by (yyyymm, region)
    agg = (
        df.groupby(["yyyymm", "region"])["total_sales"]
          .sum()
          .reset_index(name="region_sales")
    )

    # 4) Pivot so that each region becomes its own column
    pivot = agg.pivot(index="yyyymm", columns="region", values="region_sales").fillna(0)

    # 5) Convert “yyyymm” (string like “202301”) → datetime index
    pivot = pivot.reset_index()
    pivot["month_year"] = pd.to_datetime(pivot["yyyymm"], format="%Y%m")
    pivot = pivot.set_index("month_year").drop(columns=["yyyymm"])

    # 6) Trim trailing rows where all regions = 0
    nonzero_mask = pivot.sum(axis=1) > 0
    if nonzero_mask.any():
        last_date = pivot.index[nonzero_mask][-1]
        pivot = pivot.loc[:last_date]

    # 7) Build a Plotly stacked‐bar figure (dark template)
    fig = px.bar(
        pivot,
        x=pivot.index,
        y=pivot.columns.tolist(),
        labels={"value": "Unit Sales", "month_year": "Month"},
        title=title,
        template="plotly_dark"
    )

    # 8) Adjust layout: stacked bars, rotate x‐labels, format y‐axis, legend on top
    fig.update_layout(
        barmode="stack",
        xaxis=dict(
            tickformat="%b %Y",   # e.g. “Jan 2023”
            tickangle=-45,
            title_text=""
        ),
        yaxis=dict(
            title_text="Unit Sales",
            tickformat=","      # e.g. “1,234,567”
        ),
        legend=dict(
            title_text="",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=40, t=60, b=100)
    )

    # 9) Add numeric labels INSIDE each bar segment
    for region_name in pivot.columns:
        idx = [trace.name for trace in fig.data].index(region_name)
        fig.data[idx].update(texttemplate="%{y:,.0f}", textposition="inside")

    return fig

def load_and_prepare():
    """
    Load and clean raw Excel data, pivot monthly columns into a long DataFrame,
    and return with proper numeric types and extracted year/month.
    """
    RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
    all_dfs = []
    for file in RAW_DIR.glob("*.xlsx"):
        sheets = pd.read_excel(file, sheet_name=None, header=1)
        for _, df in sheets.items():
            # Normalize headers
            df.columns = (
                df.columns
                  .astype(str)
                  .str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace("/", "_")
            )
            # Ensure powertrain as string
            df["powertrain"] = df.get("powertrain", "").astype(str)
            all_dfs.append(df)

    # Combine all sheets
    combined = pd.concat(all_dfs, ignore_index=True).fillna(0)

    # Identify date columns YYYYMM
    date_cols = [c for c in combined.columns if re.match(r"^\d{6}$", c)]
    id_cols   = [c for c in combined.columns if c not in date_cols]

    # Melt wide to long
    long = combined.melt(
        id_vars=id_cols,
        value_vars=date_cols,
        var_name="yyyymm",
        value_name="total_sales"
    )

    # Clean and convert sales to float
    long["total_sales"] = (
        long["total_sales"]
          .astype(str)
          .str.strip()
          .replace(r"^-+$", "0", regex=True)
          .str.replace(r"[^\d\.-]", "", regex=True)
          .replace("", "0")
          .astype(float)
    )

    # Extract year and month
    long["year"]  = long["yyyymm"].str[:4].astype(int)
    long["month"] = long["yyyymm"].str[4:].astype(int)

    # Ensure key categoricals are strings
    for col in ["group", "maker_brand", "country"]:
        if col in long:
            long[col] = long[col].astype(str)

    return long

# Map each region name to the list of its countries
region_map = {
    "North America": ["USA","Canada"],
    "China": ["China"],
    "Europe": [
        "Albania","Andorra","Armenia","Austria","Azerbaijan","Belarus","Belgium",
        "Bosnia and Herzegovina","Bulgaria","Croatia","Cyprus","Czech Republic",
        "Denmark","Estonia","Finland","France","Georgia","Germany","Greece",
        "Hungary","Iceland","Ireland","Italy","Kazakhstan","Kosovo","Latvia",
        "Liechtenstein","Lithuania","Luxembourg","Malta","Moldova","Monaco",
        "Montenegro","Netherlands","North Macedonia","Norway","Poland","Portugal",
        "Romania","Russia","San Marino","Serbia","Slovakia","Slovenia","Spain",
        "Sweden","Switzerland","Turkey","Ukraine","UK"
    ],
    "Southeast Asia": [
        "Brunei","Myanmar","Cambodia","Timor-Leste","Indonesia","Laos","Malaysia",
        "Philippines","Singapore","Thailand","Vietnam"
    ],
    "LATAM": [
        "Argentina","Bolivia","Brazil","Chile","Colombia","Costa Rica","Cuba",
        "Dominican Republic","Ecuador","El Salvador","Guatemala","Honduras",
        "Mexico","Nicaragua","Panama","Paraguay","Peru","Uruguay","Venezuela"
    ],
    "Japan & Korea": ["Japan","Korea","South Korea","Republic of Korea"],
    "MEA": [
        "Algeria","Angola","Benin","Botswana","Burkina Faso","Burundi","Cameroon",
        "Cape Verde","Central African Republic","Chad","Comoros",
        "Democratic Republic of the Congo","Republic of the Congo","Djibouti",
        "Equatorial Guinea","Eritrea","Eswatini","Ethiopia","Gabon","Gambia",
        "Ghana","Guinea","Guinea-Bissau","Côte d'Ivoire","Kenya","Lesotho","Liberia",
        "Libya","Madagascar","Malawi","Mali","Mauritania","Mauritius","Morocco",
        "Mozambique","Namibia","Niger","Nigeria","Rwanda","São Tomé and Príncipe",
        "Senegal","Seychelles","Sierra Leone","Somalia","South Africa","South Sudan",
        "Sudan","Tanzania","Togo","Tunisia","Uganda","Zambia","Zimbabwe",
        "Bahrain","Cyprus","Egypt","Iran","Iraq","Israel","Jordan","Kuwait",
        "Lebanon","Oman","Palestine","Qatar","Saudi Arabia","Syria","United Arab Emirates","Yemen"
    ]
}
# ─────────────────────────────────────────────────────────
# List of Chinese OEMs to aggregate
chinese_oems = [
    "dongfeng (dongfeng motor corp.)",
    "baic group",
    "great wall motor company ltd. (gwm)",
    "changan/chana (changan automobile (group))",
    "saic (shanghai automotive industry corporation (group))",
    "chery automobile",
    "byd auto",
    "faw (china faw group corp.)",
    "gac group",
    "anhui jianghuai automotive group",
    "seres group",
    "jiangling motors co. group (2022-)",
    "shineray group",
    "fujian motor industry group co. (fjmg)",
    "li auto",
    "nio",
    "xpeng",
    "leapmotor",
    "daewoo bus corporation",
    "haima automobile group",
    "china national heavy duty truck group",
    "xiaomi"

]
# Lower-case them (your df["group"] is already lower-cased)
chinese_oems = [name.lower() for name in chinese_oems]
# ─────────────────────────────────────────────────────────

df = load_and_prepare()
st.title("Auto Research")

# ——— Sidebar: Region → OEM → Country → Brand → Year → Chart —————
region = st.sidebar.selectbox(
    "Region",
    list(region_map.keys()),
    key="region_selectbox"
)
df_region = df[df["country"].isin(region_map[region])]

oem_list = sorted(df_region["group"].astype(str).unique())
oem = st.sidebar.selectbox(
    "OEM (Group)",
    oem_list,
    key="oem_selectbox"
)

country_list = sorted(df_region["country"].astype(str).unique())
country = st.sidebar.selectbox(
    "Country",
    country_list,
    key="country_selectbox"
)

brand_list = sorted(
    df_region[
        (df_region["group"] == oem) &
        (df_region["country"] == country)
    ]["maker_brand"]
      .astype(str)
      .unique()
)
brand = st.sidebar.selectbox(
    "Brand",
    brand_list,
    key="brand_selectbox"
)

year_options = ["All"] + sorted(df_region["year"].astype(int).unique().tolist())
year = st.sidebar.selectbox(
    "Year",
    year_options,
    key="year_selectbox"
)

chart_choice = st.sidebar.selectbox(
    "Choose a chart:",
    [
      "Region Sales",
      "Country Sales",
      "OEM Sales",
      "Brand Sales",
      "Market Share",
      "Sales by Region (Pie)",
      "Consolidated Sales"
    ],
    key="chart_selectbox"
)
ma_option = st.sidebar.selectbox(
    "Moving Average:",
    ["None", "3-month", "12-month"],
    key="ma_option"
)

# ─────────────────────────────────────────────────────────


# REGION-LEVEL AGGREGATE (all countries in region, for chosen year or all)
if chart_choice == "Region Sales":
    # ─────────────────────────────────────────────────────────
    # 1) Filter by year
    if year == "All":
        df_region_year = df_region.copy()
    else:
        df_region_year = df_region[df_region["year"] == int(year)].copy()

    # 2) Compute PHEV / EV / ICE columns
    region_tr = df_region_year.copy()
    region_tr["phev"] = np.where(
        region_tr["powertrain"].str.lower().str.contains("phv", na=False),
        region_tr["total_sales"], 0
    )
    region_tr["ev"] = np.where(
        (region_tr["powertrain"].str.lower().str.contains("ev", na=False)) &
        (~region_tr["powertrain"].str.lower().str.contains("phv", na=False)),
        region_tr["total_sales"], 0
    )
    region_tr["ice"] = region_tr["total_sales"] - region_tr["ev"] - region_tr["phev"]

    # 3) Monthly sums
    region_sales_trend = region_tr.groupby("yyyymm")[["phev","ev","ice","total_sales"]].sum()
    region_sales_trend.index = pd.to_datetime(region_sales_trend.index, format="%Y%m")
    region_sales_trend.columns = ["PHEV Sales","EV Sales","ICE Sales","Total Sales"]
    nz_r = region_sales_trend.sum(axis=1) > 0
    if nz_r.any():
        region_sales_trend = region_sales_trend.loc[:region_sales_trend.index[nz_r][-1]]

    # ─────────────────────────────────────────────────────────
    # 4) Region-wide line chart + MA + download
    st.subheader(f"{region} – Region-wide Sales by Month")
    fig_region = px.line(
        region_sales_trend,
        x=region_sales_trend.index,
        y=["Total Sales","EV Sales","PHEV Sales","ICE Sales"],
        labels={"value":"Units","index":"Month"},
        template="plotly_white"
    )
    fig_region.update_layout(xaxis_tickformat="%b %Y", xaxis_tickangle=-45, yaxis_tickformat=",", legend_title_text="")
    # Remove the x-axis title
    fig_region.update_xaxes(title_text="")
    if ma_option in ("3-month","12-month"):
        w = 3 if ma_option=="3-month" else 12
        ma = region_sales_trend["Total Sales"].rolling(window=w).mean()
        fig_region.add_scatter(x=ma.index, y=ma, mode="lines", name=f"{w}-Mo MA", line=dict(dash="dash",color="black"))
    st.plotly_chart(fig_region, use_container_width=True)
    png1 = fig_region.to_image(format="png", width=800, height=400)
    st.download_button("📥 Download Region Sales as PNG", png1, f"{region}_region_sales.png", "image/png")

    # ─────────────────────────────────────────────────────────
    # 5) Region-wide composition bar
    comp_r = region_sales_trend.div(region_sales_trend["Total Sales"], axis=0)
    fig_bar_r = px.bar(
        comp_r,
        x=comp_r.index,
        y=["EV Sales","PHEV Sales","ICE Sales"],
        labels={"value":"% of Sales","index":"Month"},
        template="plotly_white"
    )
    fig_bar_r.update_layout(barmode="stack", xaxis_tickformat="%b %Y", xaxis_tickangle=-45, yaxis_tickformat=".0%", legend_title_text="")
    fig_bar_r.update_xaxes(title_text="")
    st.plotly_chart(fig_bar_r, use_container_width=True)
    png2 = fig_bar_r.to_image(format="png", width=800, height=400)
    st.download_button("📥 Download Region Composition as PNG", png2, f"{region}_composition.png", "image/png")

    # ─────────────────────────────────────────────────────────
    # 6) Country-specific line chart + MA + download
    if year == "All":
        df_cty = df_region[df_region["country"]==country].copy()
    else:
        df_cty = df_region[
            (df_region["country"]==country)&
            (df_region["year"]==int(year))
        ].copy()

    if df_cty.empty:
        st.warning(f"No {country} data in {region} for {year}.")
    else:
        country_tr = df_cty.copy()
        country_tr["phev"] = np.where(country_tr["powertrain"].str.lower().str.contains("phv", na=False), country_tr["total_sales"], 0)
        country_tr["ev"] = np.where(
            (country_tr["powertrain"].str.lower().str.contains("ev", na=False)) &
            (~country_tr["powertrain"].str.lower().str.contains("phv", na=False)),
            country_tr["total_sales"], 0
        )
        country_tr["ice"] = country_tr["total_sales"] - country_tr["ev"] - country_tr["phev"]

        country_sales_trend = country_tr.groupby("yyyymm")[["phev","ev","ice","total_sales"]].sum()
        country_sales_trend.index = pd.to_datetime(country_sales_trend.index, format="%Y%m")
        country_sales_trend.columns = ["PHEV Sales","EV Sales","ICE Sales","Total Sales"]
        nz_c = country_sales_trend.sum(axis=1) > 0
        if nz_c.any():
            country_sales_trend = country_sales_trend.loc[:country_sales_trend.index[nz_c][-1]]

        st.subheader(f"{country} – Sales by Month")
        fig_cty = px.line(
            country_sales_trend,
            x=country_sales_trend.index,
            y=["Total Sales","EV Sales","PHEV Sales","ICE Sales"],
            labels={"value":"Units","index":"Month"},
            template="plotly_white"
        )
        fig_cty.update_layout(xaxis_tickformat="%b %Y", xaxis_tickangle=-45, yaxis_tickformat=",", legend_title_text="")
        fig_cty.update_xaxes(title_text="")
        if ma_option in ("3-month","12-month"):
            w2 = 3 if ma_option=="3-month" else 12
            ma2 = country_sales_trend["Total Sales"].rolling(window=w2).mean()
            fig_cty.add_scatter(x=ma2.index, y=ma2, mode="lines", name=f"{w2}-Mo MA", line=dict(dash="dash",color="black"))
        st.plotly_chart(fig_cty, use_container_width=True)
        png3 = fig_cty.to_image(format="png", width=800, height=400)
        st.download_button("📥 Download Country Sales as PNG", png3, f"{country}_sales.png", "image/png")

    # ─────────────────────────────────────────────────────────
    # 7) Country-specific composition bar
    if not df_cty.empty:
        comp_c = country_sales_trend.div(country_sales_trend["Total Sales"], axis=0)
        fig_bar_c = px.bar(
            comp_c,
            x=comp_c.index,
            y=["EV Sales","PHEV Sales","ICE Sales"],
            labels={"value":"% of Sales","index":"Month"},
            template="plotly_white"
        )
        fig_bar_c.update_layout(barmode="stack", xaxis_tickformat="%b %Y", xaxis_tickangle=-45, yaxis_tickformat=".0%", legend_title_text="")
        fig_bar_c.update_xaxes(title_text="")
        st.plotly_chart(fig_bar_c, use_container_width=True)
        png4 = fig_bar_c.to_image(format="png", width=800, height=400)
        st.download_button("📥 Download Country Composition as PNG", png4, f"{country}_composition.png", "image/png")
    # ─────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────
# OEM-LEVEL AGGREGATE (all countries in region, for chosen year or all)
if chart_choice == "OEM Sales":
    # ─────────────────────────────────────────────────────────
    # 1) OEM‐LEVEL (region-wide) – filtered by year or All
    if year == "All":
        df_oem_year = df_region[df_region["group"] == oem].copy()
    else:
        df_oem_year = df_region[
            (df_region["group"] == oem) &
            (df_region["year"] == int(year))
        ].copy()

    # Compute EV / PHEV / ICE breakdown for OEM region‐wide
    df_oem_year["phev_sales"] = np.where(
        df_oem_year["powertrain"].str.lower().str.contains("phv", na=False),
        df_oem_year["total_sales"],
        0.0
    )
    df_oem_year["ev_sales"] = np.where(
        (df_oem_year["powertrain"].str.lower().str.contains("ev", na=False)) &
        (~df_oem_year["powertrain"].str.lower().str.contains("phv", na=False)),
        df_oem_year["total_sales"],
        0.0
    )
    df_oem_year["ice_sales"] = (
        df_oem_year["total_sales"]
        - df_oem_year["ev_sales"]
        - df_oem_year["phev_sales"]
    )

    oem_sales_trend = df_oem_year.groupby("yyyymm")[
        ["ev_sales", "phev_sales", "ice_sales", "total_sales"]
    ].sum()

    # Convert index “yyyymm” → datetime
    oem_sales_trend.index = pd.to_datetime(oem_sales_trend.index, format="%Y%m")

    # Rename for display
    oem_sales_trend = oem_sales_trend.rename(columns={
        "ev_sales":    "EV Sales",
        "phev_sales":  "PHEV Sales",
        "ice_sales":   "ICE Sales",
        "total_sales": "Total Sales"
    })

    # Trim trailing zero‐rows
    nonzero_mask = (oem_sales_trend.sum(axis=1) > 0)
    if nonzero_mask.any():
        last_idx = oem_sales_trend.index[nonzero_mask][-1]
        oem_sales_trend = oem_sales_trend.loc[:last_idx]

 ######################################################

    st.subheader(f"{oem} (Region-wide) – Sales by Month")

    fig_oem = px.line(
    oem_sales_trend,
    x=oem_sales_trend.index,
    y=["EV Sales", "PHEV Sales", "ICE Sales", "Total Sales"],
    labels={"value": "Units Sold", "index": "Month"},
    title=f"{oem} – Total vs. EV/PHEV/ICE Sales",
    template="plotly_white"
    )
    fig_oem.update_layout(
    xaxis=dict(
        tickmode="auto",
        tickformat="%b %Y",
        tickangle=-45
    ),
    yaxis=dict(
        title="Units Sold",
        tickformat=","
    ),
    legend_title_text=""
    )
    fig_oem.update_xaxes(title_text="")
    if ma_option in ("3-month", "12-month"):
        window = 3 if ma_option == "3-month" else 12
        ma_oem = oem_sales_trend["Total Sales"].rolling(window=window).mean()
        fig_oem.add_scatter(
            x=ma_oem.index,
            y=ma_oem.values,
            mode="lines",
            name=f"{window}-Mo MA",
            line=dict(dash="dash", color="black")
        )
    st.plotly_chart(fig_oem, use_container_width=True)

    png_oem = fig_oem.to_image(format="png", width=800, height=400)
    st.download_button(
    label="📥 Download OEM Sales as PNG",
    data=png_oem,
    file_name=f"{oem}_oem_sales.png",
    mime="image/png"
    )


    # ─────────────────────────────────────────────────────────
    # 2) OEM’s sales in the selected Country
    df_oem_country = df_region[
        (df_region["group"] == oem) &
        (df_region["country"] == country) &
        ((df_region["year"] == int(year)) if year != "All" else True)
    ].copy()

    if not df_oem_country.empty:
        df_oem_country["phev_sales"] = np.where(
            df_oem_country["powertrain"].str.lower().str.contains("phv", na=False),
            df_oem_country["total_sales"],
            0.0
        )
        df_oem_country["ev_sales"] = np.where(
            (df_oem_country["powertrain"].str.lower().str.contains("ev", na=False)) &
            (~df_oem_country["powertrain"].str.lower().str.contains("phv", na=False)),
            df_oem_country["total_sales"],
            0.0
        )
        df_oem_country["ice_sales"] = (
            df_oem_country["total_sales"]
            - df_oem_country["ev_sales"]
            - df_oem_country["phev_sales"]
        )

        oem_country_trend = df_oem_country.groupby("yyyymm")[
            ["ev_sales", "phev_sales", "ice_sales", "total_sales"]
        ].sum()
        oem_country_trend.index = pd.to_datetime(oem_country_trend.index, format="%Y%m")
        oem_country_trend = oem_country_trend.rename(columns={
            "ev_sales":    "EV Sales",
            "phev_sales":  "PHEV Sales",
            "ice_sales":   "ICE Sales",
            "total_sales": "Total Sales"
        })

        # Trim trailing zeros
        nz_mask2 = (oem_country_trend.sum(axis=1) > 0)
        if nz_mask2.any():
            last_idx2 = oem_country_trend.index[nz_mask2][-1]
            oem_country_trend = oem_country_trend.loc[:last_idx2]

    # After computing oem_country_trend...
        fig_oem_cty = px.line(
        oem_country_trend,
        x=oem_country_trend.index,
        y=["EV Sales", "PHEV Sales", "ICE Sales", "Total Sales"],
        labels={"value": "Units Sold", "index": "Month"},
        title=f"{oem} in {country} – Total vs. EV/PHEV/ICE Sales",
        template="plotly_white"
        )
        fig_oem_cty.update_layout(
        xaxis=dict(tickformat="%b %Y", tickangle=-45),
        yaxis=dict(title="Units Sold", tickformat=","),
        legend_title_text=""
        )
        if ma_option in ("3-month","12-month"):
            w = 3 if ma_option=="3-month" else 12
            ma_cty = oem_country_trend["Total Sales"].rolling(window=w).mean()
            fig_oem_cty.add_scatter(x=ma_cty.index,y=ma_cty,mode="lines",name=f"{w}-Mo MA",line=dict(dash="dash",color="black"))
        st.plotly_chart(fig_oem_cty, use_container_width=True)

        png_oem_cty = fig_oem_cty.to_image(format="png", width=800, height=400)
        st.download_button(
        label="📥 Download OEM‐in‐Country Sales as PNG",
        data=png_oem_cty,
        file_name=f"{oem}_{country}_sales.png",
        mime="image/png"
        )

        
    else:
        st.warning(f"No {oem} sales data in {country} for {year}.")
    # ─────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────


# Show OEM-level chart below (we’ll insert it next)


brand_list = sorted(
    df_region[
        (df_region["group"] == oem) &
        (df_region["country"] == country)
    ]["maker_brand"]
      .astype(str)
      .unique()
)


# ─────────────────────────────────────────────────────────
# BRAND-LEVEL FILTER & CHART (filtered by chosen year or “All”)
if chart_choice == "Brand Sales":
    # ─────────────────────────────────────────────────────────
    # 1) BRAND‐LEVEL (region‐wide) – filtered by chosen year or “All”
    if year == "All":
        df_brand_region = df_region[
            (df_region["group"] == oem) &
            (df_region["maker_brand"] == brand)
        ].copy()
    else:
        df_brand_region = df_region[
            (df_region["group"] == oem) &
            (df_region["maker_brand"] == brand) &
            (df_region["year"] == int(year))
        ].copy()

    if df_brand_region.empty:
        st.warning(f"No {brand} data in the region for {year}.")
    else:
        total_sales = df_brand_region["total_sales"].sum()
        # Compute EV and EV+PHV share for the metric (optional; unchanged)
        ev_mask = df_brand_region["powertrain"].str.lower().str.contains("ev", na=False)
        ev_sales = df_brand_region.loc[ev_mask, "total_sales"].sum()
        ev_share = ev_sales / total_sales if total_sales else 0.0

        c1, c2 = st.columns(2)
        c1.metric("Total Sales (Region)", int(total_sales))
        c2.metric("EV Adoption Rate", f"{ev_share:.1%}")

        region_tr = df_brand_region.copy()

        # 1) PHEV sales
        region_tr["phev_sales"] = np.where(
            region_tr["powertrain"].str.lower().str.contains("phv", na=False),
            region_tr["total_sales"],
            0.0
        )
        # 2) EV sales (excluding PHV)
        region_tr["ev_sales"] = np.where(
            (region_tr["powertrain"].str.lower().str.contains("ev", na=False)) &
            (~region_tr["powertrain"].str.lower().str.contains("phv", na=False)),
            region_tr["total_sales"],
            0.0
        )
        # 3) ICE sales = total – EV – PHEV
        region_tr["ice_sales"] = (
            region_tr["total_sales"]
            - region_tr["ev_sales"]
            - region_tr["phev_sales"]
        )

        brand_region_trend = region_tr.groupby(
            "yyyymm"
        )[
            ["ev_sales", "phev_sales", "ice_sales", "total_sales"]
        ].sum()
        brand_region_trend.index = pd.to_datetime(brand_region_trend.index, format="%Y%m")
        brand_region_trend = brand_region_trend.rename(columns={
            "ev_sales":    "EV Sales",
            "phev_sales":  "PHEV Sales",
            "ice_sales":   "ICE Sales",
            "total_sales": "Total Sales"
        })

        # Trim trailing zero‐rows
        nz_mask_r = (brand_region_trend.sum(axis=1) > 0)
        if nz_mask_r.any():
            last_idx_r = brand_region_trend.index[nz_mask_r][-1]
            brand_region_trend = brand_region_trend.loc[:last_idx_r]

        # (1) Region‐wide Brand chart
        st.subheader(f"{brand} (Region-wide) – Sales by Month")

        fig_brand_reg = px.line(
            brand_region_trend,
            x=brand_region_trend.index,
            y=["EV Sales", "PHEV Sales", "ICE Sales", "Total Sales"],
            labels={"value": "Units Sold", "index": "Month"},
            title=f"{brand} (Region-wide) – Total vs. EV/PHEV/ICE Sales",
            template="plotly_white"
        )
        fig_brand_reg.update_layout(
            xaxis=dict(tickformat="%b %Y", tickangle=-45),
            yaxis=dict(title="Units Sold", tickformat=","),
            legend_title_text=""
        )
        fig_brand_reg.update_xaxes(title_text="")
        # Overlay moving average on Brand region‐wide “Total Sales”
        if ma_option in ("3-month", "12-month"):
            window = 3 if ma_option == "3-month" else 12
            ma_brand_reg = brand_region_trend["Total Sales"].rolling(window=window).mean()
            fig_brand_reg.add_scatter(
                x=ma_brand_reg.index,
                y=ma_brand_reg.values,
                mode="lines",
                name=f"{window}-Mo MA",
                line=dict(dash="dash", color="black")
            )

        st.plotly_chart(fig_brand_reg, use_container_width=True)

        png_brand_reg = fig_brand_reg.to_image(format="png", width=800, height=400)
        st.download_button(
            label="📥 Download Brand Region Sales as PNG",
            data=png_brand_reg,
            file_name=f"{brand}_region_sales.png",
            mime="image/png"
        )

    # ─────────────────────────────────────────────────────────
    # 2) BRAND SALES in the selected Country
    df_brand_country = df_region[
        (df_region["group"] == oem) &
        (df_region["maker_brand"] == brand) &
        (df_region["country"] == country) &
        ((df_region["year"] == int(year)) if year != "All" else True)
    ].copy()

    if not df_brand_country.empty:
        df_brand_country["phev_sales"] = np.where(
            df_brand_country["powertrain"].str.lower().str.contains("phv", na=False),
            df_brand_country["total_sales"],
            0.0
        )
        df_brand_country["ev_sales"] = np.where(
            (df_brand_country["powertrain"].str.lower().str.contains("ev", na=False)) &
            (~df_brand_country["powertrain"].str.lower().str.contains("phv", na=False)),
            df_brand_country["total_sales"],
            0.0
        )
        df_brand_country["ice_sales"] = (
            df_brand_country["total_sales"]
            - df_brand_country["ev_sales"]
            - df_brand_country["phev_sales"]
        )

        brand_country_trend = df_brand_country.groupby(
            "yyyymm"
        )[
            ["ev_sales", "phev_sales", "ice_sales", "total_sales"]
        ].sum()
        brand_country_trend.index = pd.to_datetime(brand_country_trend.index, format="%Y%m")
        brand_country_trend = brand_country_trend.rename(columns={
            "ev_sales":    "EV Sales",
            "phev_sales":  "PHEV Sales",
            "ice_sales":   "ICE Sales",
            "total_sales": "Total Sales"
        })

        # Trim trailing zero‐rows
        nz_mask_c = (brand_country_trend.sum(axis=1) > 0)
        if nz_mask_c.any():
            last_idx_c = brand_country_trend.index[nz_mask_c][-1]
            brand_country_trend = brand_country_trend.loc[:last_idx_c]

        # (2) Brand‐in‐Country chart
        st.subheader(f"{brand} in {country} – Sales by Month")

        fig_brand_cty = px.line(
            brand_country_trend,
            x=brand_country_trend.index,
            y=["EV Sales", "PHEV Sales", "ICE Sales", "Total Sales"],
            labels={"value": "Units Sold", "index": "Month"},
            title=f"{brand} in {country} – Total vs. EV/PHEV/ICE Sales",
            template="plotly_white"
        )
        fig_brand_cty.update_layout(
            xaxis=dict(tickformat="%b %Y", tickangle=-45),
            yaxis=dict(title="Units Sold", tickformat=","),
            legend_title_text=""
        )
        # Overlay moving average on Brand‐in‐Country “Total Sales”
        if ma_option in ("3-month", "12-month"):
            window = 3 if ma_option == "3-month" else 12
            ma_brand_cty = brand_country_trend["Total Sales"].rolling(window=window).mean()
            fig_brand_cty.add_scatter(
                x=ma_brand_cty.index,
                y=ma_brand_cty.values,
                mode="lines",
                name=f"{window}-Mo MA",
                line=dict(dash="dash", color="black")
            )

        st.plotly_chart(fig_brand_cty, use_container_width=True)

        png_brand_cty = fig_brand_cty.to_image(format="png", width=800, height=400)
        st.download_button(
            label="📥 Download Brand Country Sales as PNG",
            data=png_brand_cty,
            file_name=f"{brand}_{country}_sales.png",
            mime="image/png"
        )
    else:
        st.warning(f"No {brand} data in {country} for {year}.")
    # ─────────────────────────────────────────────────────────

   
# ─────────────────────────────────────────────────────────

elif chart_choice == "Market Share":
    # ─────────────────────────────────────────────────────────
    # 1) REGION-LEVEL Market Share
    df_reg = df[df["country"].isin(region_map[region])].copy()
    if year != "All":
        df_reg = df_reg[df_reg["year"] == int(year)].copy()

    if df_reg.empty:
        st.warning(f"No data for {region} in {year}.")
    else:
        # a) Parse month_year
        df_reg["month_year"] = pd.to_datetime(df_reg["yyyymm"], format="%Y%m")

        # b) Sum sales per OEM per month
        month_sum = (
            df_reg
              .groupby(["month_year", "group"])["total_sales"]
              .sum()
              .reset_index(name="oem_sales")
        )
        # c) Total region sales per month
        region_tot = (
            df_reg
              .groupby("month_year")["total_sales"]
              .sum()
              .reset_index(name="region_total")
        )

        # d) Merge and compute share
        merged = pd.merge(month_sum, region_tot, on="month_year")
        merged["share"] = merged["oem_sales"] / merged["region_total"]

        # e) Pivot into wide form
        share_wide_reg = (
            merged
              .pivot(index="month_year", columns="group", values="share")
              .fillna(0)
        )

        # 🚨 Drop any pre-existing Chinese OEM Share column
        if "Chinese OEM Share" in share_wide_reg.columns:
            share_wide_reg = share_wide_reg.drop(columns=["Chinese OEM Share"])

        # f) Compute aggregated Chinese OEM Share
        chinese_cols = [
            col for col in share_wide_reg.columns
            if any(kw in col.lower() for kw in chinese_oems)
        ]
        share_wide_reg["Chinese OEM Share"] = share_wide_reg[chinese_cols].sum(axis=1)

        # g) Keep only OEMs >4% or our new series
        cols_reg = [
            col for col in share_wide_reg.columns
            if share_wide_reg[col].max() > 0.04
        ] + ["Chinese OEM Share"]
        share_wide_reg = share_wide_reg[cols_reg]

        # 🚨 Final dedupe: ensure unique column names
        share_wide_reg = share_wide_reg.loc[:, ~share_wide_reg.columns.duplicated(keep="first")]

        # h) Trim trailing zero-rows
        nz = share_wide_reg.sum(axis=1) > 0
        if nz.any():
            last = share_wide_reg.index[nz][-1]
            share_wide_reg = share_wide_reg.loc[:last]

        # i) Plot Region-level Market Share
        st.subheader(f"{region} – OEM Market Share (Region-wide){'' if year=='All' else f' – {year}'}")
        fig_reg_ms = px.line(
            share_wide_reg,
            x=share_wide_reg.index,
            y=share_wide_reg.columns,
            labels={"value": "Market Share", "month_year": "Month"},
            title=f"{region} – OEM Market Share",
            template="plotly_white"
        )
        fig_reg_ms.update_layout(
            xaxis=dict(tickformat="%b %Y", tickangle=-45),
            yaxis=dict(title="Market Share", tickformat=".0%"),
            legend_title_text=""
        )
        fig_reg_ms.update_xaxes(title_text="")
        st.plotly_chart(fig_reg_ms, use_container_width=True)
        png_reg_ms = fig_reg_ms.to_image(format="png", width=800, height=400)
        st.download_button(
            label="📥 Download Region Market Share as PNG",
            data=png_reg_ms,
            file_name=f"{region}_market_share.png",
            mime="image/png"
        )

    # ─────────────────────────────────────────────────────────
    # 2) COUNTRY-LEVEL Market Share
    df_cty = df_reg[df_reg["country"] == country].copy()

    if df_cty.empty:
        st.warning(f"No data for {country} in {year}.")
    else:
        # a) Parse month_year
        df_cty["month_year"] = pd.to_datetime(df_cty["yyyymm"], format="%Y%m")

        # b) Sum sales per OEM per month for country
        month_sum_cty = (
            df_cty
              .groupby(["month_year", "group"])["total_sales"]
              .sum()
              .reset_index(name="oem_sales")
        )
        # c) Total country sales per month
        country_tot = (
            df_cty
              .groupby("month_year")["total_sales"]
              .sum()
              .reset_index(name="country_total")
        )

        # d) Merge and compute share
        merged_cty = pd.merge(month_sum_cty, country_tot, on="month_year")
        merged_cty["share"] = merged_cty["oem_sales"] / merged_cty["country_total"]

        # e) Pivot into wide form
        share_wide_cty = (
            merged_cty
              .pivot(index="month_year", columns="group", values="share")
              .fillna(0)
        )

        # 🚨 Drop any pre-existing Chinese OEM Share column
        if "Chinese OEM Share" in share_wide_cty.columns:
            share_wide_cty = share_wide_cty.drop(columns=["Chinese OEM Share"])

        # f) Compute aggregated Chinese OEM Share
        chinese_cols_cty = [
            col for col in share_wide_cty.columns
            if any(kw in col.lower() for kw in chinese_oems)
        ]
        share_wide_cty["Chinese OEM Share"] = share_wide_cty[chinese_cols_cty].sum(axis=1)

        # g) Keep only OEMs >4% or our new series
        cols_cty = [
            col for col in share_wide_cty.columns
            if share_wide_cty[col].max() > 0.04
        ] + ["Chinese OEM Share"]
        share_wide_cty = share_wide_cty[cols_cty]

        # 🚨 Final dedupe: ensure unique column names
        share_wide_cty = share_wide_cty.loc[:, ~share_wide_cty.columns.duplicated(keep="first")]

        # h) Trim trailing zero-rows
        nz_cty = share_wide_cty.sum(axis=1) > 0
        if nz_cty.any():
            last_cty = share_wide_cty.index[nz_cty][-1]
            share_wide_cty = share_wide_cty.loc[:last_cty]

        # i) Plot Country-level Market Share
        st.subheader(f"{country} – OEM Market Share (Country-level){'' if year=='All' else f' – {year}'}")
        fig_cty_ms = px.line(
            share_wide_cty,
            x=share_wide_cty.index,
            y=share_wide_cty.columns,
            labels={"value": "Market Share", "month_year": "Month"},
            title=f"{country} – OEM Market Share",
            template="plotly_white"
        )
        fig_cty_ms.update_layout(
            xaxis=dict(tickformat="%b %Y", tickangle=-45),
            yaxis=dict(title="Market Share", tickformat=".0%"),
            legend_title_text=""
        )
        fig_cty_ms.update_xaxes(title_text="")
        st.plotly_chart(fig_cty_ms, use_container_width=True)
        png_cty_ms = fig_cty_ms.to_image(format="png", width=800, height=400)
        st.download_button(
            label="📥 Download Country Market Share as PNG",
            data=png_cty_ms,
            file_name=f"{country}_market_share.png",
            mime="image/png"
        )
    # ─────────────────────────────────────────────────────────



if chart_choice == "Country Sales":
    # ─────────────────────────────────────────────────────────
    # 1) Filter df by the chosen region and year (or all)
    df_region = df[df["country"].isin(region_map[region])].copy()

    if year != "All":
        df_region = df_region[df_region["year"] == int(year)].copy()

    # 2) If there’s no data, warn and skip
    if df_region.empty:
        st.warning(f"No data for {region} in {year}.")
    else:
        # 3) Sum total_sales by country
        country_totals = (
            df_region
              .groupby("country")["total_sales"]
              .sum()
              .reset_index(name="total_sales")
        )

        # 4) Sort descending so biggest countries appear first
        country_totals = country_totals.sort_values(
            by="total_sales", ascending=False
        )

        # 5) (Optional) Drop any country with zero if you like:
        country_totals = country_totals[country_totals["total_sales"] > 0]

        # 6) Build a Plotly bar chart
        import plotly.express as px
        fig_country = px.bar(
            country_totals,
            x="country",
            y="total_sales",
            labels={"country": "Country", "total_sales": "Total Sales"},
            title=(
                f"{region} – Total Sales by Country"
                + ("" if year == "All" else f" – {year}")
            ),
            template="plotly_white"
        )
        fig_country.update_layout(
            xaxis_tickangle=-45,
            yaxis=dict(tickformat=","),
            margin={"t": 40, "b": 80},
        )

        # 7) Render in Streamlit
        st.plotly_chart(fig_country, use_container_width=True)
        
        # 8) Provide a Download‐PNG button
        png_country = fig_country.to_image(format="png", width=800, height=500)
        st.download_button(
            label="📥 Download Country Sales Bar Chart as PNG",
            data=png_country,
            file_name=f"{region}_country_sales.png",
            mime="image/png"
        )
    # ─────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────
if chart_choice == "Sales by Region (Pie)":
    # ─────────────────────────────────────────────────────────
    # Pie of total sales by region, for selected OEM (or Brand)
    # Ask user if pie should be by OEM or by Brand
    pie_level = st.sidebar.radio(
        "Pie level:",
        ["OEM", "Brand"],
        key = "pielevel_radiobutton"
    )

    # Base DataFrame: filter by year if needed
    if year == "All":
        df_for_pie = df.copy()
    else:
        df_for_pie = df[df["year"] == int(year)].copy()

    # Further filter by OEM or Brand
    if pie_level == "OEM":
        df_for_pie = df_for_pie[df_for_pie["group"] == oem].copy()
    else:  # pie_level == "Brand"
        df_for_pie = df_for_pie[
            (df_for_pie["group"] == oem) &
            (df_for_pie["maker_brand"] == brand)
        ].copy()

    # Map each country → region via region_map
    country_to_region = {c: r for r,countries in region_map.items() for c in countries}
    df_for_pie["region"] = df_for_pie["country"].map(country_to_region)

    pie_data = (
        df_for_pie
          .groupby("region")["total_sales"]
          .sum()
          .reset_index(name="region_sales")
          .dropna(subset=["region"])
    )

    st.subheader(
        f"Sales by Region ({pie_level})"
        + (f" – {year}" if year != "All" else " (All Years)")
    )

    # Draw Plotly pie chart
    import plotly.express as px
    fig = px.pie(
        pie_data,
        names="region",
        values="region_sales",
        hole=0.3,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
    png_pie = fig.to_image(format="png", width=600, height=500)
    st.download_button(
    label="📥 Download Pie Chart as PNG",
    data=png_pie,
    file_name=f"sales_by_region_pie_{year}.png",
    mime="image/png"
    )
    # ─────────────────────────────────────────────────────────
elif chart_choice == "Consolidated Sales":
    st.subheader("Consolidated Unit Sales, by Region")

    # (A) OEM‐level consolidated chart
    df_oem = df[df["group"] == oem].copy()
    if df_oem.empty:
        st.warning(f"No data available for OEM: {oem}")
    else:
        fig_oem = plot_consolidated_sales_by_region(
            df_oem,
            region_map,
            f"{oem}’s Consolidated Unit Sales, by Region"
        )
        st.plotly_chart(fig_oem, use_container_width=True)

        png_oem_con = fig_oem.to_image(format="png", width=900, height=500)
        st.download_button(
            label="📥 Download OEM Consolidated as PNG",
            data=png_oem_con,
            file_name=f"{oem}_consolidated.png",
            mime="image/png"
        )

    # (B) BRAND‐level consolidated chart (if a brand is also selected)
    if brand:
        df_brand = df[
            (df["group"] == oem) &
            (df["maker_brand"] == brand)
        ].copy()
        if df_brand.empty:
            st.warning(f"No brand-level data for {brand}")
        else:
            fig_brand = plot_consolidated_sales_by_region(
                df_brand,
                region_map,
                f"{brand}’s Consolidated Unit Sales, by Region"
            )
            st.plotly_chart(fig_brand, use_container_width=True)
            
            png_brand_con = fig_brand.to_image(format="png", width=900, height=500)
            st.download_button(
                label="📥 Download Brand Consolidated as PNG",
                data=png_brand_con,
                file_name=f"{brand}_consolidated.png",
                mime="image/png"
            )


#run app
#conda activate car-market-app
#cd C:\Users\kwortelboer\Documents\car-market-app
#streamlit run src\app.py
