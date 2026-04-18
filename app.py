"""
app.py — Streamlit dashboard for earthquake_analysis.

Run:  streamlit run app.py

The app loads data/cleaned.csv (produced by scripts/run_pipeline.py) and
displays interactive charts for each of the four research questions.
"""

import os
import sys
import pandas as pd
import streamlit as st
import plotly.express as px

sys.path.insert(0, os.path.dirname(__file__))
from earthquake_analysis.analyze import (
    magnitude_vs_impact,
    depth_vs_impact,
    regional_impact,
    vulnerability_index,
    yearly_trends,
    rolling_average,
)

CLEANED_PATH = os.path.join(os.path.dirname(__file__), "data", "analysis_subset.csv")

st.set_page_config(page_title="Global Earthquake Analysis", layout="wide")
st.title("🌍 Global Earthquake Patterns & Real-World Impact (2000–2024)")


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    return df


if not os.path.exists(CLEANED_PATH):
    st.error(
        f"Analysis dataset not found at `{CLEANED_PATH}`.\n\n"
        "Run the pipeline first:\n```\npython scripts/run_pipeline.py\n```"
    )
    st.stop()

df = load_data(CLEANED_PATH)
st.caption(f"Dataset: {len(df):,} matched USGS+NCEI earthquake records")

# Sidebar magnitude filter
mag_range = st.sidebar.slider(
    "Magnitude range",
    float(df["magnitude"].min()),
    float(df["magnitude"].max()),
    (4.0, float(df["magnitude"].max())),
    step=0.5,
)
df_filt = df[(df["magnitude"] >= mag_range[0]) & (df["magnitude"] <= mag_range[1])]
st.sidebar.write(f"{len(df_filt):,} events in selection")


# ── Q1: Magnitude vs. Impact ──────────────────────────────────────────────────
st.header("Q1 — Does higher magnitude mean more deaths and damage?")

col1, col2 = st.columns(2)
q1 = magnitude_vs_impact(df_filt)

with col1:
    fig = px.bar(
        q1.dropna(subset=["median_deaths"]),
        x="mag_bin", y="median_deaths",
        title="Median Deaths by Magnitude Bin",
        labels={"mag_bin": "Magnitude", "median_deaths": "Median Deaths"},
    )
    st.plotly_chart(fig, width='stretch')

with col2:
    fig = px.bar(
        q1.dropna(subset=["median_damage_millions"]),
        x="mag_bin", y="median_damage_millions",
        title="Median Damage (M$) by Magnitude Bin",
        labels={"mag_bin": "Magnitude", "median_damage_millions": "Median Damage (M$)"},
        color_discrete_sequence=["#e07b2e"],
    )
    st.plotly_chart(fig, width='stretch')

with st.expander("Show magnitude–impact table"):
    st.dataframe(q1)


# ── Q2: Depth vs. Impact ─────────────────────────────────────────────────────
st.header("Q2 — Does depth influence damage severity?")

q2 = depth_vs_impact(df_filt)
col3, col4 = st.columns(2)

with col3:
    fig = px.bar(
        q2, x="depth_category", y="median_deaths",
        title="Median Deaths by Depth Category (M5+)",
        color="depth_category",
        labels={"depth_category": "Depth", "median_deaths": "Median Deaths"},
    )
    st.plotly_chart(fig, width='stretch')

with col4:
    fig = px.bar(
        q2, x="depth_category", y="total_events",
        title="Event Count by Depth Category",
        color="depth_category",
        labels={"depth_category": "Depth", "total_events": "# Events"},
    )
    st.plotly_chart(fig, width='stretch')


# ── Q3: Regional Vulnerability ────────────────────────────────────────────────
st.header("Q3 — Which regions are hit hardest?")

top_n = st.slider("Number of regions to show", 5, 30, 15)
q3 = regional_impact(df_filt, top_n=top_n)

fig = px.bar(
    q3.sort_values("total_deaths"),
    x="total_deaths", y="region", orientation="h",
    title=f"Top {top_n} Regions by Total Deaths (2000–2024)",
    labels={"total_deaths": "Total Deaths", "region": "Region"},
    color="deaths_per_event",
    color_continuous_scale="Reds",
)
st.plotly_chart(fig, width='stretch')

st.subheader("Vulnerability Index (deaths per event / median magnitude)")
vi = vulnerability_index(df_filt, min_events=3).head(top_n)
fig2 = px.bar(
    vi.sort_values("vulnerability_score"),
    x="vulnerability_score", y="region", orientation="h",
    title="Most Vulnerable Regions (disproportionate deaths per quake size)",
    labels={"vulnerability_score": "Vulnerability Score", "region": "Region"},
    color="vulnerability_score",
    color_continuous_scale="Oranges",
)
st.plotly_chart(fig2, width='stretch')


# ── Q4: Trends Over Time ─────────────────────────────────────────────────────
st.header("Q4 — Has the human cost changed over time?")

q4 = rolling_average(yearly_trends(df_filt), window=5)

col5, col6 = st.columns(2)

with col5:
    fig = px.bar(
        q4, x="year", y="total_deaths", title="Annual Deaths from Significant Earthquakes",
        labels={"year": "Year", "total_deaths": "Total Deaths"},
    )
    fig.add_scatter(x=q4["year"], y=q4["deaths_rolling"], mode="lines",
                    name="5-year rolling mean", line=dict(color="red", width=2))
    st.plotly_chart(fig, width='stretch')

with col6:
    fig = px.bar(
        q4, x="year", y="total_damage_millions",
        title="Annual Property Damage (M$)",
        labels={"year": "Year", "total_damage_millions": "Damage (M$)"},
        color_discrete_sequence=["#e07b2e"],
    )
    fig.add_scatter(x=q4["year"], y=q4["damage_rolling"], mode="lines",
                    name="5-year rolling mean", line=dict(color="red", width=2))
    st.plotly_chart(fig, width='stretch')

with st.expander("Show yearly data table"):
    st.dataframe(q4)
