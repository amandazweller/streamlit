"""
analyze.py — Answer the four research questions about global earthquake impact.

Each function takes the cleaned merged DataFrame and returns a DataFrame or
Series that can be plotted directly or passed to the Streamlit app.

Research questions:
  Q1. Does higher magnitude mean more deaths/damage? Is there a deadly threshold?
  Q2. Does depth influence damage severity?
  Q3. Which regions are hit hardest? Are some areas more vulnerable per magnitude?
  Q4. Has the human cost / property damage changed over time?
"""

import numpy as np
import pandas as pd


# ── Q1: Magnitude vs. impact ──────────────────────────────────────────────────

def magnitude_vs_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin earthquakes by 0.5-magnitude intervals and compute median deaths,
    median damage, and total records per bin.

    Returns a DataFrame indexed by magnitude bin with columns:
        median_deaths, median_damage_millions, total_events, pct_with_deaths
    """
    df = df.copy()
    df["mag_bin"] = pd.cut(df["magnitude"], bins=np.arange(0, 10.5, 0.5))

    grouped = df.groupby("mag_bin", observed=True, sort=True).agg(
        median_deaths        =("deaths",          "median"),
        median_damage_millions=("damage_millions", "median"),
        total_events         =("magnitude",        "count"),
        total_deaths         =("deaths",           "sum"),
    )
    grouped["pct_with_deaths"] = (
        df.groupby("mag_bin", observed=True, sort=True)["deaths"]
        .apply(lambda s: (s > 0).mean())
    )
    result = grouped.reset_index()
    result["mag_bin"] = result["mag_bin"].astype(str)
    return result


def deadly_threshold(df: pd.DataFrame, death_cutoff: int = 10) -> float:
    """
    Return the magnitude at which 50%+ of earthquakes with impact data
    caused at least `death_cutoff` deaths.

    Uses a simple rolling median over magnitude-binned data.
    """
    has_data = df[df["deaths"].notna()].copy()
    has_data["mag_bin"] = pd.cut(has_data["magnitude"], bins=np.arange(0, 10.5, 0.5))
    pct_deadly = (
        has_data.groupby("mag_bin", observed=True)["deaths"]
        .apply(lambda s: (s >= death_cutoff).mean())
    )
    above = pct_deadly[pct_deadly >= 0.5]
    if above.empty:
        return None
    return above.index[0].left


# ── Q2: Depth vs. impact ─────────────────────────────────────────────────────

def depth_vs_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare median deaths and damage across depth categories
    (shallow / intermediate / deep), controlling for magnitude by looking
    only at M5+ earthquakes.

    Returns a DataFrame with one row per depth category.
    """
    m5 = df[df["magnitude"] >= 5].copy()
    if "depth_category" not in m5.columns:
        m5["depth_category"] = pd.cut(
            m5["depth"],
            bins=[-float("inf"), 30, 150, float("inf")],
            labels=["shallow", "intermediate", "deep"],
        )
    return (
        m5.groupby("depth_category", observed=True)
        .agg(
            median_deaths        =("deaths",          "median"),
            median_damage_millions=("damage_millions", "median"),
            total_events         =("magnitude",        "count"),
        )
        .reset_index()
    )


# ── Q3: Regional vulnerability ────────────────────────────────────────────────

def regional_impact(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Aggregate total deaths, total damage, and event count by region.

    Also computes deaths_per_event and damage_per_event as vulnerability proxies.

    Returns the top_n regions by total deaths.
    """
    grouped = (
        df.groupby("region")
        .agg(
            total_deaths         =("deaths",          "sum"),
            total_damage_millions=("damage_millions",  "sum"),
            total_events         =("magnitude",        "count"),
            median_magnitude     =("magnitude",        "median"),
        )
        .reset_index()
    )
    grouped["deaths_per_event"] = grouped["total_deaths"] / grouped["total_events"]
    grouped["damage_per_event"] = grouped["total_damage_millions"] / grouped["total_events"]
    return grouped.nlargest(top_n, "total_deaths").reset_index(drop=True)


def vulnerability_index(df: pd.DataFrame, min_events: int = 5) -> pd.DataFrame:
    """
    Identify regions that suffer disproportionately high deaths relative to
    the median magnitude of earthquakes they experience.

    vulnerability_score = deaths_per_event / median_magnitude

    Returns regions with at least `min_events` records, sorted by score.
    """
    grouped = regional_impact(df, top_n=len(df))
    filtered = grouped[grouped["total_events"] >= min_events].copy()
    filtered["vulnerability_score"] = (
        filtered["deaths_per_event"] / filtered["median_magnitude"].replace(0, np.nan)
    )
    return filtered.sort_values("vulnerability_score", ascending=False).reset_index(drop=True)


# ── Q4: Trends over time ─────────────────────────────────────────────────────

def yearly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate deaths, damage, and event count by year.

    Returns a DataFrame with one row per year.
    """
    if "year" not in df.columns:
        raise ValueError("DataFrame must have a 'year' column — run clean_merged() first.")
    return (
        df.groupby("year")
        .agg(
            total_deaths         =("deaths",          "sum"),
            total_damage_millions=("damage_millions",  "sum"),
            total_events         =("magnitude",        "count"),
            median_magnitude     =("magnitude",        "median"),
        )
        .reset_index()
    )


def rolling_average(yearly_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Add rolling-mean columns to a yearly_trends DataFrame.

    Adds: deaths_rolling, damage_rolling (window-year centred means).
    """
    df = yearly_df.copy().sort_values("year")
    df["deaths_rolling"] = (
        df["total_deaths"].rolling(window, center=True, min_periods=1).mean()
    )
    df["damage_rolling"] = (
        df["total_damage_millions"].rolling(window, center=True, min_periods=1).mean()
    )
    return df
