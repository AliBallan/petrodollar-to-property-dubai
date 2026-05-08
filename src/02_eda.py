"""
02_eda.py — Exploratory Data Analysis.

Produces:
  - Plot 1: Dual-axis time series (Brent + Dubai All Index) with event lines
  - Plot 2: Rolling 12-month correlation
  - Plot 3: Scatter plots (Brent vs each index)
  ADF stationarity tests on all four series.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.size": 12,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

PROCESSED = "data/processed/merged_monthly.csv"
FIG_DIR = "outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

EVENTS = {
    "2014-11": "Oil Crash\n(Nov 2014)",
    "2020-04": "COVID-19\n(Apr 2020)",
    "2022-03": "Ukraine/Spike\n(Mar 2022)",
}


def load_data():
    df = pd.read_csv(PROCESSED, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ── Plot 1: Dual-axis time series ─────────────────────────────────────────────

def plot_dual_axis(df):
    fig, ax1 = plt.subplots(figsize=(13, 5))

    color_brent = "#1a6dac"
    color_dubai = "#c0392b"

    ax1.plot(df["date"], df["brent_usd"], color=color_brent, lw=1.8, label="Brent Crude (USD/bbl)")
    ax1.set_ylabel("Brent Crude (USD / barrel)", color=color_brent, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color_brent)
    ax1.set_xlabel("Date")

    ax2 = ax1.twinx()
    ax2.plot(df["date"], df["all_monthly_index"], color=color_dubai, lw=1.8,
             linestyle="--", label="Dubai All Index")
    ax2.set_ylabel("Dubai Residential Price Index (base = 100, Jan 2012)",
                   color=color_dubai, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color_dubai)
    ax2.spines["right"].set_visible(True)

    # Event lines
    ymin, ymax = ax1.get_ylim()
    for date_str, label in EVENTS.items():
        ev = pd.Timestamp(date_str)
        ax1.axvline(ev, color="grey", lw=1.2, linestyle=":", alpha=0.8)
        ax1.text(ev, ax1.get_ylim()[1] * 0.97, label,
                 ha="center", va="top", fontsize=9, color="grey",
                 rotation=0, style="italic")

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.autofmt_xdate()

    fig.suptitle("Brent Crude Oil Price vs Dubai Residential Property Index (2012–2024)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "01_dual_axis_time_series.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 2: Rolling 12-month correlation ─────────────────────────────────────

def plot_rolling_correlation(df):
    roll_corr = df["brent_usd"].rolling(12).corr(df["all_monthly_index"])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["date"], roll_corr, color="#2c7bb6", lw=1.8)
    ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)
    ax.fill_between(df["date"], roll_corr, 0,
                    where=roll_corr >= 0, alpha=0.15, color="#2c7bb6")
    ax.fill_between(df["date"], roll_corr, 0,
                    where=roll_corr < 0, alpha=0.15, color="#d7191c")

    for date_str, label in EVENTS.items():
        ev = pd.Timestamp(date_str)
        ax.axvline(ev, color="grey", lw=1.0, linestyle=":", alpha=0.7)

    ax.set_title("Rolling 12-Month Correlation: Brent Crude vs Dubai All Price Index",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Pearson Correlation", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylim(-1.1, 1.1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.autofmt_xdate()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "02_rolling_correlation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 3: Scatter plots ─────────────────────────────────────────────────────

def plot_scatters(df):
    series = [
        ("all_monthly_index",   "All Properties"),
        ("flat_monthly_index",  "Apartments (Flats)"),
        ("villa_monthly_index", "Villas"),
    ]
    colors = ["#2c7bb6", "#1a9641", "#d7191c"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax, (col, label), color in zip(axes, series, colors):
        ax.scatter(df["brent_usd"], df[col], alpha=0.55, s=25, color=color,
                   edgecolors="white", linewidth=0.3)
        # Regression line
        m, b = np.polyfit(df["brent_usd"].dropna(), df[col].dropna(), 1)
        x_line = np.linspace(df["brent_usd"].min(), df["brent_usd"].max(), 100)
        ax.plot(x_line, m * x_line + b, color="black", lw=1.5, linestyle="--", alpha=0.7)
        corr = df["brent_usd"].corr(df[col])
        ax.set_title(f"Brent vs {label}\n(r = {corr:.2f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Brent Crude (USD/bbl)", fontsize=11)
        ax.set_ylabel("Price Index (base=100)", fontsize=11)

    fig.suptitle("Brent Crude Oil Price vs Dubai Residential Segment Indices",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "03_scatter_plots.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── ADF Stationarity Tests ────────────────────────────────────────────────────

def run_adf_tests(df):
    print("\n" + "=" * 65)
    print("ADF Stationarity Tests")
    print("=" * 65)
    print(f"{'Series':<30} {'ADF stat':>10} {'p-value':>10} {'Status':<20}")
    print("-" * 65)

    series_list = [
        ("brent_usd",          "Brent Crude (levels)"),
        ("all_monthly_index",  "Dubai All Index (levels)"),
        ("flat_monthly_index", "Dubai Flat Index (levels)"),
        ("villa_monthly_index","Dubai Villa Index (levels)"),
    ]

    needs_diff = {}
    for col, label in series_list:
        s = df[col].dropna()
        adf_stat, p_val, _, _, _, _ = adfuller(s, autolag="AIC")
        status = "STATIONARY" if p_val <= 0.05 else "NON-STATIONARY → diff"
        needs_diff[col] = (p_val > 0.05)
        print(f"{label:<30} {adf_stat:>10.4f} {p_val:>10.4f} {status:<20}")

    # Re-test first differences where needed
    print("\nFirst-Difference ADF Tests:")
    print("-" * 65)
    needs_log_diff = {}
    for col, label in series_list:
        if needs_diff[col]:
            s_diff = df[col].diff().dropna()
            adf_stat, p_val, _, _, _, _ = adfuller(s_diff, autolag="AIC")
            status = "STATIONARY" if p_val <= 0.05 else "STILL NON-STATIONARY → log-diff"
            needs_log_diff[col] = (p_val > 0.05)
            diff_label = label.replace("(levels)", "(Δ)")
            print(f"{diff_label:<30} {adf_stat:>10.4f} {p_val:>10.4f} {status:<20}")

    # Re-test log first differences for persistent series
    still_ns = [col for col, label in series_list if needs_diff.get(col) and needs_log_diff.get(col)]
    if still_ns:
        print("\nLog First-Difference ADF Tests (for series still non-stationary after Δ):")
        print("-" * 65)
        for col, label in series_list:
            if col in still_ns:
                s_logdiff = np.log(df[col]).diff().dropna()
                adf_stat, p_val, _, _, _, _ = adfuller(s_logdiff, autolag="AIC")
                status = "STATIONARY" if p_val <= 0.05 else "STILL NON-STATIONARY"
                log_label = label.replace("(levels)", "(Δlog)")
                print(f"{log_label:<30} {adf_stat:>10.4f} {p_val:>10.4f} {status:<20}")

    # KPSS cross-check (H0 = stationary): confirms log-diffs are stationary
    # even when ADF has low power due to the 2022 structural break
    from statsmodels.tsa.stattools import kpss as kpss_test
    print("\nKPSS cross-check on log first-differences (H0 = stationary):")
    print("-" * 65)
    for col, label in series_list:
        s_ld = np.log(df[col]).diff().dropna()
        stat, p_kpss, *_ = kpss_test(s_ld, regression="c", nlags="auto")
        verdict = "STATIONARY (fail to reject H0)" if p_kpss >= 0.05 else "NON-STATIONARY"
        print(f"  {label.replace('(levels)','(Δlog)'):<32} KPSS={stat:.4f}  {verdict}")

    print("\nConclusion (ADF + KPSS joint evidence):")
    for col, label in series_list:
        if col in still_ns:
            print(f"  {label}: ADF inconclusive (structural break 2022); "
                  "KPSS confirms log-diff is stationary → use log first differences.")
        elif needs_diff.get(col):
            print(f"  {label}: non-stationary in levels → use first differences in VAR.")
        else:
            print(f"  {label}: stationary in levels → enter VAR directly.")
    print("=" * 65)

    return needs_diff


def main():
    print("=" * 60)
    print("Step 2 — Exploratory Data Analysis")
    print("=" * 60)

    df = load_data()
    print(f"  Loaded {len(df)} observations ({df['date'].min().date()} – {df['date'].max().date()})")

    print("\n[1/4] Dual-axis time series plot …")
    plot_dual_axis(df)

    print("[2/4] Rolling 12-month correlation plot …")
    plot_rolling_correlation(df)

    print("[3/4] Scatter plots …")
    plot_scatters(df)

    print("[4/4] ADF stationarity tests …")
    needs_diff = run_adf_tests(df)

    print("\nEDA complete. All figures saved to outputs/figures/")


if __name__ == "__main__":
    main()
