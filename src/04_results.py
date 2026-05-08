"""
04_results.py — Publication-quality summary figure and final diagnostics.

Combines all three IRFs into one comparison chart (main_result.png).
Prints final summary with observation count, p-values, and file list.
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR

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
VAR_SUMMARY = "data/processed/var_summary.csv"
FIG_DIR = "outputs/figures"
MAX_LAGS = 8
IRF_HORIZON = 12


def make_stationary(series):
    _, p, *_ = adfuller(series.dropna(), autolag="AIC")
    if p <= 0.05:
        return series
    s_diff = series.diff().dropna()
    _, p2, *_ = adfuller(s_diff, autolag="AIC")
    if p2 <= 0.05:
        return series.diff()
    return np.log(series).diff()


def load_and_diff():
    df = pd.read_csv(PROCESSED, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    result = {}
    for col in ["brent_usd", "all_monthly_index", "flat_monthly_index", "villa_monthly_index"]:
        result[f"{col}_diff"] = make_stationary(df[col]).iloc[1:].reset_index(drop=True)

    return pd.DataFrame(result)


def get_irf_data(diff_df, brent_col, index_col):
    data = diff_df[[brent_col, index_col]].dropna()
    model = VAR(data)
    lag_results = model.select_order(MAX_LAGS)
    opt_lag = max(lag_results.aic, 1)
    fitted = model.fit(opt_lag)

    gc_results = grangercausalitytests(
        data[[index_col, brent_col]], maxlag=opt_lag, verbose=False
    )
    gc_pval = gc_results[opt_lag][0]["ssr_ftest"][1]

    irf = fitted.irf(IRF_HORIZON)
    irf_mean = irf.irfs[:, 1, 0]

    try:
        stderr = irf.stderr(orth=False)
        irf_lower = irf_mean - 1.96 * stderr[:, 1, 0]
        irf_upper = irf_mean + 1.96 * stderr[:, 1, 0]
    except Exception:
        std_val = np.std(irf_mean)
        irf_lower = irf_mean - 1.96 * std_val
        irf_upper = irf_mean + 1.96 * std_val

    # FEVD at 12-month horizon
    # decomp shape: (neqs, periods, neqs) → decomp[eq, h-1, var]
    fevd = fitted.fevd(IRF_HORIZON)
    n_periods = fevd.decomp.shape[1]
    h12_idx = min(11, n_periods - 1)
    oil_share_12 = fevd.decomp[1, h12_idx, 0] * 100

    return {
        "opt_lag": opt_lag,
        "gc_pval": gc_pval,
        "irf_mean": irf_mean,
        "irf_lower": irf_lower,
        "irf_upper": irf_upper,
        "oil_share_12": oil_share_12,
    }


def plot_main_result(segments_data):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    periods = np.arange(IRF_HORIZON + 1)
    colors = ["#1a6dac", "#2ca25f", "#d7191c"]
    labels = ["All Properties", "Apartments (Flats)", "Villas"]

    for ax, seg, color, label in zip(axes, segments_data, colors, labels):
        irf_mean  = seg["irf_mean"]
        irf_lower = seg["irf_lower"]
        irf_upper = seg["irf_upper"]
        gc_pval   = seg["gc_pval"]
        oil_share = seg["oil_share_12"]
        opt_lag   = seg["opt_lag"]

        ax.plot(periods, irf_mean, color=color, lw=2.2, zorder=3)
        ax.fill_between(periods, irf_lower, irf_upper,
                        alpha=0.20, color=color, zorder=2)
        ax.axhline(0, color="black", lw=1.0, linestyle="--", alpha=0.5)

        # Significance annotation
        sig_str = "p < 0.01**" if gc_pval < 0.01 else \
                  ("p < 0.05*" if gc_pval < 0.05 else f"p = {gc_pval:.3f}")

        ax.set_title(f"{label}\n(VAR lag={opt_lag}, {sig_str})",
                     fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Months after oil shock", fontsize=11)
        ax.set_xticks(periods[::2])

        # Oil-share annotation box
        ax.text(0.97, 0.97,
                f"Oil explains\n{oil_share:.1f}% at h=12",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color=color,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8))

    axes[0].set_ylabel("Response of Dubai Price Index\n(index pts, first-differenced)", fontsize=11)

    fig.suptitle(
        "Oil Price Shock Transmission to Dubai Residential Property:\n"
        "Impulse Response Functions (VAR, 12-month horizon, 95% CI)",
        fontsize=13, fontweight="bold", y=1.04
    )

    # Add legend for CI shading
    from matplotlib.patches import Patch
    legend_els = [
        plt.Line2D([0], [0], color="black", lw=2, label="IRF point estimate"),
        Patch(facecolor="grey", alpha=0.3, label="95% Confidence Band"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.03), frameon=True)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "main_result.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Main result figure saved: {path}")
    return path


def print_final_summary(df, segments_data):
    n_obs = len(df)
    print("\n" + "=" * 65)
    print("FINAL RESEARCH SUMMARY")
    print("=" * 65)
    print(f"\n  Study window: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Observations in merged dataset: {n_obs} months")

    print("\n  Granger Causality: H0 = 'Brent does NOT cause Dubai prices'")
    print(f"  {'Segment':<25} {'p-value':>10} {'Decision'}")
    print("  " + "─" * 55)

    labels = ["All Properties", "Apartments (Flats)", "Villas"]
    for label, seg in zip(labels, segments_data):
        pval = seg["gc_pval"]
        dec = "Reject H0 (causality found)" if pval < 0.05 else "Fail to reject H0"
        flag = "**" if pval < 0.01 else ("*" if pval < 0.05 else "")
        print(f"  {label:<25} {pval:>10.4f}  {flag}  {dec}")

    # Strongest transmission
    min_seg = min(zip(labels, segments_data), key=lambda x: x[1]["gc_pval"])
    print(f"\n  Strongest oil price transmission segment: {min_seg[0]}")
    print(f"  (p = {min_seg[1]['gc_pval']:.4f}, oil explains {min_seg[1]['oil_share_12']:.1f}% at h=12)")

    print("\n  FEVD — Share of Dubai price variance explained by oil shock:")
    for label, seg in zip(labels, segments_data):
        print(f"    {label}: {seg['oil_share_12']:.1f}% at 12-month horizon")

    print("\n  Output files in outputs/figures/:")
    figs = sorted(glob.glob(os.path.join(FIG_DIR, "*.png")))
    for f in figs:
        size_kb = os.path.getsize(f) / 1024
        print(f"    {os.path.basename(f):<45} ({size_kb:.1f} KB)")

    print("\n  All scripts completed successfully.")
    print("=" * 65)


def main():
    print("=" * 60)
    print("Step 4 — Publication Summary Figure & Final Report")
    print("=" * 60)

    df = pd.read_csv(PROCESSED, parse_dates=["date"])
    diff_df = load_and_diff()

    seg_configs = [
        ("brent_usd_diff", "all_monthly_index_diff"),
        ("brent_usd_diff", "flat_monthly_index_diff"),
        ("brent_usd_diff", "villa_monthly_index_diff"),
    ]

    print("\nComputing IRFs for all three segments …")
    segments_data = []
    for brent_col, index_col in seg_configs:
        data = get_irf_data(diff_df, brent_col, index_col)
        segments_data.append(data)
        print(f"  {index_col.replace('_diff','')}: lag={data['opt_lag']}, Granger p={data['gc_pval']:.4f}")

    print("\nGenerating main result figure …")
    plot_main_result(segments_data)

    print_final_summary(df, segments_data)


if __name__ == "__main__":
    main()
