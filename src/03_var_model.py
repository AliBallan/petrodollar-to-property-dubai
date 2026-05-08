"""
03_var_model.py — VAR models, Granger causality, IRF, and FEVD.

Runs the full analysis three times:
  (A) Brent vs Dubai All Index
  (B) Brent vs Dubai Flat (Apartment) Index
  (C) Brent vs Dubai Villa Index
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.size": 11,
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
MAX_LAGS = 8
IRF_HORIZON = 12


def make_stationary(series):
    """Return a stationary transformation of series plus a label describing it."""
    _, p, *_ = adfuller(series.dropna(), autolag="AIC")
    if p <= 0.05:
        return series, "levels"
    # Try first difference
    s_diff = series.diff().dropna()
    _, p2, *_ = adfuller(s_diff, autolag="AIC")
    if p2 <= 0.05:
        return series.diff(), "first-diff"
    # Fall back to log first difference
    return np.log(series).diff(), "log-diff"


def load_and_diff():
    df = pd.read_csv(PROCESSED, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    result = {}
    transforms = {}
    for col in ["brent_usd", "all_monthly_index", "flat_monthly_index", "villa_monthly_index"]:
        s_transformed, label = make_stationary(df[col])
        result[f"{col}_diff"] = s_transformed.iloc[1:].reset_index(drop=True)
        transforms[col] = label

    diff_df = pd.DataFrame(result)
    print(f"  Transformations applied:")
    for col, label in transforms.items():
        print(f"    {col}: {label}")
    return diff_df


# ── VAR Analysis ──────────────────────────────────────────────────────────────

def run_var_analysis(diff_df, brent_col, index_col, segment_name, segment_short):
    print(f"\n{'─'*60}")
    print(f"Segment: {segment_name}")
    print(f"{'─'*60}")

    data = diff_df[[brent_col, index_col]].dropna()

    # ── Lag selection via AIC
    model = VAR(data)
    lag_results = model.select_order(MAX_LAGS)
    opt_lag = lag_results.aic
    opt_lag = max(opt_lag, 1)
    print(f"  Optimal lag (AIC): {opt_lag}")

    # ── Fit VAR(p)
    fitted = model.fit(opt_lag)

    # ── Granger causality
    # H0: brent does NOT Granger-cause Dubai prices
    gc_results = grangercausalitytests(
        data[[index_col, brent_col]], maxlag=opt_lag, verbose=False
    )
    # Use F-test p-value at optimal lag
    gc_pval = gc_results[opt_lag][0]["ssr_ftest"][1]
    conclusion = "REJECT H0 (Brent Granger-causes Dubai)" if gc_pval < 0.05 else "FAIL TO REJECT H0"
    print(f"  Granger causality p-value (lag {opt_lag}): {gc_pval:.4f}  →  {conclusion}")

    # ── IRF
    irf = fitted.irf(IRF_HORIZON)
    plot_irf(irf, segment_name, segment_short, brent_col, index_col, opt_lag, gc_pval)

    # ── FEVD
    fevd = fitted.fevd(IRF_HORIZON)
    fevd_table = print_fevd(fevd, index_col, segment_name)

    return {
        "segment": segment_name,
        "short": segment_short,
        "opt_lag": opt_lag,
        "gc_pval": gc_pval,
        "conclusion": conclusion,
        "fevd_table": fevd_table,
        "fitted": fitted,
        "irf": irf,
        "data_cols": (brent_col, index_col),
    }


def plot_irf(irf, segment_name, segment_short, brent_col, index_col, opt_lag, gc_pval):
    periods = np.arange(IRF_HORIZON + 1)

    # IRF: response of index_col to a shock in brent_col
    # In statsmodels VAR, columns are in the order of the input data
    # data was [brent_col, index_col], so impulse=0 (brent), response=1 (index)
    impulse_idx = 0
    response_idx = 1

    irf_mean = irf.irfs[:, response_idx, impulse_idx]
    try:
        stderr = irf.stderr(orth=False)
        irf_lower = irf_mean - 1.96 * stderr[:, response_idx, impulse_idx]
        irf_upper = irf_mean + 1.96 * stderr[:, response_idx, impulse_idx]
    except Exception:
        std_val = np.std(irf_mean)
        irf_lower = irf_mean - 1.96 * std_val
        irf_upper = irf_mean + 1.96 * std_val

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(periods, irf_mean, color="#1a6dac", lw=2, label="IRF (point estimate)")
    ax.fill_between(periods, irf_lower, irf_upper, alpha=0.18, color="#1a6dac",
                    label="95% CI")
    ax.axhline(0, color="black", lw=0.9, linestyle="--", alpha=0.6)
    ax.set_title(
        f"IRF: Response of {segment_name} Price Index\nto a Brent Oil Price Shock\n"
        f"(VAR lag={opt_lag}, Granger p={gc_pval:.3f})",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Months after shock", fontsize=11)
    ax.set_ylabel("Response (index points, differenced)", fontsize=11)
    ax.set_xticks(periods)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"irf_{segment_short}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  IRF saved: {path}")


def print_fevd(fevd, index_col, segment_name):
    # statsmodels FEVD decomp shape: (neqs, periods, neqs)
    # decomp[eq_idx, h-1, var_idx]
    # Dubai equation = idx 1, oil (brent) variable = idx 0
    decomp = fevd.decomp
    n_periods = decomp.shape[1]
    horizons = [1, 3, 6, 12]
    rows = []
    print(f"\n  FEVD — % of {segment_name} variance explained by oil shock:")
    print(f"  {'Horizon':<10} {'Oil (%)':<12} {'Own (%)':<12}")
    print(f"  {'─'*35}")
    for h in horizons:
        h_idx = min(h - 1, n_periods - 1)
        oil_share = decomp[1, h_idx, 0] * 100
        own_share = decomp[1, h_idx, 1] * 100
        print(f"  {h:<10} {oil_share:<12.2f} {own_share:<12.2f}")
        rows.append({"horizon": h, "oil_pct": oil_share, "own_pct": own_share})
    return pd.DataFrame(rows)


def plot_fevd_bars(results_list):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    horizons = [1, 3, 6, 12]
    colors_oil = ["#1a6dac", "#2ca25f", "#d7191c"]

    for ax, res, color in zip(axes, results_list, colors_oil):
        ft = res["fevd_table"]
        oil_vals = ft["oil_pct"].values
        own_vals = ft["own_pct"].values
        x = np.arange(len(horizons))
        width = 0.4
        ax.bar(x - width/2, oil_vals, width, label="Oil shock", color=color, alpha=0.85)
        ax.bar(x + width/2, own_vals, width, label="Own shock", color="lightgrey",
               edgecolor="grey", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"h={h}" for h in horizons])
        ax.set_title(res["segment"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Horizon (months)")
        ax.legend(fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel("% Variance Explained", fontsize=11)

    fig.suptitle("Forecast Error Variance Decomposition (FEVD): Share of Dubai Price Variance\nExplained by Oil Shock",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fevd_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  FEVD comparison chart saved: {path}")


def print_summary_table(results_list):
    print("\n" + "=" * 70)
    print("VAR RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Segment':<20} {'Opt.Lag':>8} {'Granger p':>12} {'Conclusion'}")
    print("─" * 70)
    for res in results_list:
        sig = "**" if res["gc_pval"] < 0.01 else ("*" if res["gc_pval"] < 0.05 else "")
        print(f"{res['segment']:<20} {res['opt_lag']:>8} {res['gc_pval']:>12.4f}  {sig}  {res['conclusion'][:35]}")
    print("─" * 70)
    print("  ** p<0.01   * p<0.05")

    # Strongest transmission
    min_pval = min(results_list, key=lambda r: r["gc_pval"])
    print(f"\n  Strongest oil price transmission: {min_pval['segment']} (p = {min_pval['gc_pval']:.4f})")
    print("=" * 70)


def main():
    print("=" * 60)
    print("Step 3 — VAR Models, Granger Causality, IRF & FEVD")
    print("=" * 60)

    diff_df = load_and_diff()
    print(f"  Working data: {len(diff_df)} observations (first-differenced where needed)")

    segments = [
        ("brent_usd_diff", "all_monthly_index_diff",   "All Properties",       "all"),
        ("brent_usd_diff", "flat_monthly_index_diff",  "Apartments (Flats)",   "flat"),
        ("brent_usd_diff", "villa_monthly_index_diff", "Villas",               "villa"),
    ]

    results_list = []
    for brent_col, index_col, name, short in segments:
        res = run_var_analysis(diff_df, brent_col, index_col, name, short)
        results_list.append(res)

    print("\n[Final] FEVD comparison chart …")
    plot_fevd_bars(results_list)

    print_summary_table(results_list)

    # Save results for 04_results.py
    summary = pd.DataFrame([{
        "segment": r["segment"],
        "short": r["short"],
        "opt_lag": r["opt_lag"],
        "gc_pval": r["gc_pval"],
    } for r in results_list])
    summary.to_csv("data/processed/var_summary.csv", index=False)
    print("\n  VAR summary saved to data/processed/var_summary.csv")


if __name__ == "__main__":
    main()
