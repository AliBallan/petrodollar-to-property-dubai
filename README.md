# Petrodollar to Property: Oil Price Shocks & the Dubai Residential Market

**Research question:** Do Brent crude oil price shocks causally affect Dubai residential property prices, and does the effect differ between mid-market apartments and luxury villas?

I ran a Vector Autoregression (VAR) on 149 months of data from January 2012 to May 2024, using official price indices from the Dubai Land Department and Brent crude prices from FRED.

---

## Data Sources

### Brent Crude Oil Price (USD/barrel)
- **Source:** [FRED — Federal Reserve Bank of St. Louis](https://fred.stlouisfed.org/series/MCOILBRENTEU)
- **Series:** `MCOILBRENTEU` (ICE Brent benchmark, EIA via FRED)
- **Frequency:** Monthly
- **Coverage:** Jan 2012 to May 2024 (full series goes back to May 1987)

### Dubai Residential Sale Price Index
- **Source:** [Dubai Land Department (DLD) via Dubai Pulse](https://www.dubaipulse.gov.ae/)
- **Base period:** January 2012 = 1.00
- **Coverage:** March 2011 to May 2024

The DLD index is built from actual registered sale prices, covering all areas and property types in Dubai. Three monthly sub-indices are used:

| Column | What it covers |
|---|---|
| `all_monthly_index` | All residential property types |
| `flat_monthly_index` | Apartments only |
| `villa_monthly_index` | Villas only |

---

## Methodology

Four scripts run in sequence:

| Step | Script | What it does |
|---|---|---|
| 1 | `src/01_data.py` | Loads both datasets, aligns to monthly frequency, merges, restricts to 2012-01 through 2024-05 |
| 2 | `src/02_eda.py` | Time series plots, rolling correlation, scatter plots, ADF and KPSS stationarity tests |
| 3 | `src/03_var_model.py` | VAR(p) with AIC lag selection, Granger causality F-test, IRF over 12-month horizon, FEVD |
| 4 | `src/04_results.py` | Three-panel IRF comparison figure |

### Stationarity

I ran Augmented Dickey-Fuller (ADF) tests on all four series. Brent first-differences cleanly (ADF p < 0.001). The Dubai indices were harder. First-differences left them borderline non-stationary, ADF p-values around 0.60. The 2022 to 2024 price surge is so sharp it breaks the assumptions the ADF test relies on, which kills its power. I ran the KPSS test as a second check. KPSS tests the opposite null: it assumes stationarity unless the data rejects it. All three Dubai series passed in log first-differences. That is what goes into the VAR.

**Transformations:**
- Brent crude: first difference (Δ)
- Dubai All / Flat / Villa indices: log first difference (Δlog, approximately monthly % return)

### VAR Specification

- Lag length chosen by AIC, up to 8 lags maximum
- Granger causality tested with an F-test. H₀: Brent does not Granger-cause Dubai prices
- IRFs computed over a 12-month horizon with 95% confidence bands
- FEVD reported at horizons 1, 3, 6, and 12 months

---

## Results

### Granger Causality

| Segment | Optimal Lag | Granger p-value | Decision at 5% |
|---|---|---|---|
| All Properties | 2 months | 0.218 | Fail to reject H₀ |
| Apartments | 2 months | 0.294 | Fail to reject H₀ |
| Villas | 3 months | 0.084 | Fail to reject H₀ (borderline at 10%) |

Nothing clears 5%. Villas get to p = 0.084 with a 3-month lag. GCC buyers tend to buy villas, and their wealth tracks oil more directly than the average Dubai resident, so a 3-month lag showing up there makes sense. It still does not cross the significance threshold.

### Forecast Error Variance Decomposition

At each forecast horizon, how much of the price variance is explained by oil shocks vs. the series' own past values:

| Horizon | All Properties | Apartments | Villas |
|---|---|---|---|
| h = 1 month | 0.2% | 0.0% | 0.2% |
| h = 3 months | 0.8% | 0.5% | 2.0% |
| h = 6 months | 1.0% | 0.7% | 2.2% |
| h = 12 months | **1.0%** | **0.7%** | **2.1%** |

At 12 months, oil explains 2.1% of villa price variance and less than 1% for everything else.

---

## What This Tells Us

Oil prices do not significantly predict Dubai residential prices over this period. That contradicts the assumption that Dubai property runs on oil money, but the numbers do not support it.

Dubai's economy is not built on oil. Most of its GDP comes from trade, tourism, logistics, and finance. That is Abu Dhabi. Because Dubai's economy is not directly tied to oil revenue, Brent price swings were not going to have a strong effect on its property market the way they might in a city whose budget depends on the oil price.

The 2022 to 2024 boom makes this even clearer. It was the biggest price run in DLD history, and it had almost nothing to do with oil. It came from a wave of arrivals from Russia, Europe, and South Asia after the Ukraine war, plus the Golden Visa expansion pulling in long-term residents. None of that correlates with Brent crude.

Villas are still the most interesting result. p = 0.084 at a 3-month lag is not significant at the conventional threshold, but it is not noise either. If oil has any path into Dubai property, it is through GCC nationals buying in the luxury segment with wealth that moves with oil. The FEVD numbers cap that effect at 2.1% of price variance at 12 months. Small, but the direction is right.

---

## Figures

| File | Description |
|---|---|
| [`01_dual_axis_time_series.png`](outputs/figures/01_dual_axis_time_series.png) | Brent (left axis) vs Dubai All Index (right axis), 2012 to 2024, with event markers |
| [`02_rolling_correlation.png`](outputs/figures/02_rolling_correlation.png) | Rolling 12-month Pearson correlation, Brent vs All Index |
| [`03_scatter_plots.png`](outputs/figures/03_scatter_plots.png) | Brent vs each sub-index with OLS trend line |
| [`irf_all.png`](outputs/figures/irf_all.png) | IRF: All Properties response to an oil shock |
| [`irf_flat.png`](outputs/figures/irf_flat.png) | IRF: Apartment response to an oil shock |
| [`irf_villa.png`](outputs/figures/irf_villa.png) | IRF: Villa response to an oil shock |
| [`fevd_comparison.png`](outputs/figures/fevd_comparison.png) | FEVD bar chart, oil-shock share across all three segments |
| [`main_result.png`](outputs/figures/main_result.png) | Main figure: IRF comparison, all three segments side by side |

---

## Setup & Reproduction

```bash
pip install -r requirements.txt

python src/01_data.py       # Data loading and processing
python src/02_eda.py        # Exploratory analysis and stationarity tests
python src/03_var_model.py  # VAR, Granger causality, IRF, FEVD
python src/04_results.py    # Summary figure
```

Python 3.9+ required.

---

## Limitations & Next Steps

Granger causality is not the same as structural causality. It tests whether past Brent values help predict Dubai prices in a two-variable system. A structural VAR with sign restrictions, or one that controls for global interest rates and capital flows, would give a cleaner answer.

This analysis also does not test for cointegration. If oil prices and Dubai property levels have a long-run relationship, a Vector Error Correction Model (VECM) is the right framework. Working in first and log-differences removes that long-run information.

The 2022 to 2024 surge is a real issue. That price acceleration was so fast, and so clearly driven by migration and policy rather than oil, that it probably pulls the estimated oil coefficients toward zero. A split-sample analysis, or a Markov-switching VAR that handles regime changes, would separate that period from the rest of the data.

Finally, the DLD index only starts in late 2011. One full oil price cycle is not a lot of data to draw firm conclusions from.

---

*This project was conducted independently by Ali Ballan, a Grade 10 student at Dubai American Academy, Dubai. Initiated May 2026.*
