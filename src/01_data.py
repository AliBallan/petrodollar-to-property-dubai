"""
01_data.py — Data acquisition, alignment, and merging.

Loads:
  - Brent crude: downloaded from FRED (MCOILBRENTEU)
  - Dubai residential price index: official DLD data (data/raw/dubai_price_index.csv)
Merges on monthly date, restricts to 2012-01 → 2024-05.
"""

import os
import requests
import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

STUDY_START = "2012-01-01"
STUDY_END   = "2024-05-01"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


# ── Brent Crude ───────────────────────────────────────────────────────────────

def download_brent():
    path = os.path.join(RAW_DIR, "brent_crude_monthly.csv")
    urls = [
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MCOILBRENTEU",
        "https://eco3min.fr/dataset/brent-crude-oil.csv",
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200 and "," in r.text[:100]:
                with open(path, "wb") as f:
                    f.write(r.content)
                print(f"  Brent downloaded from: {url}")
                return
        except Exception as e:
            print(f"  Brent URL failed ({url}): {e}")


def load_brent():
    path = os.path.join(RAW_DIR, "brent_crude_monthly.csv")
    if not os.path.exists(path):
        download_brent()

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "brent_usd"})
    df["date"] = pd.to_datetime(df["date"])
    df["brent_usd"] = pd.to_numeric(df["brent_usd"], errors="coerce")
    df = df.dropna(subset=["brent_usd"])
    df = df.set_index("date").resample("MS").mean()
    df.index.name = "date"
    return df.reset_index()


# ── Dubai Price Index (real DLD data) ─────────────────────────────────────────

def load_dubai():
    path = os.path.join(RAW_DIR, "dubai_price_index.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. "
            "Download the DLD residential sale price index CSV and save it there."
        )

    df = pd.read_csv(path)

    # Identify date column
    date_col = next(c for c in df.columns if "date" in c.lower() or "month" in c.lower())
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])

    # Keep only the three monthly index columns we need
    required = {
        "all_monthly_index":   ["all_monthly_index"],
        "flat_monthly_index":  ["flat_monthly_index"],
        "villa_monthly_index": ["villa_monthly_index"],
    }
    col_map = {}
    for target, candidates in required.items():
        for c in df.columns:
            if c.lower() in candidates or c == target:
                col_map[c] = target
                break

    df = df.rename(columns=col_map)
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Could not find '{col}' in {list(df.columns)}")

    keep = ["date"] + list(required.keys())
    df = df[keep].copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=list(required.keys()))

    # DLD index is in decimal form (1.0 = base). Scale to 100 for readability.
    for col in required:
        df[col] = df[col] * 100.0

    df = df.set_index("date").resample("MS").mean()
    df.index.name = "date"
    return df.reset_index()


# ── Merge & Save ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Step 1 — Data Acquisition & Processing")
    print("=" * 60)

    print("\n[1/4] Loading Brent crude oil data …")
    download_brent()
    brent = load_brent()
    print(f"  Brent: {len(brent)} months, "
          f"{brent['date'].min().date()} – {brent['date'].max().date()}")

    print("\n[2/4] Loading Dubai residential price index (DLD) …")
    dubai = load_dubai()
    print(f"  Dubai: {len(dubai)} months, "
          f"{dubai['date'].min().date()} – {dubai['date'].max().date()}")

    print("\n[3/4] Merging and restricting to study window …")
    merged = pd.merge(brent, dubai, on="date", how="inner")
    merged = merged[
        (merged["date"] >= STUDY_START) & (merged["date"] <= STUDY_END)
    ].reset_index(drop=True)

    print("\n[4/4] Saving processed data …")
    out_path = os.path.join(PROCESSED_DIR, "merged_monthly.csv")
    merged.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"  Shape : {merged.shape}")
    print(f"  Columns: {list(merged.columns)}")
    print(f"  Window : {merged['date'].min().date()} → {merged['date'].max().date()}")
    print(f"\nFirst 5 rows:")
    print(merged.head().to_string(index=False))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
