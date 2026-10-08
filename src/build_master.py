"""Raw -> silver -> gold.

  data/raw/yahoo_ohlcv.csv      Yahoo download, untouched (downloaded once; later runs read this file)
  data/silver/features.csv      58 cleaned inputs for every trading day (returns, VIX, indicators)
  data/gold/master.csv          silver + target_next_ret, frozen. The only file models read.

Run:  python src/build_master.py            (offline once data/raw/yahoo_ohlcv.csv exists)
      python src/build_master.py --refresh  (download from Yahoo again; changes the numbers!)
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C


# ---------- indicators (standard formulas, same as the `ta` library) ----------
def rsi(close, n=14):
    d = close.diff()
    up, dn = d.clip(lower=0), (-d).clip(lower=0)
    au = up.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    ad = dn.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    return 100 - 100 / (1 + au / ad)


def macd_diff(close, fast=12, slow=26, sig=9):
    line = close.ewm(span=fast, min_periods=fast, adjust=False).mean() - \
           close.ewm(span=slow, min_periods=slow, adjust=False).mean()
    return line - line.ewm(span=sig, min_periods=1, adjust=False).mean()


def bb_pctb(close, n=20, k=2):
    m, s = close.rolling(n).mean(), close.rolling(n).std(ddof=0)
    return (close - (m - k * s)) / (2 * k * s)


# ---------- raw ----------
def load_raw(refresh=False):
    """Read data/raw/yahoo_ohlcv.csv. Download from Yahoo only if it is missing or --refresh is given."""
    f = C.RAW_PATH
    if f.exists() and not refresh:
        raw = pd.read_csv(f, header=[0, 1], index_col=0, parse_dates=True)
        raw.columns.names = [None, "Ticker"]
        print("Using saved raw data:", f.relative_to(C.ROOT))
        return raw
    import yfinance as yf
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    tickers = list(C.TICKER_WEIGHTS) + [C.INDEX_TICKER, C.VIX_TICKER]
    raw = yf.download(tickers, start=C.DOWNLOAD_START, end=C.END_DATE, auto_adjust=False,
                      actions=False, group_by="column", progress=False, threads=True)
    raw.to_csv(f)
    print("Downloaded from Yahoo and saved", f.relative_to(C.ROOT))
    return raw


# ---------- build ----------
def pick_top50(raw):
    close = raw["Close"]
    first = close.apply(lambda s: s.first_valid_index())
    ranked = sorted(C.TICKER_WEIGHTS, key=lambda t: -C.TICKER_WEIGHTS[t])
    sib = {a: b for a, b in C.SHARE_CLASS_DUPES} | {b: a for a, b in C.SHARE_CLASS_DUPES}
    top = []
    for t in ranked:
        if sib.get(t) in top or t not in close or pd.isna(first.get(t)):
            continue
        if first[t] > pd.Timestamp(C.START_DATE) + pd.Timedelta(days=7):   # must have prices for the whole study period
            continue
        top.append(t)
        if len(top) == C.TOP_N:
            break
    return top


def build_silver(raw):
    """58 inputs for every trading day from START_DATE to the last downloaded day. No target."""
    # Yahoo has VIX rows on US market holidays (equities blank). Keep only days the S&P 500 traded,
    # otherwise dropna() would delete the real trading day before each holiday.
    raw = raw[raw["Close"][C.INDEX_TICKER].notna()]
    top = pick_top50(raw)
    idx_c = raw["Close"][C.INDEX_TICKER]
    df = pd.DataFrame(index=raw.index)
    df["ret_index"] = idx_c.pct_change(fill_method=None)
    for t in top:
        df[f"{t}_ret"] = raw["Adj Close"][t].pct_change(fill_method=None)
    vix = raw["Close"][C.VIX_TICKER]
    df["VIX"] = vix
    df["VIX_chg"] = vix.pct_change(fill_method=None)
    df["RSI_14"] = rsi(idx_c)
    df["MACD_diff"] = macd_diff(idx_c)
    df["BB_pctB"] = bb_pctb(idx_c)
    # Yahoo shows 0 volume on 2023-05-24 (glitch): carry the last real volume forward instead of dropping days
    df["vol_chg"] = raw["Volume"][C.INDEX_TICKER].replace(0, np.nan).ffill().pct_change(fill_method=None)
    df["hl_range"] = (raw["High"][C.INDEX_TICKER] - raw["Low"][C.INDEX_TICKER]) / idx_c
    df = df.dropna().loc[C.START_DATE:]        # drop the indicator warm-up rows
    lost = raw.loc[C.START_DATE:].index.difference(df.index)
    assert len(lost) == 0, f"{len(lost)} trading days lost to blanks, e.g. {list(lost[:3].date)}: check the download"
    df.index.name = "Date"
    return df, top


def build_gold(silver):
    """Add tomorrow's return on today's row; the last day has no tomorrow, so it ends on LAST_ROW_DATE."""
    df = silver.copy()
    df[C.TARGET] = df["ret_index"].shift(-1)
    return df.dropna().loc[:C.LAST_ROW_DATE]


def build(raw):
    silver, top = build_silver(raw)
    return build_gold(silver), top


def run_checks(df, top, real=False, top50_file=None):
    feats = [c for c in df.columns if c != C.TARGET]
    assert df.shape[1] == C.EXPECTED_N_FEATURES + 1, f"expected 59 cols, got {df.shape[1]}"
    assert not df.isna().any().any(), "blank cells found"
    assert df.index.is_monotonic_increasing and df.index.is_unique, "dates not sorted/unique"
    assert np.allclose(df[C.TARGET].iloc[:-1].values, df["ret_index"].shift(-1).iloc[:-1].values), \
        "target[t] != ret_index[t+1]"
    assert [c[:-4] for c in feats if c.endswith("_ret") and c != "ret_index"] == top, "stock cols != top50 list"
    if top50_file is not None:
        assert pd.read_csv(top50_file)["Symbol"].tolist() == top, "stock cols != top50_tickers.csv"
    r = df["ret_index"] * 100
    if real:   # check 7: no spare replaced a top-50 stock (the Overview's fixed list, Alphabet once)
        assert top == [t for t in list(C.TICKER_WEIGHTS)[:51] if t != "GOOG"], "a spare replaced a top-50 stock"
    if real:   # check 6: period is exactly the Overview's 2021-01-04 to 2026-09-21
        assert df.index.min() == pd.Timestamp(C.START_DATE) and df.index.max() == pd.Timestamp(C.LAST_ROW_DATE), \
            f"period {df.index.min().date()} to {df.index.max().date()} != {C.START_DATE} to {C.LAST_ROW_DATE}"
    if real:   # check 5: summary stats close to the Doc (mean 0.056%, sd 1.04%, ~54% up days)
        assert 0.0 < r.mean() < 0.12 and 0.9 < r.std() < 1.2 and 51 < (r > 0).mean() * 100 < 57, \
            "summary stats far from the Doc: check the download"
    print(f"CHECKS PASSED. Shape {df.shape}; {df.index.min().date()} to {df.index.max().date()}")
    print(f"ret_index mean {r.mean():.3f}%  sd {r.std():.3f}%  up days {(r > 0).mean()*100:.1f}%  "
          f"VIX>= {C.VIX_STRESS}: {(df['VIX'] >= C.VIX_STRESS).mean()*100:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="download from Yahoo again (numbers will change)")
    a = ap.parse_args()
    raw = load_raw(refresh=a.refresh)
    silver, top = build_silver(raw)
    df = build_gold(silver)
    run_checks(df, top, real=True)
    C.SILVER_DIR.mkdir(parents=True, exist_ok=True)
    C.GOLD_DIR.mkdir(parents=True, exist_ok=True)
    silver.to_csv(C.SILVER_PATH)
    pd.DataFrame({"Rank": range(1, len(top) + 1), "Symbol": top,
                  "Weight (%)": [round(C.TICKER_WEIGHTS[t] * 100, 2) for t in top]}
                 ).to_csv(C.TOP50_PATH, index=False)
    run_checks(df, top, real=True, top50_file=C.TOP50_PATH)   # check 4 against the saved file
    import hashlib
    mf = C.MASTER_PATH
    if mf.exists():
        mf.chmod(0o644)
    df.to_csv(mf)
    mf.with_suffix(".sha256").write_text(hashlib.sha256(mf.read_bytes()).hexdigest() + f"  rows={len(df)}\n")
    mf.chmod(0o444)    # read-only: frozen
    print("Saved", C.SILVER_PATH.relative_to(C.ROOT), "and", mf.relative_to(C.ROOT), "(frozen: do not edit)")


if __name__ == "__main__":
    main()
