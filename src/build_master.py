"""Build and freeze data/master_v1.csv.

Real data:      python src/build_master.py
Offline test:   python src/build_master.py --synthetic   (fake data, tests only)
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


# ---------- download ----------
def download_raw():
    import yfinance as yf
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    f = C.RAW_DIR / "yahoo_snapshot.pkl"
    if f.exists():
        print("Using saved Yahoo snapshot:", f)
        return pd.read_pickle(f)
    tickers = list(C.TICKER_WEIGHTS) + [C.INDEX_TICKER, C.VIX_TICKER]
    raw = yf.download(tickers, start=C.START_DATE, end=C.END_DATE, auto_adjust=False,
                      actions=False, group_by="column", progress=False, threads=True)
    raw.to_pickle(f)
    return raw


def synthetic_raw(seed=0):
    """FAKE market data with the same shape as the Yahoo download. For tests only."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(C.START_DATE, "2026-09-22")
    n = len(days)
    tickers = list(C.TICKER_WEIGHTS)
    mkt = rng.normal(0.0004, 0.01, n)
    cols = {}
    for t in tickers + [C.INDEX_TICKER]:
        r = mkt * (1.0 if t == C.INDEX_TICKER else rng.uniform(0.6, 1.4)) + rng.normal(0, 0.008, n)
        close = 100 * np.exp(np.cumsum(r))
        hi, lo = close * (1 + abs(rng.normal(0, .004, n))), close * (1 - abs(rng.normal(0, .004, n)))
        for fld, v in [("Adj Close", close), ("Close", close), ("High", hi), ("Low", lo),
                       ("Open", close), ("Volume", rng.uniform(1e6, 2e6, n))]:
            cols[(fld, t)] = v
    vix = 15 + 5 * np.abs(np.sin(np.arange(n) / 60)) + 80 * np.abs(mkt)
    for fld in ["Adj Close", "Close", "High", "Low", "Open", "Volume"]:
        cols[(fld, C.VIX_TICKER)] = vix
    return pd.DataFrame(cols, index=days)


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
        if first[t] > pd.Timestamp(C.START_DATE) + pd.Timedelta(days=7):
            continue
        top.append(t)
        if len(top) == C.TOP_N:
            break
    return top


def build(raw):
    top = pick_top50(raw)
    idx_c = raw["Close"][C.INDEX_TICKER]
    df = pd.DataFrame(index=raw.index)
    df["ret_index"] = idx_c.pct_change()
    for t in top:
        df[f"{t}_ret"] = raw["Adj Close"][t].pct_change()
    vix = raw["Close"][C.VIX_TICKER]
    df["VIX"] = vix
    df["VIX_chg"] = vix.pct_change()
    df["RSI_14"] = rsi(idx_c)
    df["MACD_diff"] = macd_diff(idx_c)
    df["BB_pctB"] = bb_pctb(idx_c)
    df["vol_chg"] = raw["Volume"][C.INDEX_TICKER].replace(0, np.nan).pct_change()
    df["hl_range"] = (raw["High"][C.INDEX_TICKER] - raw["Low"][C.INDEX_TICKER]) / idx_c
    df[C.TARGET] = df["ret_index"].shift(-1)   # tomorrow's return, on today's row
    df = df.dropna()
    df.index.name = "Date"
    return df, top


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
    if real:   # check 5: summary stats close to the Doc (mean 0.056%, sd 1.04%, ~54% up days)
        assert 0.0 < r.mean() < 0.12 and 0.9 < r.std() < 1.2 and 51 < (r > 0).mean() * 100 < 57, \
            "summary stats far from the Doc: check the download"
    print(f"CHECKS PASSED. Shape {df.shape}; {df.index.min().date()} to {df.index.max().date()}")
    print(f"ret_index mean {r.mean():.3f}%  sd {r.std():.3f}%  up days {(r > 0).mean()*100:.1f}%  "
          f"VIX>= {C.VIX_STRESS}: {(df['VIX'] >= C.VIX_STRESS).mean()*100:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="FAKE data, for tests only")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    raw = synthetic_raw() if a.synthetic else download_raw()
    df, top = build(raw)
    run_checks(df, top, real=not a.synthetic)
    out = Path(a.outdir) if a.outdir else (C.DATA_DIR / "synthetic" if a.synthetic else C.DATA_DIR)
    out.mkdir(parents=True, exist_ok=True)
    mf = out / "master_v1.csv"
    if mf.exists():
        mf.chmod(0o644)
    df.to_csv(mf)
    pd.DataFrame({"Rank": range(1, len(top) + 1), "Symbol": top,
                  "Weight (%)": [round(C.TICKER_WEIGHTS[t] * 100, 2) for t in top]}
                 ).to_csv(out / "top50_tickers.csv", index=False)
    run_checks(df, top, real=not a.synthetic, top50_file=out / "top50_tickers.csv")   # check 4 against the saved file
    import hashlib
    (out / "master_v1.sha256").write_text(hashlib.sha256(mf.read_bytes()).hexdigest() + f"  rows={len(df)}\n")
    mf.chmod(0o444)    # read-only: frozen
    print("Saved", out / "master_v1.csv", "(FAKE DATA)" if a.synthetic else "(frozen: do not edit)")


if __name__ == "__main__":
    main()
