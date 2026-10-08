"""Shared harness. Model owners write ONLY fit(train) and predict(model, full, test_index)."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

FOLDS = C.FOLDS
LATENCY_SAMPLES = 10
LAG_NAMES = None


def load_master(path=None):
    path = Path(path or C.MASTER_PATH)
    sha = path.with_suffix(".sha256")
    if sha.exists():   # frozen file: stop if anyone changed it
        import hashlib
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha.read_text().split()[0]:
            raise RuntimeError(f"{path.name} changed after it was frozen. Restore it from git.")
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    if sha.exists() and path == Path(C.MASTER_PATH) and len(df) != C.EXPECTED_N_ROWS:
        raise RuntimeError(f"{path.name} has {len(df)} rows, config.EXPECTED_N_ROWS is {C.EXPECTED_N_ROWS}.")
    return df


def hardware():
    import platform, os
    gpu = "none"
    try:
        import subprocess
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip() or "none"
    except Exception:
        pass
    return {"cpu": platform.processor() or platform.machine(), "cores": os.cpu_count(), "gpu": gpu}


def feature_cols(df):
    return [c for c in df.columns if c != C.TARGET]


def make_window(df, n=C.WINDOW):
    """Flat table: last n days of every feature -> shape (rows, n*58). Row t uses days t-n+1..t.
    Returns (X, y) aligned on the same index. Rows without a full window are dropped."""
    feats = df[feature_cols(df)]
    parts = {f"{c}_lag{k}": feats[c].shift(k) for k in range(n) for c in feats.columns}
    X = pd.DataFrame(parts, index=df.index).dropna()
    return X, df.loc[X.index, C.TARGET]


def make_window_3d(df, n=C.WINDOW):
    """Array (rows, n, 58) for LSTM, oldest day first."""
    X, y = make_window(df, n)
    f = len(feature_cols(df))
    arr = X.values.reshape(len(X), n, f)[:, ::-1, :]   # columns are lag-major; flip so oldest day is first
    return arr, y


def fold_split(df, fold):
    name, a, b = fold
    train = df[df.index < pd.Timestamp(a)]
    test = df[(df.index >= pd.Timestamp(a)) & (df.index <= pd.Timestamp(b))]
    return train, test


def val_split(train, frac=C.VAL_FRACTION):
    """Last 10% of the training period = validation slice (early stopping). No shuffling."""
    k = int(len(train) * (1 - frac))
    return train.iloc[:k], train.iloc[k:]


def direction_pct(pred, actual):
    m = actual != 0
    return 100 * (np.sign(pred[m]) == np.sign(actual[m])).mean()


def score(actual, pred, naive_rmse=None):
    err = actual - pred
    rmse = float(np.sqrt((err ** 2).mean()))
    return {"RMSE": rmse, "MAE": float(err.abs().mean()),
            "Direction %": float(direction_pct(pred, actual)),
            "Skill vs naive": (1 - rmse / naive_rmse) if naive_rmse else 0.0}


def walk_forward(fit, predict, name, df=None, out_dir=None, verbose=True):
    """fit(train_df) -> model ; predict(model, full_df, test_index) -> Series indexed by test_index.
    Saves preds_<name>.csv and timing_<name>.json. Times training and per-day latency."""
    df = df if df is not None else load_master()
    out = Path(out_dir or C.OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    preds, timing = [], {"train_s": {}, "latency_ms_per_day": {}, "hardware": hardware()}
    for fold in FOLDS:
        train, test = fold_split(df, fold)
        if len(test) == 0:
            continue
        t0 = time.perf_counter()
        model = fit(train)
        timing["train_s"][fold[0]] = time.perf_counter() - t0
        p = predict(model, df, test.index)          # all test days (used for scoring)
        # Latency = time for ONE next-day prediction, using only data up to that day.
        # Median over LATENCY_SAMPLES evenly spaced test days.
        days = test.index[np.linspace(0, len(test) - 1, LATENCY_SAMPLES).astype(int)]
        times = []
        for d in days:
            t0 = time.perf_counter()
            predict(model, df.loc[:d], pd.DatetimeIndex([d]))
            times.append((time.perf_counter() - t0) * 1000)
        timing["latency_ms_per_day"][fold[0]] = float(np.median(times))
        p = pd.Series(np.asarray(p, dtype=float), index=test.index)
        preds.append(pd.DataFrame({"fold": fold[0], "actual": test[C.TARGET], "pred": p,
                                   "VIX": test["VIX"]}))
        if verbose:
            print(f"[{name}] fold {fold[0]}: train {len(train)} days, test {len(test)} days, "
                  f"train {timing['train_s'][fold[0]]:.1f}s, {timing['latency_ms_per_day'][fold[0]]:.3f} ms for one prediction")
    P = pd.concat(preds)
    P.to_csv(out / f"preds_{name}.csv")
    (out / f"timing_{name}.json").write_text(json.dumps(timing, indent=2))
    return P, timing
