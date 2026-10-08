"""Shared harness. Model owners write ONLY fit(train) and predict(model, full, test_index)."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

FOLDS = C.FOLDS
LAG_NAMES = None


def load_master(path=None):
    df = pd.read_csv(path or C.MASTER_PATH, index_col="Date", parse_dates=True)
    return df


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
    arr = X.values.reshape(len(X), f, n)[:, :, ::-1].transpose(0, 2, 1)
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
    preds, timing = [], {"train_s": {}, "latency_ms_per_day": {}}
    for fold in FOLDS:
        train, test = fold_split(df, fold)
        if len(test) == 0:
            continue
        t0 = time.perf_counter()
        model = fit(train)
        timing["train_s"][fold[0]] = time.perf_counter() - t0
        t0 = time.perf_counter()
        p = predict(model, df, test.index)
        timing["latency_ms_per_day"][fold[0]] = (time.perf_counter() - t0) * 1000 / len(test)
        p = pd.Series(np.asarray(p, dtype=float), index=test.index)
        preds.append(pd.DataFrame({"fold": fold[0], "actual": test[C.TARGET], "pred": p,
                                   "VIX": test["VIX"]}))
        if verbose:
            print(f"[{name}] fold {fold[0]}: train {len(train)} days, test {len(test)} days, "
                  f"train {timing['train_s'][fold[0]]:.1f}s, {timing['latency_ms_per_day'][fold[0]]:.3f} ms/day")
    P = pd.concat(preds)
    P.to_csv(out / f"preds_{name}.csv")
    (out / f"timing_{name}.json").write_text(json.dumps(timing, indent=2))
    return P, timing
