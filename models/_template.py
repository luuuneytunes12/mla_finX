"""Copy this file to models/<yourmodel>_model.py and fill in fit() and predict()."""
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "src")]
import numpy as np, pandas as pd
import config as C
from fincare_common import make_window, val_split, walk_forward, load_master

NAME = "template"          # must be one of: xgboost, dnn, lstm, sarima, chronos_asis, chronos_ft


def fit(train):
    X, y = make_window(train)
    Xtr, ytr = X.iloc[: int(len(X) * (1 - C.VAL_FRACTION))], y.iloc[: int(len(X) * (1 - C.VAL_FRACTION))]
    raise NotImplementedError("train your model on (Xtr, ytr); validate on the last 10%")


def predict(model, full, test_index):
    X, _ = make_window(full)
    return pd.Series(model.predict(X.loc[test_index].values), index=test_index)


if __name__ == "__main__":
    walk_forward(fit, predict, NAME, df=load_master())
