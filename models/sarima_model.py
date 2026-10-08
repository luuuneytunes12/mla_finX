"""SARIMA reference (reads ret_index only). Weekly cycle 5, p,q in 0-2, P,Q in 0-1, d=D=0, lowest AIC."""
import sys, itertools, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fincare_common import walk_forward, load_master

NAME = "sarima"
warnings.filterwarnings("ignore")


def fit(train):
    y = train["ret_index"].values
    best = (np.inf, None)
    for p, q, P, Q in itertools.product(range(3), range(3), range(2), range(2)):
        try:
            r = SARIMAX(y, order=(p, 0, q), seasonal_order=(P, 0, Q, 5), trend="c").fit(disp=False)
            if r.aic < best[0]:
                best = (r.aic, r)
        except Exception:
            pass
    return {"fit": best[1], "n_train": len(y)}


def predict(model, full, test_index):
    hist = full["ret_index"].loc[: test_index[-1]]
    applied = model["fit"].apply(hist.values)
    p = applied.predict(start=model["n_train"], end=len(hist) - 1)
    return pd.Series(p, index=hist.index[model["n_train"]:]).reindex(test_index)


if __name__ == "__main__":
    walk_forward(fit, predict, NAME, df=load_master())
