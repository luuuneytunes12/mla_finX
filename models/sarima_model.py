"""SARIMA (Hong Rong). Reads ret_index only. Weekly cycle s = 5 trading days, p,q in 0-2, P,Q in 0-1,
d = D = 0 (returns are stationary). Order = lowest AIC on the training years, then parameters are frozen
and rolled forward one day at a time through the test year (no refit inside a fold)."""
import sys, itertools, warnings
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "src")]
import numpy as np, pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from finx_common import walk_forward, load_master

NAME = "sarima"
SEASON = 5                     # one trading week
warnings.filterwarnings("ignore")


def fit(train, max_p=2, max_q=2, max_P=1, max_Q=1):
    """Search the grid by AIC. Fits whose optimiser did not converge are skipped (their AIC is not reliable)."""
    y = train["ret_index"].values
    best, tried = (np.inf, None, None), []
    for p, q, P, Q in itertools.product(range(max_p + 1), range(max_q + 1), range(max_P + 1), range(max_Q + 1)):
        try:
            r = SARIMAX(y, order=(p, 0, q), seasonal_order=(P, 0, Q, SEASON), trend="c").fit(disp=False)
        except Exception:
            continue
        ok = bool(r.mle_retvals.get("converged", True))
        tried.append({"order": (p, 0, q), "seasonal_order": (P, 0, Q, SEASON), "aic": r.aic, "converged": ok})
        if ok and r.aic < best[0]:
            best = (r.aic, r, ((p, 0, q), (P, 0, Q, SEASON)))
    return {"fit": best[1], "order": best[2][0], "seasonal_order": best[2][1], "aic": best[0],
            "n_train": len(y), "grid": pd.DataFrame(tried)}


def predict(model, full, test_index):
    """Row t gets the forecast of ret_index[t+1] made with data up to day t (parameters fixed)."""
    hist = full["ret_index"].loc[: test_index[-1]]
    applied = model["fit"].apply(hist.values)
    p = applied.predict(start=model["n_train"] + 1, end=len(hist))   # positions n_train+1 .. len(hist)
    return pd.Series(p, index=hist.index[model["n_train"]:]).reindex(test_index)


def label(model):
    (p, d, q), (P, D, Q, s) = model["order"], model["seasonal_order"]
    return f"SARIMA({p},{d},{q})({P},{D},{Q},{s})"


if __name__ == "__main__":
    walk_forward(fit, predict, NAME, df=load_master())
