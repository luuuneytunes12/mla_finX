"""Naive, MA(5) and ARIMA in the walk_forward format (fit, predict)."""
import itertools, warnings
import numpy as np, pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import config as C

warnings.filterwarnings("ignore")


# Naive: tomorrow = 0
naive_fit = lambda train: None
naive_predict = lambda m, full, idx: pd.Series(0.0, index=idx)


# MA(5): tomorrow = mean of the last 5 known returns (today included)
ma_fit = lambda train: None
def ma_predict(m, full, idx, n=C.WINDOW):
    return full["ret_index"].rolling(n).mean().reindex(idx)


# ARIMA: order (p,0,q) chosen by lowest AIC on training returns, parameters then frozen.
def arima_fit(train, max_p=3, max_q=3):
    y = train["ret_index"].values
    best = (np.inf, (0, 0, 0))
    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        try:
            aic = ARIMA(y, order=(p, 0, q), trend="c").fit().aic
            if aic < best[0]:
                best = (aic, (p, 0, q))
        except Exception:
            pass
    fit = ARIMA(y, order=best[1], trend="c").fit()
    return {"fit": fit, "order": best[1], "n_train": len(y)}


def arima_predict(m, full, idx):
    """One-step-ahead forecasts through the test period, parameters fixed (Kalman filter)."""
    hist = full["ret_index"].loc[: idx[-1]]
    applied = m["fit"].apply(hist.values)
    pred = applied.predict(start=m["n_train"], end=len(hist) - 1)
    return pd.Series(pred, index=hist.index[m["n_train"]:]).reindex(idx)
