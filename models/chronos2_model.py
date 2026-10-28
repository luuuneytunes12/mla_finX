"""Chronos-2, used as-is (Owen). Reads ret_index only. A pretrained forecaster (amazon/chronos-2, 120M), the successor
to Chronos-Bolt: no training on our data, no scaling (the model scales each history itself). For test day t it reads
every ret_index value up to and including day t and forecasts day t+1; the point forecast is the median (0.5 quantile).
fit() only loads the weights, so "train time" is load time. Quantiles come out directly (no sampling), so forecasts are
deterministic."""
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "src")]
import numpy as np, pandas as pd
import config as C
from finx_common import walk_forward, load_master

NAME = "chronos2"
MODEL_ID = "amazon/chronos-2"            # 120M parameters
CONTEXT = 8192                           # Chronos-2's max history; master has 1,435 rows, so the full history always fits
DEVICE = "cpu"                           # CPU, like every other FinX model, so the timings are comparable
QUANTILES = [0.1, 0.5, 0.9]


def fit(train):
    import torch
    from chronos import BaseChronosPipeline      # pip install chronos-forecasting (2.x)
    torch.manual_seed(C.SEED)
    return BaseChronosPipeline.from_pretrained(MODEL_ID, device_map=DEVICE, torch_dtype=torch.float32)


def predict_quantiles(model, full, test_index):
    """One row per test day t: the 10%, 50% and 90% quantiles of ret_index[t+1], from ret_index up to day t."""
    r = full["ret_index"].values.astype("float32")
    pos = full.index.get_indexer(test_index)
    q, _ = model.predict_quantiles([r[max(0, i + 1 - CONTEXT): i + 1] for i in pos],
                                   prediction_length=1, quantile_levels=QUANTILES)
    arr = np.stack([x[0, 0, :].numpy() for x in q])   # each item: (target, step, quantile)
    return pd.DataFrame(arr, index=test_index, columns=[f"q{int(x * 100)}" for x in QUANTILES])


def predict(model, full, test_index):
    return predict_quantiles(model, full, test_index)["q50"]


if __name__ == "__main__":
    walk_forward(fit, predict, NAME, df=load_master())
