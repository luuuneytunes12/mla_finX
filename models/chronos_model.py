"""Chronos-Bolt, used as-is (Owen). Reads ret_index only. A pretrained forecaster (amazon/chronos-bolt-base): no
training on our data, no scaling (the model scales each history itself). For test day t it reads every ret_index value
up to and including day t and forecasts day t+1; the point forecast is the median (0.5 quantile). fit() only loads the
weights, so "train time" is load time. Bolt outputs quantiles directly (no sampling), so forecasts are deterministic."""
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "src")]
import numpy as np, pandas as pd
import config as C
from finx_common import walk_forward, load_master

NAME = "chronos_asis"
MODEL_ID = "amazon/chronos-bolt-base"    # 205M parameters, the largest Bolt; "amazon/chronos-bolt-small" (48M) is lighter
CONTEXT = 2048                           # Bolt's max history; master has 1,435 rows, so the full history always fits
DEVICE = "cpu"                           # CPU, like every other FinX model, so the timings are comparable
QUANTILES = [0.1, 0.5, 0.9]


def fit(train):
    import torch
    from chronos import BaseChronosPipeline      # pip install chronos-forecasting
    torch.manual_seed(C.SEED)
    return BaseChronosPipeline.from_pretrained(MODEL_ID, device_map=DEVICE, torch_dtype=torch.float32)


def predict_quantiles(model, full, test_index):
    """One row per test day t: the 10%, 50% and 90% quantiles of ret_index[t+1], from ret_index up to day t."""
    import torch
    r = full["ret_index"]
    pos = r.index.get_indexer(test_index)
    ctx = [torch.tensor(r.values[max(0, i + 1 - CONTEXT): i + 1], dtype=torch.float32) for i in pos]
    q, _ = model.predict_quantiles(ctx, prediction_length=1, quantile_levels=QUANTILES)
    return pd.DataFrame(q[:, 0, :].numpy(), index=test_index, columns=[f"q{int(x * 100)}" for x in QUANTILES])


def predict(model, full, test_index):
    return predict_quantiles(model, full, test_index)["q50"]


if __name__ == "__main__":
    walk_forward(fit, predict, NAME, df=load_master())
