"""Chronos-Bolt, fine-tuned (Owen). Same checkpoint (amazon/chronos-bolt-small), input (ret_index only) and prediction
code as chronos_model.py (as-is); the only difference is a short fine-tune on each fold's training years.
Fine-tune: every training day e with at least MIN_CONTEXT days before it is one example (history ret_index[:e] ->
next value ret_index[e]). Examples whose target falls in the last 10% of the training period (val_split) form the
validation slice; the rest are trained on with the model's own quantile loss. Validation loss is checked every
EVAL_EVERY steps, training stops after PATIENCE checks without improvement, and the best weights are kept (step 0 =
the as-is model, so fine-tuning can only be kept if it lowers validation loss). Training runs on the GPU if there is
one; the fine-tuned model is then moved to the CPU and predicts exactly like the as-is model. Test years are never seen."""
import copy, os, sys
from pathlib import Path
os.environ.setdefault("USE_TF", "0")      # stop transformers importing TensorFlow if it is installed
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "src")]
import numpy as np, pandas as pd
import config as C
import chronos_model as base              # same checkpoint, CONTEXT and predict code as the as-is run
from finx_common import val_split, walk_forward, load_master

NAME = "chronos_ft"
MODEL_ID = base.MODEL_ID                  # amazon/chronos-bolt-small
DEVICE = base.DEVICE                      # prediction device (CPU), so latency is comparable with every model
CONTEXT = base.CONTEXT                    # 2,048 days, Bolt's max history
MIN_CONTEXT = 128                         # shortest history used as a training example (test contexts are 753+ days)
LR = 1e-5
BATCH = 32
MAX_STEPS = 1000
EVAL_EVERY = 25
PATIENCE = 8                              # checks without improvement before stopping (= 200 steps)

predict_quantiles = base.predict_quantiles
predict = base.predict


def _examples(r, ends):
    """(history, next value) pairs: history = r[:e] (last CONTEXT values), target = r[e]."""
    return [(r[max(0, e - base.CONTEXT): e], r[e]) for e in ends]


def _batch(ex, torch, device):
    n = max(len(h) for h, _ in ex)
    ctx = torch.full((len(ex), n), float("nan"))          # left-pad with NaN: the model masks missing values
    for i, (h, _) in enumerate(ex):
        ctx[i, n - len(h):] = torch.from_numpy(h)
    tgt = torch.tensor([[y] for _, y in ex], dtype=torch.float32)
    return ctx.to(device), tgt.to(device)


def _val_loss(model, val, torch, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(val), BATCH):
            ctx, tgt = _batch(val[i: i + BATCH], torch, device)
            total += model(context=ctx, target=tgt).loss.item() * len(ctx)
    return total / len(val)


def fit(train):
    import torch
    from chronos import BaseChronosPipeline
    torch.manual_seed(C.SEED)
    rng = np.random.RandomState(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = BaseChronosPipeline.from_pretrained(MODEL_ID, device_map=device, torch_dtype=torch.float32)
    model = pipe.model

    r = train["ret_index"].values.astype("float32")
    k = len(val_split(train)[0])                          # first 90% = fit days, last 10% = validation days
    tr, val = _examples(r, range(MIN_CONTEXT, k)), _examples(r, range(k, len(r)))

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    best, best_step, waited = _val_loss(model, val, torch, device), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    log = [{"step": 0, "val_loss": best}]
    for step in range(1, MAX_STEPS + 1):
        model.train()
        idx = rng.choice(len(tr), size=BATCH, replace=False)
        ctx, tgt = _batch([tr[i] for i in idx], torch, device)
        loss = model(context=ctx, target=tgt).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0:
            v = _val_loss(model, val, torch, device)
            log.append({"step": step, "val_loss": v, "train_loss": loss.item()})
            if v < best:
                best, best_step, waited = v, step, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                waited += 1
                if waited >= PATIENCE:
                    break

    model.load_state_dict(best_state)
    model.to(DEVICE).eval()
    pipe.ft_info = {"device": device, "fit_examples": len(tr), "val_examples": len(val), "best_step": best_step,
                    "steps_run": step, "val_loss_as_is": log[0]["val_loss"], "val_loss_best": best,
                    "log": pd.DataFrame(log)}
    if device == "cuda":
        torch.cuda.empty_cache()
    return pipe


if __name__ == "__main__":
    walk_forward(fit, predict, NAME, df=load_master())
