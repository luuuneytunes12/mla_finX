"""DNN (Adorcia) 
Input  : make_window(df) -> last 5 trading days x 58 features = 290 flat columns.
"""
import os, sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "src")]
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")          # hide TensorFlow info logs
os.environ.setdefault("PYTHONHASHSEED", "42")
import numpy as np, pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import config as C
from finx_common import make_window, walk_forward, load_master, fold_split, FOLDS

NAME = "dnn"

# ---- Hyperparameters (fixed up front; tuned only on validation slices, never on test) ----
HIDDEN = (64, 32)        
DROPOUT = 0.2 
L2 = 1e-4
LR = 1e-3
BATCH = 32
MAX_EPOCHS = 300
PATIENCE = 20

HISTORY = {}


def set_seed(seed=C.SEED):
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build(n_inputs):
    reg = tf.keras.regularizers.l2(L2)
    layers = [tf.keras.Input(shape=(n_inputs,))]
    for units in HIDDEN:
        layers += [tf.keras.layers.Dense(units, activation="relu", kernel_regularizer=reg),
                   tf.keras.layers.Dropout(DROPOUT)]
    layers += [tf.keras.layers.Dense(1)]
    net = tf.keras.Sequential(layers)
    net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss="mse")
    return net


def fit(train):
    set_seed()
    X, y = make_window(train)
    k = int(len(X) * (1 - C.VAL_FRACTION))
    xs = StandardScaler().fit(X.iloc[:k].values)    # scale on training rows only
    y_mu, y_sd = float(y.iloc[:k].mean()), float(y.iloc[:k].std())

    Xtr, Xva = xs.transform(X.iloc[:k].values), xs.transform(X.iloc[k:].values)
    ytr, yva = (y.iloc[:k].values - y_mu) / y_sd, (y.iloc[k:].values - y_mu) / y_sd

    net = build(X.shape[1])
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                            restore_best_weights=True)
    h = net.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=MAX_EPOCHS, batch_size=BATCH,
                shuffle=True, callbacks=[stop], verbose=0)
    HISTORY[str(train.index[-1].date())] = pd.DataFrame(h.history)
    return {"net": net, "xs": xs, "y_mu": y_mu, "y_sd": y_sd, "cols": list(X.columns),
            "best_epoch": int(np.argmin(h.history["val_loss"])) + 1, "epochs_run": len(h.history["loss"])}


def predict(model, full, test_index):
    X, _ = make_window(full)
    Xt = model["xs"].transform(X.loc[test_index, model["cols"]].values)
    z = model["net"](Xt, training=False).numpy().ravel()
    return pd.Series(z * model["y_sd"] + model["y_mu"], index=test_index)


def save_learning_curves(out=C.OUT_DIR / "dnn"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(HISTORY), figsize=(4.2 * len(HISTORY), 3.4), sharey=True)
    for ax, (end, h), fold in zip(np.atleast_1d(axes), HISTORY.items(), FOLDS):
        h.to_csv(out / f"history_{fold[0]}.csv", index_label="epoch")
        ax.plot(h.index + 1, h["loss"], label="train", color="#0f2d6b")
        ax.plot(h.index + 1, h["val_loss"], label="validation", color="#d1495b")
        best = int(h["val_loss"].idxmin()) + 1
        ax.axvline(best, ls="--", color="grey", lw=0.8)
        ax.set_title(f"Test {fold[0]} (best epoch {best})", loc="left", fontsize=10)
        ax.set_xlabel("epoch")
    np.atleast_1d(axes)[0].set_ylabel("MSE (scaled y)")
    np.atleast_1d(axes)[0].legend(frameon=False)
    plt.tight_layout(); plt.savefig(out / "fig_dnn_learning_curves.png", dpi=200); plt.close()


if __name__ == "__main__":
    df = load_master()
    P, timing = walk_forward(fit, predict, NAME, df=df)
    save_learning_curves()
    from finx_common import score
    naive_rmse = float(np.sqrt((P["actual"] ** 2).mean()))
    s = score(P["actual"], P["pred"], naive_rmse)
    print(f"[dnn] pooled: RMSE {s['RMSE']*100:.4f}%  MAE {s['MAE']*100:.4f}%  "
          f"Direction {s['Direction %']:.2f}%  Skill vs naive {s['Skill vs naive']*100:.2f}%")
