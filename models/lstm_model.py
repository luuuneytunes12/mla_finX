"""LSTM regressor using the shared walk-forward harness."""
import os
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "src")]
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", "42")
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

import config as C
from finx_common import load_master, make_window_3d, walk_forward

NAME = "lstm"
HIDDEN = 32
DROPOUT = 0.2
LR = 1e-3
BATCH = 32
MAX_EPOCHS = 200
PATIENCE = 10

HISTORY = {}


def set_seed(seed=C.SEED):
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(5, 58)),
        tf.keras.layers.LSTM(HIDDEN),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss="mse")
    return model


def fit(train):
    set_seed()
    X, y = make_window_3d(train, n=5)
    k = int(len(X) * (1 - C.VAL_FRACTION))
    Xtr, Xval = X[:k], X[k:]
    ytr, yval = y.iloc[:k], y.iloc[k:]

    x_scaler = StandardScaler()
    Xtr = x_scaler.fit_transform(Xtr.reshape(-1, Xtr.shape[-1])).reshape(Xtr.shape)
    Xval = x_scaler.transform(Xval.reshape(-1, Xval.shape[-1])).reshape(Xval.shape)

    y_scaler = StandardScaler()
    ytr_scaled = y_scaler.fit_transform(ytr.to_numpy().reshape(-1, 1)).ravel()
    yval_scaled = y_scaler.transform(yval.to_numpy().reshape(-1, 1)).ravel()

    tf.keras.backend.clear_session()
    model = build()
    stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
    )
    history = model.fit(
        Xtr,
        ytr_scaled,
        validation_data=(Xval, yval_scaled),
        epochs=MAX_EPOCHS,
        batch_size=BATCH,
        callbacks=[stop],
        shuffle=False,
        verbose=0,
    )
    HISTORY[str(train.index[-1].date())] = pd.DataFrame(history.history)
    return model, x_scaler, y_scaler


def predict(model, full, test_index):
    trained_model, x_scaler, y_scaler = model
    X, y = make_window_3d(full, n=5)
    X = X[y.index.get_indexer(test_index)]
    X = x_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    values = trained_model.predict(X, verbose=0).reshape(-1, 1)
    values = y_scaler.inverse_transform(values).ravel()
    return pd.Series(values, index=test_index)


def save_learning_curves(out=C.OUT_DIR / "lstm"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(HISTORY), figsize=(4.2 * len(HISTORY), 3.4), sharey=True)
    for ax, (end, history), fold in zip(np.atleast_1d(axes), HISTORY.items(), C.FOLDS):
        history.to_csv(out / f"history_{fold[0]}.csv", index_label="epoch")
        ax.plot(history.index + 1, history["loss"], label="train", color="#0f2d6b")
        ax.plot(history.index + 1, history["val_loss"], label="validation", color="#d1495b")
        best = int(history["val_loss"].idxmin()) + 1
        ax.axvline(best, ls="--", color="grey", lw=0.8)
        ax.set_title(f"Test {fold[0]} (best epoch {best})", loc="left", fontsize=10)
        ax.set_xlabel("epoch")
    np.atleast_1d(axes)[0].set_ylabel("MSE (scaled y)")
    np.atleast_1d(axes)[0].legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out / "fig_lstm_learning_curves.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    df = load_master()
    P, timing = walk_forward(fit, predict, NAME, df=df)
    save_learning_curves()
    from finx_common import score
    naive_rmse = float(np.sqrt((P["actual"] ** 2).mean()))
    s = score(P["actual"], P["pred"], naive_rmse)
    print(f"[lstm] pooled: RMSE {s['RMSE'] * 100:.4f}%  MAE {s['MAE'] * 100:.4f}%  "
          f"Direction {s['Direction %']:.2f}%  Skill vs naive {s['Skill vs naive'] * 100:.2f}%")