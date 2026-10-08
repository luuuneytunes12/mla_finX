"""XGBoost regressor (worked example of the model interface)."""
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "src")]
import numpy as np, pandas as pd
import xgboost as xgb
import config as C
from finx_common import make_window, walk_forward, load_master

NAME = "xgboost"


def fit(train):
    X, y = make_window(train)
    k = int(len(X) * (1 - C.VAL_FRACTION))          # last 10% = validation slice
    m = xgb.XGBRegressor(max_depth=3, learning_rate=0.03, n_estimators=500, subsample=0.8,
                         colsample_bytree=0.3, random_state=C.SEED, early_stopping_rounds=30,
                         n_jobs=2)
    m.fit(X.iloc[:k], y.iloc[:k], eval_set=[(X.iloc[k:], y.iloc[k:])], verbose=False)
    return m


def predict(model, full, test_index):
    X, _ = make_window(full)
    return pd.Series(model.predict(X.loc[test_index]), index=test_index)


def save_importance_chart(model, out=C.OUT_DIR / "figures"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    names = make_window(load_master())[0].columns
    imp = pd.Series(model.feature_importances_, index=names).nlargest(15)[::-1]
    ax = imp.plot.barh(figsize=(7, 5), color="#0f2d6b")
    ax.set_title("XGBoost: 15 most-used inputs (last fold)", loc="left", fontweight="bold")
    plt.tight_layout(); plt.savefig(out / "xgboost_feature_importance.png", dpi=200); plt.close()


if __name__ == "__main__":
    walk_forward(fit, predict, NAME, df=load_master())
    # chart from a model trained on the last fold's training data
    from finx_common import fold_split, FOLDS
    save_importance_chart(fit(fold_split(load_master(), FOLDS[-1])[0]))
