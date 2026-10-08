"""Ablation: do VIX and the technical indicators add value?
Same data (master_v1.csv), same folds, same 5-day window, same XGBoost settings.
ONLY the input columns change. Averaged over 3 seeds because XGBoost uses random column sampling.

    python src/ablation.py
Writes outputs/ablation_table.csv"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
import xgboost as xgb
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from fincare_common import load_master, make_window, fold_split, score, feature_cols


def column_sets(df):
    base = ["ret_index"] + [c for c in feature_cols(df) if c.endswith("_ret")]   # index + 50 stock returns
    return {
        "A. Returns only (index + 50 stocks)": base,
        "B. A + VIX": base + C.GROUP_VIX,
        "C. A + technical indicators": base + C.GROUP_TECH,
        "D. A + VIX + technical indicators": base + C.GROUP_VIX + C.GROUP_TECH,
        "E. Full master (D + index volume/range)": base + C.GROUP_VIX + C.GROUP_TECH + C.GROUP_IDXVOL,
    }


def run_one(df, cols, seed):
    preds = []
    for fold in C.FOLDS:
        train, test = fold_split(df, fold)
        X, y = make_window(df, cols=cols)
        Xtr, ytr = X.loc[X.index.isin(train.index)], y.loc[X.index.isin(train.index)]
        k = int(len(Xtr) * (1 - C.VAL_FRACTION))
        m = xgb.XGBRegressor(max_depth=3, learning_rate=0.03, n_estimators=500, subsample=0.8,
                             colsample_bytree=0.3, random_state=seed, early_stopping_rounds=30, n_jobs=2)
        m.fit(Xtr.iloc[:k], ytr.iloc[:k], eval_set=[(Xtr.iloc[k:], ytr.iloc[k:])], verbose=False)
        preds.append(pd.DataFrame({"actual": test[C.TARGET], "pred": m.predict(X.loc[test.index])}, index=test.index))
    return pd.concat(preds)


def main(master=None, out=None):
    df = load_master(master)
    out = Path(out or C.OUT_DIR); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, cols in column_sets(df).items():
        res = []
        for seed in C.ABLATION_SEEDS:
            P = run_one(df, cols, seed)
            nr = float(np.sqrt((P["actual"] ** 2).mean()))
            res.append(score(P["actual"], P["pred"], nr))
        r = pd.DataFrame(res)
        rows.append({"Input set": name, "Inputs per day": len(cols),
                     "RMSE (%)": r["RMSE"].mean() * 100, "RMSE spread across seeds (%)": (r["RMSE"].max() - r["RMSE"].min()) * 100,
                     "Direction (%)": r["Direction %"].mean(), "Skill vs naive (%)": r["Skill vs naive"].mean() * 100})
    t = pd.DataFrame(rows)
    base = t["RMSE (%)"].iloc[0]
    t["RMSE change vs A"] = t["RMSE (%)"] - base
    t["Helps?"] = np.where(t["RMSE change vs A"] < -t["RMSE spread across seeds (%)"].clip(lower=0.002), "yes",
                  np.where(t["RMSE change vs A"] > t["RMSE spread across seeds (%)"].clip(lower=0.002), "no (worse)", "no clear gain"))
    t.loc[0, "Helps?"] = "baseline"
    t.round(4).to_csv(out / "ablation_table.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print(t.round(3).to_string(index=False))
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--master"); ap.add_argument("--out")
    a = ap.parse_args(); main(a.master, a.out)
