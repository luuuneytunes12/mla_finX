"""Read every outputs/preds_<model>.csv + timing_<model>.json -> results tables.
Writes: results_table.csv (main), table_A_by_fold.csv, table_B_no_big_days.csv, calm_vs_stressed.csv"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from fincare_common import score

DISPLAY = {"naive": "Naive", "ma5": "MA(5)", "arima": "ARIMA", "xgboost": "XGBoost",
           "dnn": "DNN", "lstm": "LSTM", "sarima": "SARIMA",
           "chronos_asis": "Chronos-Bolt (as-is)", "chronos_ft": "Chronos-Bolt (fine-tuned)"}


def load(out):
    models = {}
    for f in sorted(Path(out).glob("preds_*.csv")):
        k = f.stem[len("preds_"):]
        models[k] = (pd.read_csv(f, index_col=0, parse_dates=True),
                     json.loads((Path(out) / f"timing_{k}.json").read_text()))
    return models


def rows_for(P, naive_P, timing, label, mask=None):
    P, nP = (P, naive_P) if mask is None else (P[mask], naive_P[mask])
    nr = float(np.sqrt((nP["actual"] ** 2).mean()))   # naive predicts 0
    s = score(P["actual"], P["pred"], nr)
    s["RMSE"], s["MAE"] = s["RMSE"] * 100, s["MAE"] * 100   # show in % points
    if label == "Naive":
        s["Direction %"] = float("nan")   # constant 0 has no direction
    s["Model"] = label
    s["Train time (s) per fold, mean"] = np.mean(list(timing["train_s"].values()))
    s["Latency (ms/day)"] = np.mean(list(timing["latency_ms_per_day"].values()))
    return s


def main(out=None):
    out = Path(out or C.OUT_DIR)
    M = load(out)
    if "naive" not in M:
        raise SystemExit("naive predictions missing: run src/smoke_test.py first")
    naive = M["naive"][0]
    cols = ["Model", "RMSE", "MAE", "Direction %", "Skill vs naive",
            "Train time (s) per fold, mean", "Latency (ms/day)"]
    main_rows, foldA, noBig, regime = [], [], [], []
    always_up = 100 * (naive["actual"][naive["actual"] != 0] > 0).mean()
    for k, (P, T) in M.items():
        lab = DISPLAY.get(k, k)
        P = P.dropna(subset=["pred"])          # MA/ARIMA may have no value on a few days
        nP = naive.loc[P.index]
        main_rows.append(rows_for(P, nP, T, lab))
        for fold, g in P.groupby("fold"):
            r = rows_for(g, nP.loc[g.index], T, lab); r["Fold"] = fold; foldA.append(r)
        big = P["actual"].abs().nlargest(5).index
        keep = ~P.index.isin(big)
        noBig.append(rows_for(P, nP, T, lab, keep))
        for reg, m in [("Calm (VIX<20)", P["VIX"] < C.VIX_STRESS), ("Stressed (VIX>=20)", P["VIX"] >= C.VIX_STRESS)]:
            if m.sum():
                r = rows_for(P, nP, T, lab, m.values); r["Regime"] = reg; r["Days"] = int(m.sum()); regime.append(r)
    f = lambda rows, extra=[]: pd.DataFrame(rows)[extra + cols].round(4)
    tabs = {"results_table.csv": f(main_rows), "table_A_by_fold.csv": f(foldA, ["Fold"]),
            "table_B_no_big_days.csv": f(noBig), "calm_vs_stressed.csv": f(regime, ["Regime", "Days"])}
    for n, t in tabs.items():
        t.to_csv(out / n, index=False)
    pd.set_option("display.width", 200)
    print(tabs["results_table.csv"].to_string(index=False))
    print(f"\nAlways-up direction benchmark: {always_up:.1f}%")
    return tabs


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
