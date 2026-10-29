"""Read every outputs/preds_<model>.csv + timing_<model>.json -> results tables laid out like Doc tab 4.
Writes: results_table.csv, table_A_by_fold.csv, table_B_no_big_days.csv, calm_vs_stressed.csv"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from finx_common import score

ORDER = [("naive", "Naive (return = 0)", "Baseline"), ("ma5", "Moving average (5-day)", "Baseline"),
         ("arima", "ARIMA", "Baseline"), ("sarima", "SARIMA", "Traditional ML"),
         ("xgboost", "XGBoost", "Traditional ML"), ("dnn", "DNN", "Deep learning"),
         ("lstm", "LSTM", "Deep learning"), ("chronos_asis", "Chronos-Bolt (as-is)", "LLM / foundation"),
         ("chronos_ft", "Chronos-Bolt (fine-tuned)", "LLM / foundation")]
INFO = {k: (lab, tier) for k, lab, tier in ORDER}


def load(out):
    M = {}
    for k, _, _ in ORDER:
        f = Path(out) / f"preds_{k}.csv"
        if f.exists():
            M[k] = (pd.read_csv(f, index_col=0, parse_dates=True), json.loads((Path(out) / f"timing_{k}.json").read_text()))
    return M


def metrics(P, nP):
    nr = float(np.sqrt((nP["actual"] ** 2).mean()))        # naive predicts 0
    s = score(P["actual"], P["pred"], nr)
    return {"RMSE (%)": s["RMSE"] * 100, "MAE (%)": s["MAE"] * 100, "Direction (%)": s["Direction %"],
            "Skill vs naive (%)": s["Skill vs naive"] * 100}


def main(out=None):
    out = Path(out or C.OUT_DIR)
    M = load(out)
    if "naive" not in M:
        raise SystemExit("naive predictions missing: run src/smoke_test.py first")
    naive = M["naive"][0]
    ns = {k: len(v[0].dropna(subset=["pred"])) for k, v in M.items()}
    if len(set(ns.values())) > 1:
        print("WARNING: models scored on different numbers of days:", ns)
    main_rows, A, B, R = [], [], [], []
    for k, (P, T) in M.items():
        lab, tier = INFO[k]
        P = P.dropna(subset=["pred"]); nP = naive.loc[P.index]
        m = metrics(P, nP)
        if k == "naive":
            m["Direction (%)"] = float("nan")
        main_rows.append({"Model": lab, "Tier": tier, **m,
                          "Train time per fold (s)": np.mean(list(T["train_s"].values())),
                          "Inference per day (ms)": np.mean(list(T["latency_ms_per_day"].values()))})
        a = {"Model": lab}
        for fold, g in P.groupby("fold"):
            mm = metrics(g, nP.loc[g.index]); a[f"RMSE {fold}"] = mm["RMSE (%)"]; a[f"Direction {fold}"] = mm["Direction (%)"]
        if k == "naive":
            a = {x: (np.nan if x.startswith("Direction") else y) for x, y in a.items()}
        A.append(a)
        keep = ~P.index.isin(P["actual"].abs().nlargest(5).index)
        B.append({"Model": lab, "RMSE all days": m["RMSE (%)"], "RMSE without 5 biggest days": metrics(P[keep], nP[keep])["RMSE (%)"]})
        r = {"Model": lab}
        for reg, msk in [("calm", P["VIX"] < C.VIX_STRESS), ("stressed", P["VIX"] >= C.VIX_STRESS)]:
            mm = metrics(P[msk.values], nP[msk.values]) if msk.sum() else {"RMSE (%)": np.nan, "Direction (%)": np.nan}
            r[f"RMSE {reg}"] = mm["RMSE (%)"]; r[f"Direction {reg}"] = mm["Direction (%)"]
        if k == "naive":
            r = {x: (np.nan if x.startswith("Direction") else y) for x, y in r.items()}
        R.append(r)
    up = 100 * (naive["actual"][naive["actual"] != 0] > 0).mean()
    main_rows.append({"Model": "Always up (direction benchmark)", "Tier": "Benchmark", "Direction (%)": up})
    Bd = pd.DataFrame(B)
    Bd["Rank all days"] = Bd["RMSE all days"].rank(); Bd["Rank without 5 biggest days"] = Bd["RMSE without 5 biggest days"].rank()
    Bd["Rank change"] = (Bd["Rank all days"] - Bd["Rank without 5 biggest days"]).astype(int)
    tabs = {"results_table.csv": pd.DataFrame(main_rows), "table_A_by_fold.csv": pd.DataFrame(A),
            "table_B_no_big_days.csv": Bd.drop(columns=["Rank all days", "Rank without 5 biggest days"]),
            "calm_vs_stressed.csv": pd.DataFrame(R)}
    hw = next(iter(M.values()))[1].get("hardware", {})
    for n, t in tabs.items():
        t.round(4).to_csv(out / n, index=False)
    (out / "hardware.txt").write_text(f"Timings measured on: {hw}\n")
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print(tabs["results_table.csv"].round(3).to_string(index=False)); print("Hardware:", hw)
    return tabs


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
