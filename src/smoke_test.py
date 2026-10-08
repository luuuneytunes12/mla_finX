"""Smoke test: run Naive, MA(5), ARIMA through the shared walk-forward code.
Check 1: accuracy matches the midterm (Naive 0.966, MA 1.073, ARIMA 0.964 RMSE; small drift ok).
Check 2: train time and latency print for every model (timer works)."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import baselines as B
from fincare_common import load_master, walk_forward
import make_results

MIDTERM = {"Naive": 0.966, "MA(5)": 1.073, "ARIMA": 0.964}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = load_master(a.master)
    out = a.out or C.OUT_DIR
    for name, fit, pred in [("naive", B.naive_fit, B.naive_predict), ("ma5", B.ma_fit, B.ma_predict),
                            ("arima", B.arima_fit, B.arima_predict)]:
        walk_forward(fit, pred, name, df=df, out_dir=out)
    tabs = make_results.main(out)
    t = tabs["results_table.csv"].set_index("Model")
    if "synthetic" in str(a.master):
        print("\nFAKE DATA: midterm comparison skipped (only plumbing is tested).")
        return
    print("\nMidterm check (RMSE in %, tolerance 0.01):")
    for m, v in MIDTERM.items():
        got = t.loc[m, "RMSE"]
        print(f"  {m:8s} midterm {v:.3f}  now {got:.3f}  {'OK' if abs(got - v) <= 0.01 else 'DIFFERENT -> look for a bug'}")
    assert (t["Train time (s) per fold, mean"] >= 0).all() and (t["Latency (ms/day)"] > 0).all(), "timer broken"


if __name__ == "__main__":
    main()
