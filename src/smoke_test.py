"""Smoke test: run Naive, MA(5), ARIMA through the shared walk-forward code.
Check 1: accuracy matches the midterm (Naive 0.966, MA 1.073, ARIMA 0.964 RMSE; small drift ok).
Check 2: train time and latency print for every model (timer works)."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import baselines as B
from finx_common import load_master, walk_forward
import make_results

# midterm numbers: RMSE %, MAE %, Direction %
MIDTERM = {"Naive (return = 0)": (0.966, 0.659, None), "Moving average (5-day)": (1.073, 0.729, 50.1),
           "ARIMA": (0.964, 0.655, 54.3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=None, help="default: data/gold/master.csv")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = load_master(a.master)
    out = a.out or C.OUT_DIR
    for name, fit, pred in [("naive", B.naive_fit, B.naive_predict), ("ma5", B.ma_fit, B.ma_predict),
                            ("arima", B.arima_fit, B.arima_predict)]:
        walk_forward(fit, pred, name, df=df, out_dir=out)
    tabs = make_results.main(out)
    t = tabs["results_table.csv"].set_index("Model")
    bad = []
    print("\nMidterm check (RMSE/MAE tolerance 0.01, Direction tolerance 1.5 points):")
    for m, (r, mae, d) in MIDTERM.items():
        ok = abs(t.loc[m, "RMSE (%)"] - r) <= 0.01 and abs(t.loc[m, "MAE (%)"] - mae) <= 0.01 and \
             (d is None or abs(t.loc[m, "Direction (%)"] - d) <= 1.5)
        print(f"  {m:24s} RMSE {t.loc[m,'RMSE (%)']:.3f} (midterm {r})  MAE {t.loc[m,'MAE (%)']:.3f} ({mae})  "
              f"Dir {t.loc[m,'Direction (%)']:.1f} ({d})  {'OK' if ok else 'DIFFERENT'}")
        if not ok: bad.append(m)
    assert t.loc["ARIMA", "Train time per fold (s)"] > 0 and (t["Inference per day (ms)"].dropna() > 0).all(), "timer broken"
    if bad:
        raise SystemExit(f"SMOKE TEST FAILED for {bad}: fix the shared code before anyone trains a model")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
