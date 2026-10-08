"""Tests on FAKE data. They check the plumbing (no look-ahead, folds, shapes), not real accuracy."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np, pandas as pd, pytest
import config as C
import build_master as BM
import fincare_common as FC


@pytest.fixture(scope="module")
def master():
    df, top = BM.build(BM.synthetic_raw())
    BM.run_checks(df, top)
    return df


def test_shape(master):
    assert master.shape[1] == 59 and not master.isna().any().any()
    assert master.index[0] >= pd.Timestamp("2021-02-05")     # MACD warm-up


def test_target_is_tomorrow(master):
    assert np.allclose(master[C.TARGET].iloc[:-1], master["ret_index"].shift(-1).iloc[:-1])


def test_window_shapes_and_no_leak(master):
    X, y = FC.make_window(master)
    assert X.shape[1] == 290
    A, _ = FC.make_window_3d(master)
    assert A.shape[1:] == (5, 58)
    t = X.index[10]                                   # row t must hold ret_index[t] in lag0
    assert X.loc[t, "ret_index_lag0"] == master.loc[t, "ret_index"]
    assert A[10, -1, 0] == master.iloc[master.index.get_loc(t)]["ret_index"]   # newest day last


def test_folds_do_not_overlap(master):
    for f in FC.FOLDS:
        tr, te = FC.fold_split(master, f)
        assert tr.index.max() < te.index.min() and len(te) > 100 or f[0] == "2026"
    tr, va = FC.val_split(master.iloc[:1000])
    assert len(va) == 100 and tr.index.max() < va.index.min()


def test_walk_forward_end_to_end(master, tmp_path):
    import baselines as B, make_results
    for n, f, p in [("naive", B.naive_fit, B.naive_predict), ("ma5", B.ma_fit, B.ma_predict)]:
        FC.walk_forward(f, p, n, df=master, out_dir=tmp_path, verbose=False)
    t = make_results.main(tmp_path)["results_table.csv"].set_index("Model")
    assert t.loc["Naive (return = 0)", "Skill vs naive (%)"] == 0 and (tmp_path / "table_A_by_fold.csv").exists()


def test_3d_window_matches_shifted_frames(master):
    A, _ = FC.make_window_3d(master)
    X, _ = FC.make_window(master)
    feats = FC.feature_cols(master)
    i = 25
    t = X.index[i]
    pos = master.index.get_loc(t)
    expected = master[feats].iloc[pos - 4: pos + 1].values        # oldest day first, today last
    assert np.allclose(A[i], expected)


def test_arima_row_t_forecasts_next_day():
    import baselines as B
    rng = np.random.default_rng(1)
    r = np.zeros(1500)
    for i in range(1, 1500):
        r[i] = 0.6 * r[i - 1] + rng.normal(0, 0.01)           # strong AR(1): tomorrow depends on today
    idx = pd.bdate_range("2021-01-04", periods=1500)
    df = pd.DataFrame({"ret_index": r, C.TARGET: np.r_[r[1:], np.nan], "VIX": 15.0}, index=idx).iloc[:-1]
    train, test = df.iloc[:1000], df.iloc[1000:1200]
    m = B.arima_fit(train, max_p=1, max_q=0)
    p = B.arima_predict(m, df, test.index)
    assert np.corrcoef(p, test[C.TARGET])[0, 1] > 0.5


def test_ablation_column_sets(master):
    import ablation
    sets = ablation.column_sets(master)
    n = [len(v) for v in sets.values()]
    assert n == [51, 53, 54, 56, 58]
    X, _ = FC.make_window(master, cols=sets["A. Returns only (index + 50 stocks)"])
    assert X.shape[1] == 51 * 5
