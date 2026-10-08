"""Tests on the REAL committed data (data/raw). Offline; nothing is written into data/ or outputs/."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np, pandas as pd, pytest
import config as C
import build_master as BM
import finx_common as FC


@pytest.fixture(scope="module")
def raw():
    return BM.load_raw()


@pytest.fixture(scope="module")
def built(raw):
    silver, top = BM.build_silver(raw)
    gold = BM.build_gold(silver)
    BM.run_checks(gold, top, real=True)
    return silver, gold, top


@pytest.fixture(scope="module")
def master(built):
    return built[1]


def test_shape(master):
    assert master.shape[1] == 59 and not master.isna().any().any()
    assert len(FC.feature_cols(master)) == 58
    assert master.index[0] == pd.Timestamp(C.START_DATE)     # warm-up rows are trimmed
    assert master.index[-1] == pd.Timestamp(C.LAST_ROW_DATE)


def test_committed_files_equal_rebuild(built):
    silver, gold, _ = built
    pd.testing.assert_frame_equal(pd.read_csv(C.MASTER_PATH, index_col="Date", parse_dates=True), gold,
                                  check_exact=False, rtol=1e-12, atol=1e-15, check_freq=False)
    pd.testing.assert_frame_equal(pd.read_csv(C.SILVER_PATH, index_col="Date", parse_dates=True), silver,
                                  check_exact=False, rtol=1e-12, atol=1e-15, check_freq=False)


def test_sha256_matches():
    import hashlib
    want = C.MASTER_PATH.with_suffix(".sha256").read_text().split()[0]
    assert hashlib.sha256(C.MASTER_PATH.read_bytes()).hexdigest() == want
    assert len(FC.load_master()) == len(pd.read_csv(C.MASTER_PATH))


def test_period_ends(built):
    silver, gold, _ = built
    assert silver.index[-1] == pd.Timestamp("2026-09-22")
    assert gold.index[-1] == pd.Timestamp("2026-09-21")


def test_top50_list(built):
    top = built[2]
    assert len(top) == 50
    assert top == [t for t in list(C.TICKER_WEIGHTS)[:51] if t != "GOOG"]
    assert pd.read_csv(C.TOP50_PATH)["Symbol"].tolist() == top


def test_no_holidays_in_index(master):
    for d in ["2026-05-25", "2026-09-07"]:
        assert pd.Timestamp(d) not in master.index


def test_fold_sizes(master):
    sizes = [len(FC.fold_split(master, f)[1]) for f in FC.FOLDS]
    assert sizes == [252, 250, 180] and sum(sizes) == 682


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
        assert tr.index.max() < te.index.min() and len(te) >= 180
    tr, va = FC.val_split(master.iloc[:1000])
    assert len(tr) == 900 and len(va) == 100 and tr.index.max() < va.index.min()


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
