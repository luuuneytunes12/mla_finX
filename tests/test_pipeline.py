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
    assert t.loc["Naive", "Skill vs naive"] == 0 and (tmp_path / "table_A_by_fold.csv").exists()
