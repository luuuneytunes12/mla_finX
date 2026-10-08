# mla_finX — Predict tomorrow's S&P 500 return (IS460 group project)

**Big picture.** We guess the S&P 500's return for the *next* trading day, using what is known at today's close.
Six people each own one model. Everyone uses the **same data table** and the **same test code**, so the results can be compared fairly.
The full plan is `docs/1. Overview.pdf`. This repo is the code that follows that plan.

## Folder map

| Folder / file | What it is |
|---|---|
| `config.py` | Every setting in one place (dates, 50 tickers, folds, window = 5 days, seed 42) and all file paths |
| `data/raw/` | Yahoo download, saved as-is (`yahoo_ohlcv.csv` + EDA csvs). **Committed on purpose. Everyone reads it, nobody edits it.** |
| `data/silver/` | Cleaned inputs for every trading day (`features.csv`) and the 50 stocks used (`top50_tickers.csv`) |
| `data/gold/` | `master.csv`: the frozen table, 59 columns (58 inputs X + 1 target Y), 2021-01-04 to 2026-09-21, plus `master.sha256`. **Never edit.** Models read only this file |
| `notebooks/` | `01_eda`, `02_smoke_test`, `03_baselines`. Old midterm notebooks are in `archive_midterm/` |
| `docs/` | `1. Overview.pdf` (the plan) and `DOC_TALLY.md` (which plan rule lives in which code) |
| `outputs/` | Predictions, timings, results tables. Notebook figures go to `outputs/eda/` and `outputs/baselines/` |
| `src/build_master.py` | raw -> silver -> gold, runs 7 checks, freezes the master table |
| `src/finx_common.py` | Shared code: windows, folds, validation slice, scoring, timing |
| `src/baselines.py` | Naive, Moving Average (5), ARIMA |
| `src/smoke_test.py` | Runs the 3 baselines through the shared code (matches midterm? timers work?) |
| `src/make_results.py` | Collects every model's predictions into the results tables |
| `models/` | One file per model. Copy `_template.py`. `xgboost_model.py` is a worked example |
| `tests/` | Quick automatic checks on the real data (offline) |

## Quick start (Wei Lun's part, in order)

```bash
pip install -r requirements.txt
python src/build_master.py          # 1. raw -> silver -> gold + 7 checks (offline; add --refresh to re-download from Yahoo)
python -m pytest -q tests           # 2. plumbing tests on the real data (a few seconds)
python src/smoke_test.py            # 3. Naive/MA/ARIMA; RMSE must match midterm (0.966 / 1.073 / 0.964)
python src/make_results.py          # 4. results tables in outputs/
```
`data/gold/master.csv` is already built and committed. Re-running step 1 gives the same file unless you use `--refresh` (Yahoo numbers change, so only do that on purpose).

## For model owners (Owen, Ryan, Hong Rong, ...)

```bash
cp models/_template.py models/lstm_model.py      # fill in fit() and predict(); see models/README.md
python models/lstm_model.py                      # writes outputs/preds_lstm.csv + timing_lstm.json
python src/make_results.py                       # refreshes the results table
```

## The rules in one minute

- **Y** = tomorrow's S&P 500 return. **X** = 58 numbers known at today's close (index return, 50 stock returns, VIX, VIX change, RSI, MACD, Bollinger %B, volume change, high–low range).
- **Folds (walk-forward, expanding):** train 2021-01-04–Dec 2023 → test 2024; train to Dec 2024 → test 2025; train to Dec 2025 → test Jan–Sep 2026. Scores use all test days pooled together (the last training row's label is the first test day's return: standard for next-day walk-forward, 1-day overlap) (not averaged per fold).
- **Early stopping:** last 10% of each training period is the validation slice. Train once. No retraining.
- **Window:** last 5 trading days. XGBoost/DNN get 290 flat columns, LSTM gets 5×58, SARIMA and Chronos-Bolt read `ret_index` only.
- **Scale** only inside `fit`, on training rows. **Seed** 42.
- **Metrics:** RMSE, MAE, Direction %, skill vs naive (1 − RMSE/RMSE_naive), train time, latency (ms/day).

## Push to GitHub

```bash
git remote add origin https://github.com/<your-username>/mla_finX.git
git push -u origin main
```
