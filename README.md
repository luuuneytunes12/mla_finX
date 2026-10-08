# mla_finX — Predict tomorrow's S&P 500 return (IS460 group project)

**Big picture.** We guess the S&P 500's return for the *next* trading day, using what is known at today's close.
Six people each own one model. Everyone uses the **same data table** and the **same test code**, so the results can be compared fairly.
The full plan is in the Claude Doc (tabs 1–6). This repo is the code that follows that plan.

## Folder map

| Folder / file | What it is |
|---|---|
| `config.py` | Every setting in one place (dates, 50 tickers, folds, window = 5 days, seed 42) |
| `data/master_v1.csv` | The frozen table: 59 columns (58 inputs X + 1 target Y). **Never edit.** |
| `data/top50_tickers.csv` | The 50 stocks used |
| `src/build_master.py` | Downloads Yahoo data, builds the table, runs 5 checks, freezes it |
| `src/fincare_common.py` | Shared code: windows, folds, validation slice, scoring, timing |
| `src/baselines.py` | Naive, Moving Average (5), ARIMA |
| `src/smoke_test.py` | Runs the 3 baselines through the shared code (matches midterm? timers work?) |
| `src/make_results.py` | Collects every model's predictions into the results tables |
| `models/` | One file per model. Copy `_template.py`. `xgboost_model.py` is a worked example |
| `notebooks/` | `01_eda.ipynb`, `02_smoke_test.ipynb`. Old midterm notebooks are in `archive_midterm/` |
| `outputs/` | Predictions, timings, results tables, figures |
| `tests/` | Quick automatic checks (run on fake data) |

## Quick start (Wei Lun's part, in order)

```bash
pip install -r requirements.txt
python src/build_master.py          # 1. download + build + 5 checks -> data/master_v1.csv
python -m pytest -q tests           # 2. plumbing tests (fake data, 2 seconds)
python src/smoke_test.py            # 3. Naive/MA/ARIMA; RMSE must match midterm (0.966 / 1.073 / 0.964)
python src/make_results.py          # 4. results tables in outputs/
```
After step 3 passes, commit `data/master_v1.csv`, tell the team, and they start training.

## For model owners (Owen, Ryan, Hong Rong, ...)

```bash
cp models/_template.py models/lstm_model.py      # fill in fit() and predict(); see models/README.md
python models/lstm_model.py                      # writes outputs/preds_lstm.csv + timing_lstm.json
python src/make_results.py                       # refreshes the results table
```

## The rules in one minute

- **Y** = tomorrow's S&P 500 return. **X** = 58 numbers known at today's close (index return, 50 stock returns, VIX, VIX change, RSI, MACD, Bollinger %B, volume change, high–low range).
- **Folds (walk-forward, expanding):** train Feb 2021–Dec 2023 → test 2024; train to Dec 2024 → test 2025; train to Dec 2025 → test Jan–Sep 2026. Scores use all test days pooled together (not averaged per fold).
- **Early stopping:** last 10% of each training period is the validation slice. Train once. No retraining.
- **Window:** last 5 trading days. XGBoost/DNN get 290 flat columns, LSTM gets 5×58, SARIMA and Chronos-Bolt read `ret_index` only.
- **Scale** only inside `fit`, on training rows. **Seed** 42.
- **Metrics:** RMSE, MAE, Direction %, skill vs naive (1 − RMSE/RMSE_naive), train time, latency (ms/day).

## About `outputs/synthetic/` and `data/synthetic/`

Made with `--synthetic`. They are **fake random data** that only prove the code runs. They are git-ignored.
Never put those numbers in the report.

## Push to GitHub

```bash
git remote add origin https://github.com/<your-username>/mla_finX.git
git push -u origin main
```
