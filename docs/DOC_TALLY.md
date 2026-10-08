# Where each Claude Doc rule lives in code

| Claude Doc rule | Code |
|---|---|
| 58 inputs, target = next-day return | `src/build_master.py::build`, `config.py::TARGET` |
| 5 data checks | `src/build_master.py::run_checks` |
| Top 50 by Dec-2024 weight, Alphabet once | `config.py::TICKER_WEIGHTS`, `pick_top50` |
| 3 expanding folds, pooled scoring | `config.py::FOLDS`, `fincare_common.walk_forward`, `make_results` |
| Window = 5 days (290 flat / 5×58) | `make_window`, `make_window_3d` |
| 10% validation slice | `val_split`, `config.py::VAL_FRACTION` |
| Naive / MA(5) / ARIMA AIC search | `src/baselines.py` |
| XGBoost settings | `models/xgboost_model.py` |
| SARIMA settings | `models/sarima_model.py` |
| Train time + latency for every model | `walk_forward` timing json |
| Calm vs stressed (VIX 20), Table A, Table B | `src/make_results.py` |
