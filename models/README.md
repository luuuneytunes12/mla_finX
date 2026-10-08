# models/  — one file per model

Each model file has only 2 functions. The shared code does everything else
(folds, timing, saving, scoring).

    def fit(train):                      # train = master rows before the test year
        ...                              # return anything (your trained model)

    def predict(model, full, test_index):   # full = whole master table (data/gold/master.csv)
        ...                                  # return one number per test day

Run it:

    from finx_common import walk_forward
    walk_forward(fit, predict, name="xgboost")     # saves outputs/preds_xgboost.csv + timing_xgboost.json

Allowed names (so the results table recognises them):
`xgboost`, `dnn`, `lstm`, `sarima`, `chronos_asis`, `chronos_ft`.

Rules (from docs/1. Overview.pdf):
1. Use `make_window(df)` for the 5-day input (290 flat features) or `make_window_3d(df)` for LSTM (5 x 58).
2. Scale inside `fit`, on training rows only. Never on the full table.
3. Early stopping: `val_split(train)` gives the last 10% of training days as the validation slice. Train once. No retrain.
4. Seed 42. One day at a time in `predict` (no peeking at test targets).
5. SARIMA and Chronos-Bolt read `ret_index` only.

Files here: `_template.py` (copy this), `xgboost_model.py` (worked example),
`sarima_model.py` (reference for Hong Rong).
