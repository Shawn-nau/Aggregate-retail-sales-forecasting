# Experiment time origins

Verified against the cached samples in `cache/100pct` and `cache/100pct_rolling`
(cross-checked with `scripts/prepare_m5_experiments.py` split functions).

## Assumptions

- Data: `sales_train_evaluation.csv` — 1941 days, d_1 … d_1941 (d_1 = 2011-01-29)
- Config: `t_hist=56`, `horizon=28`, `valid_size=4`, `test_size=4`,
  `internal_valid_size=3`, `gap=0`, `min_train_origins=20`
- Rolling: `step_size=2`, `max_folds=5`

## Origin semantics

An origin `d_X` means:

- History window: `d_{X-55} … d_X` (56 days)
- Forecast window: `d_{X+1} … d_{X+28}` (28 days)

## Holdout (fold 0)

| Split | Origins | Forecast window |
|---|---|---|
| train (for valid) | d_56 … d_1624 (57) | — |
| valid | d_1680, d_1708, d_1736, d_1764 | 2015-09-05 … 2015-12-25 |
| trainval (for test) | d_56 … d_1764 (62) | — |
| test | d_1820, d_1848, d_1876, d_1904 | 2016-01-23 … 2016-05-13 |

Strict internal validation (used for tuning):

| Split | Origins | Forecast window |
|---|---|---|
| final train | d_56 … d_1652 (58) | — |
| internal valid | d_1708, d_1736, d_1764 | 2015-10-03 … 2015-12-25 |

Internal valid origins are a subset of the holdout valid origins.

## Rolling (5 expanding-window folds)

| Fold | Train origins | Valid origins | Forecast window |
|---|---|---|---|
| 1 | d_56 … d_1344 (47) | d_1400, d_1428, d_1456, d_1484 | 2014-11-29 … 2015-03-20 |
| 2 | d_56 … d_1400 (49) | d_1456, d_1484, d_1512, d_1540 | 2015-01-24 … 2015-05-15 |
| 3 | d_56 … d_1456 (51) | d_1512, d_1540, d_1568, d_1596 | 2015-03-21 … 2015-07-10 |
| 4 | d_56 … d_1512 (53) | d_1568, d_1596, d_1624, d_1652 | 2015-05-16 … 2015-09-04 |
| 5 | d_56 … d_1568 (55) | d_1624, d_1652, d_1680, d_1708 | 2015-07-11 … 2015-10-30 |

Rolling stops at fold 5 (`max_folds=5`, `step_size=2`), so rolling evaluation
never reaches the holdout test window — holdout remains a clean
out-of-sample test.

## Date reference

| d_N | Date | d_N | Date |
|---|---|---|---|
| d_1 | 2011-01-29 | d_1652 | 2015-08-07 |
| d_56 | 2011-03-25 | d_1680 | 2015-09-04 |
| d_1344 | 2014-10-03 | d_1708 | 2015-10-02 |
| d_1400 | 2014-11-28 | d_1764 | 2015-11-27 |
| d_1568 | 2015-05-15 | d_1820 | 2016-01-22 |
| d_1624 | 2015-07-10 | d_1904 | 2016-04-15 |
| | | d_1932 | 2016-05-13 |
| | | d_1941 | 2016-05-22 |
