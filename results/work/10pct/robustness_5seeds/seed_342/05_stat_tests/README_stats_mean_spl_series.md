Statistical test workflow summary

Requested metric: wspl
Metric used: mean_spl
Analysis level: series
Block columns: ['task', 'agg_id']
Number of complete blocks: 121
Number of compared models: 11
Friedman chi-square: 744.649136
Friedman p-value: 1.6205e-153
Nemenyi critical difference (alpha=0.10): 1.269725
Ablation models included as benchmark participants: M0_AggHistOnly, M1_AggHistFutureSummary

Series-level protocol:
Each task×series block is formed by averaging the per-origin series metric across rolling folds/origins for each model, then ranking models within that block.
