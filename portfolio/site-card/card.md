# BitcoinGraphGuard: project card (ready to paste)

**Title:** BitcoinGraphGuard

**One-liner:** A Bitcoin fraud detector that knows when to stop trusting itself, built on real
Elliptic++ data (203,769 transactions, 822,942 wallets, about 4.4M edges, 49 time steps).

**Three findings**

1. **XGBoost beat every graph model.** Test PR-AUC 0.80 against 0.62 for the best graph neural
   network (steps 35-49, pooled).
2. **All four models collapsed after step 43.** XGBoost went from 0.92 PR-AUC on steps 35-42 to 0.04
   on steps 43-49; GraphSAGE, HeteroRGCN and HGT ended between 0.04 and 0.06.
3. **Retraining recovers only if labels arrive fast.** PR-AUC on steps 43-49 with expanding
   retraining is 0.69 at a one-step label delay, 0.39 at two, 0.14 at three and 0.11 at five (frozen
   model: 0.04).

**What it includes:** a leakage-free temporal benchmark of four models, a lag-safe drift monitor
(score shift fires at step 43, performance at step 44, no critical alarm before the shift),
walk-forward retraining, a FastAPI service in Docker with GitHub Actions CI, and a live React
dashboard.

**Limitations:** one drift event, real label delay unknown, XGBoost is the deployed model.

**Tech tags:** Python · XGBoost · PyTorch Geometric · FastAPI · React + TypeScript

**Links**

- Live demo: https://bitcoin-graph-delta.vercel.app
- GitHub: https://github.com/siddhu-exe/BitcoinGraphGuard
- README: https://github.com/siddhu-exe/BitcoinGraphGuard#readme

**Images** (all in this folder): `thumbnail.png` (1200x630, card and link preview),
`01_label_delay_recovery.png`, `02_before_after_step43.png`, `03_system_architecture.png` (1600x900).

Numbers: `results/hgt/temporal_metrics.csv`, `results/walkforward/lag_curve.csv`,
`reports/monitoring_backtest_report.json`. The step-43 drop is documented in Weber et al. (2019);
a time step is about two weeks (same paper).
