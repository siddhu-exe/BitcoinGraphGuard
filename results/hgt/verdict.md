## 13. Final Verdict (measured)

### 1. Does attention close the aggregate gap on 35-49?
**HGT PR-AUC = 0.4861** (ROC-AUC 0.8937, F1 0.3106) against
XGBoost 0.8013, GraphSAGE baseline 0.6216, GraphSAGE O4 0.6001 and
RGCN 0.4682. Beats RGCN: True. Beats GraphSAGE baseline: False. Beats
GraphSAGE O4: False. Remaining gap to XGBoost: +0.3152. On the stationary
early window 35-42 HGT reaches PR-AUC 0.6921 (XGBoost 0.9215).

### 2. Does attention close the 43-49 gap?
**Verdict on 43-49: attention does not close the gap.** HGT collapses to 0.0386 on 43-49, inside the identical ~0.05 band as XGBoost (0.0427), GraphSAGE (0.0505) and RGCN (0.0550). Replacing mean aggregation with learned multi-head attention changes nothing in the drift regime.

### 3. Did HGT learn to down-weight high-degree reused addresses (the Phase 4/6 mechanism)?
No AddrAddr edges were captured in the true-positive sample, so the mechanistic attention test is inconclusive this run.

### 4. What this means for Phase 8 (MLOps)
HGT does not beat the simpler GNN baselines and does not touch the drift window. Freeze **XGBoost Optimized** as the serving model.

XGBoost's aggregate PR-AUC (0.8013) is 0.18-0.33 above every GNN, and no architecture tested - RGCN,
GraphSAGE with cross-step edges and lag features, or HGT with learned attention - has produced signal
in 43-49. The evidence therefore no longer points to an aggregation-weighting defect as the dominant
cause of the late-window collapse: it points to an **out-of-distribution regime shift** (new entities
and patterns genuinely absent from 1-34; adversarial validation AUC 1.0000, Phase 2b). No frozen
architecture - attention or otherwise - repairs that; only label refresh, continuous retraining, or
explicit drift detection can, which is exactly the Phase 8 mandate.
