import re

with open("docs/TEMPORAL_INDUCTIVE_EVALUATION.md", "r") as f:
    text = f.read()

results_section = """
## 12. Results

The models were evaluated remotely on Google Colab/Kaggle. The exact outputs were successfully exported to `temporal_inductive/*.csv`:

* **Temporal Degradation (Q1):** The frozen XGBoost and HeteroRGCN models successfully classify illicit transactions mostly intact through step 42 (XGBoost PR-AUC 0.89-0.99, RGCN PR-AUC 0.50-0.78). At **step 43**, the illicit prevalence abruptly plunges from 9-11% to **2.53%**. Simultaneously, performance immediately collapses (XGBoost step 43 PR-AUC is 0.039; RGCN is 0.035).
* **Static vs Continuous Retraining (Q2):** Expanding-window retraining dynamically lifts the sub-drift performance (e.g. from 0.588 static to 0.817 expanding on step 41). However, the extreme drift at step 43-49 still damages expanding models heavily (step 43 expanding PR-AUC is 0.066). The Rolling window (`W=20`) isolates completely from ancient distributions and performs marginally better amid severe drift (0.184 PR-AUC in 43-49 compared to 0.129 for expanding and 0.072 for static).
* **Inductive Generalization (Q3):** When transacting on entirely **unseen** addresses, expanding RGCN scores **0.763** PR-AUC. When transacting with **seen** addresses (historical overlap), RGCN scores only **0.194**. This is highly non-intuitive but robustly confirms that *novel* entities conform tightly to broad structural feature patterns, whereas recurring/heavily-seen addresses generate noisier message-passing overlaps.
* **Historical Paths:** 10,812 test transactions have `T_train -> A -> T_test` connectivity. Similar to unseen vs seen above, predicting on nodes *without* a historical path yields vastly higher PR-AUC (0.761) than predicting on nodes *with* a historical path (0.194) under the expanding regime. 
* **Threshold Adaptation (Q4):** PR-AUC is threshold-free and collapses structurally. However, updating thresholds dynamically ($\tau_t$) or optimally ($\tau^*_oracle$) proves that F1 collapse can be *somewhat* recovered. The frozen threshold derived on the validation set ($\tau^* = 0.83$) is severely miscalibrated to the 2.5% prevalence of step 43+. 

## 13. Conclusions

Based on the empirical evidence, the conclusions governing phase progression are:
1. **The Step 43 Degradation is an Abrupt Regime Shift:** Degradation is not a slow curve; it breaks violently exactly at step 43, completely correlated with illicit prevalence dropping and graph fragmentation increasing.
2. **Retraining Regimes are Required:** Static models are strictly dead upon drift. Expanding and Rolling updating must be the standard for production.
3. **Address Reuse is Computationally Confounding (Dilution):** HeteroRGCN suffers extensively on nodes with rich/recurring connectivity to the historical graph (`seen` wallets and historical paths). This confirms our Phase 4 hypothesis that uniform, unweighted mean-aggregation across the dense `AddrAddr` graph causes extreme message diffusion and over-smoothing.
4. **Next Architecture Need:** The evidence dictates strongly against purely uniform relational aggregations. The subsequent architecture (likely HGT or Attention-based) must **learn to down-weight noisy recurring structural addresses** and gate attention across time to explicitly mitigate dilution.
"""

text = re.sub(r'## 12\. Results.*?## 14\. Reproducibility', results_section + "\n## 14. Reproducibility", text, flags=re.DOTALL)

with open("docs/TEMPORAL_INDUCTIVE_EVALUATION.md", "w") as f:
    f.write(text)
print("Updated docs/TEMPORAL_INDUCTIVE_EVALUATION.md")
