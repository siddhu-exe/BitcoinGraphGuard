# Diagnostic Verdict: Out-of-Distribution (OOD) Mechanism

## Verdict: (c) NOT CONFIRMED — Mechanical Leak / Single-Feature Concentration

### Key Empirical Findings:
1. **Full Adversarial AUC**: 5-Fold Mean AUC = **1.0000 +/- 0.0000**.
2. **Attribution Concentration**: Top 1 feature = **36.5%**, Top 3 = **95.2%**, Top 10 = **98.4%**.
3. **Leak-Feature Audit**: **0** of the Top 20 features showed mechanical timestamp signatures.
4. **Leave-One-Out Robustness**:
   - Dropping Top 1 feature yields AUC **1.0000**.
   - Dropping Top 3 features yields AUC **1.0000**.
   - Dropping Top 20 features yields AUC **0.9904**.
   - Local Features Only (93 feats) yields AUC **0.9885**.

### Scientific Implication for Phase 8:
Adversarial separability collapses upon removing top features or is dominated by trivial time proxies.
