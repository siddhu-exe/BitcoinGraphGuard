## 10. Post-run Conclusions

### Required statement on the 43-49 collapse

The 43-49 collapse was **improved**: the best configuration on that
window (`recency_linear`) reaches PR-AUC **0.0483** against the recorded Phase 2 value of
0.0427, a change of **+0.0056**
(improvement threshold set in advance at +0.005). Its cost, if any, is visible in the other two
windows: 35-42 PR-AUC 0.9248 and 35-49 PR-AUC 0.8070, against 0.9228 and
0.8011 for the unweighted reference.

### Covariate or concept drift?

The pre-registered rule resolves to **covariate drift / prior shift (the inputs or the label mix moved)**:
adversarial validation AUC 1.0000 (material at >= 0.80) and median top-15 KS
0.5336 (material at >= 0.30), against an in-window CV PR-AUC of 0.9199 on 43-49 versus 0.0427 for the transferred model.
The in-window figure is a diagnostic that uses the window's own labels; it is not a model result
and is not reported as performance.

### What this means for the next phase

Reweighting a fixed 165-column tabular representation cannot recover a relationship that has
changed. Any remaining headroom in this regime is structural - step-bounded address-graph
features built with the same provenance audit - and that work belongs to Phases 4, 5 and 7.