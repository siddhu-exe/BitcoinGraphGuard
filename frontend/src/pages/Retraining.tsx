import { loadWalkforward } from "../lib/data";
import {
  expandingPooled,
  perStepColumns,
  sustainedRecoveryStep,
  thresholdRuleRecall,
  thresholdSwing,
  topKRecall,
  triageRows,
} from "../lib/derived";
import { NA, num, pct, span, trim } from "../lib/format";
import { useStatic } from "../lib/useStatic";
import PerStepColumns from "../components/PerStepColumns";
import TriageTable from "../components/TriageTable";
import { AsyncState, Card, InfoIcon } from "../components/ui";

export default function Retraining() {
  const wf = useStatic(loadWalkforward);
  if (wf.loading) return <AsyncState loading label="walk-forward results" />;
  if (wf.error || !wf.data) return <AsyncState loading={false} error={wf.error} label="walk-forward results" />;

  const w = wf.data;
  const l1 = expandingPooled(w, 1, "43-49");
  const l3 = expandingPooled(w, 3, "43-49");
  const columns = perStepColumns(w);
  const noisy = columns.filter((c) => c.noisy);
  const recovery = sustainedRecoveryStep(columns);
  const rows = triageRows(w);
  const totalPositives = w.triage.steps
    .filter((r) => r.strategy === "expanding" && r.lag === 1 && r.step >= 43)
    .reduce((s, r) => s + r.positives, 0);
  const swing = thresholdSwing(w);
  const ruleRecall = thresholdRuleRecall(w, 3);
  const topRecall = topKRecall(w, 3, 100);

  const noisyText = noisy.length
    ? `Steps ${noisy.map((c) => c.step).join(" and ")} have ${noisy.map((c) => c.positives).join(" and ")} positives, too few for a stable PR-AUC.`
    : "";

  return (
    <div className="page">
      <section className="head-block">
        <h1 className="h1">Can retraining fix the break?</h1>
        <p className="lede">
          After criminal behaviour changed at step 43, I retrained the model every two weeks on newly confirmed fraud
          cases, using only cases confirmed before each prediction. This page shows how fast it recovered and how many
          fraud cases a review team would catch.
        </p>
      </section>

      <section className="callout" aria-label="Callout">
        <InfoIcon />
        <p style={{ fontSize: 15 }}>
          Retraining works only when labels arrive within about 1–2 steps. The real delay is unknown, so treat{" "}
          {num(l1?.pr_auc ?? null, 3)} as the best case: at 3 steps the same loop reaches {num(l3?.pr_auc ?? null, 3)}.
        </p>
      </section>

      <Card>
        <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", gap: "8px 24px" }}>
          <div className="card-head">
            <h2 className="h2">Per step after the shift</h2>
            <p className="subtle">
              Expanding retrain at L = 1, PR-AUC per step. Bars below show illicit labels per step.
            </p>
          </div>
          <div className="legend">
            <span className="legend__item">
              <span className="legend__swatch legend__swatch--accent" aria-hidden="true" />
              Retrained PR-AUC
            </span>
            <span className="legend__item">
              <span className="legend__swatch legend__swatch--muted" aria-hidden="true" />
              Illicit labels
            </span>
          </div>
        </div>
        <PerStepColumns columns={columns} />
        <p className="footnote">
          Steps 35–42 predate the shift and are not scored here. {noisyText} Sustained recovery (≥ 0.5 for 3 steps) first
          holds at step {recovery ?? NA}.
        </p>
      </Card>

      <section className="split">
        <Card className="card--wide">
          <div className="card-head">
            <h2 className="h2">Alert-budget triage</h2>
            <p className="subtle">Top K flagged transactions per step, pooled over {span(43, 49)}, L = 1</p>
          </div>
          <TriageTable rows={rows} />
          <p className="footnote">
            Max recall is the sum over steps of min(K, positives) divided by {totalPositives} illicit labels.
          </p>
        </Card>

        <Card className="card--narrow card--snug">
          <h2 className="h2">Why top-K, not a cutoff</h2>
          <div className="stat">
            <span className="stat__value">
              {trim(swing.max, 3)} → {trim(swing.min, 3)}
            </span>
            <span className="stat__caption">validation threshold swing across steps {span(43, 49)}</span>
          </div>
          <p style={{ fontSize: 14, color: "var(--muted-strong)" }}>
            At L = 3 the threshold rule recalls {pct(ruleRecall, 0)} of illicit labels. Top-100 recalls{" "}
            {pct(topRecall, 0)} on the same ranking.
          </p>
          <p className="footnote">
            Expanding, recency-weighted and no-triad retrains are statistically tied. Expanding is the default, not a
            measured winner.
          </p>
        </Card>
      </section>
    </div>
  );
}
