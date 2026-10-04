import { useMemo, useState } from "react";
import { Badge, Card, CardSkeleton, ErrorState, Legend } from "../components/ui";
import { PrevalenceBars, StepLines } from "../components/charts";
import { loadMonitoring, loadPerformance } from "../lib/data";
import { ACTION_LABEL, DRIFT_START, fmt, pct, signedPct } from "../lib/format";
import { useStatic } from "../lib/useStatic";
import type { PerformanceData, WindowMetric } from "../lib/types";
import { MODELS } from "../lib/models";

type Filter = "all" | "pre" | "post";
const FILTERS: { id: Filter; label: string }[] = [
  { id: "all", label: "All steps (35–49)" },
  { id: "pre", label: "Pre-drift (35–42)" },
  { id: "post", label: "Post-drift (43–49)" },
];
const WINDOW_ORDER: WindowMetric["window"][] = ["35-49", "35-42", "43-49"];
const MODEL_ORDER = ["XGBoost", "GraphSAGE", "HGT", "HeteroRGCN"];

export default function Performance() {
  const perf = useStatic(loadPerformance);
  const mon = useStatic(loadMonitoring);
  const [filter, setFilter] = useState<Filter>("all");

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Model performance</h1>
          <p>
            PR-AUC on the temporal test window (steps 35–49). Every model was trained on steps 1–34 and frozen; the regime shifts at step{" "}
            {DRIFT_START}.
          </p>
        </div>
        <div className="seg" role="group" aria-label="Step range">
          {FILTERS.map((f) => (
            <button key={f.id} type="button" aria-pressed={filter === f.id} onClick={() => setFilter(f.id)}>
              {f.label}
            </button>
          ))}
        </div>
      </div>

      {perf.error ? (
        <Card>
          <ErrorState title="Performance data failed to load" error={perf.error} onRetry={perf.retry} />
        </Card>
      ) : !perf.data ? (
        <>
          <div className="grid grid--9-3">
            <CardSkeleton h={300} title="chart" />
            <CardSkeleton h={300} title="diagnosis" />
          </div>
          <CardSkeleton h={320} title="table" />
        </>
      ) : (
        <Loaded perf={perf.data} filter={filter} decision={mon.data?.latest_decision} />
      )}
    </>
  );
}

function Loaded({
  perf,
  filter,
  decision,
}: {
  perf: PerformanceData;
  filter: Filter;
  decision?: import("../lib/types").CachedDecision;
}) {
  const steps = useMemo(
    () =>
      perf.steps.filter((s) => (filter === "all" ? true : filter === "pre" ? s.step < DRIFT_START : s.step >= DRIFT_START)),
    [perf.steps, filter],
  );
  const w = (m: string, win: WindowMetric["window"]) => perf.windows.find((x) => x.model === m && x.window === win);
  const d = perf.drift_diagnosis;
  const noise = useMemo(
    () => Object.fromEntries(steps.filter((s) => s.low_positives).map((s) => [s.step, s.illicit])) as Record<number, number>,
    [steps],
  );
  const noisy = Object.entries(noise);
  const early = w("XGBoost", "35-42");
  const late = w("XGBoost", "43-49");

  const rows = perf.windows
    .filter((r) => (filter === "all" ? true : r.window === (filter === "pre" ? "35-42" : "43-49")))
    .sort(
      (a, b) =>
        WINDOW_ORDER.indexOf(a.window) - WINDOW_ORDER.indexOf(b.window) ||
        MODEL_ORDER.indexOf(a.model) - MODEL_ORDER.indexOf(b.model),
    );

  return (
    <>
      <div className="grid grid--9-3">
        <div className="stack">
          <Card
            title={`PR-AUC by time step (${steps[0].step}–${steps[steps.length - 1].step})`}
            sub="One step ≈ two weeks"
            aside={<Legend label="Series" items={MODELS.map((m) => ({ name: m.name, color: m.hex, dash: m.dash }))} />}
          >
            <StepLines
              data={steps}
              lines={MODELS.map((m) => ({ key: m.key, name: m.name, color: m.hex, dash: m.dash }))}
              noise={noise}
              summary="Line chart of PR-AUC per time step for XGBoost, GraphSAGE, HGT and HeteroRGCN. A data table follows the chart."
            />
            <div className="chart-head chart-head--sub row-between">
              <h3>Illicit labels per step</h3>
              <span className="hint">The positives each step's PR-AUC is computed from</span>
            </div>
            <StepLines
              data={steps.map((s) => ({ step: s.step, illicit: s.illicit }))}
              lines={[{ key: "illicit", name: "Illicit labels", color: "#475569" }]}
              noise={noise}
              height="chart--sm"
              domain={[0, "auto"]}
              bandLabel={false}
              yFormat={(v) => String(Math.round(v))}
              formatter={(v) => String(Math.round(v))}
              summary="Line chart of the number of illicit labels per time step. Steps with fewer than 10 are marked as too noisy for a per-step PR-AUC."
            />
            {noisy.length > 0 && (
              <p className="hint chart-note">
                <Badge tone="warn">Noisy</Badge>
                Steps {noisy.map(([s]) => s).join(" and ")}: too few positives ({noisy.map(([, n]) => n).join(", ")}), noisy. Hollow
                markers and the grey band flag them; read their PR-AUC as noise.
              </p>
            )}
            <details style={{ marginTop: 12 }}>
              <summary className="hint" style={{ cursor: "pointer" }}>
                View chart data as a table
              </summary>
              <div className="table-wrap" style={{ marginTop: 8 }}>
                <table>
                  <caption className="sr-only">PR-AUC per time step</caption>
                  <thead>
                    <tr>
                      <th scope="col">Step</th>
                      {MODELS.map((m) => (
                        <th scope="col" className="r" key={m.key}>
                          {m.name}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {steps.map((s) => (
                      <tr key={s.step} className={s.step >= DRIFT_START ? "row--crit" : ""}>
                        <td className="num">{s.step}</td>
                        {MODELS.map((m) => (
                          <td className="r num" key={m.key}>
                            {fmt(s[m.key], 3)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          </Card>

          <Card
            title="Illicit transaction prevalence"
            sub="Share of labelled transactions that are illicit, per step (class 3 unknown excluded)"
            aside={
              <Badge tone="crit" plain>
                {pct(early?.prevalence, 2)} → {pct(late?.prevalence, 2)} at step {DRIFT_START}
              </Badge>
            }
          >
            <PrevalenceBars
              data={steps}
              summary="Bar chart of the illicit share of labelled transactions per step; it drops after step 43."
            />
          </Card>
        </div>

        <div className="stack">
          <Card title="Drift diagnosis" sub="Steps 1–34 vs 43–49">
            <dl className="kvs">
              <div className="kv">
                <dt>Adversarial AUC</dt>
                <dd>{fmt(d.adversarial_auc_train_vs_drift, 4)}</dd>
              </div>
              <div className="kv">
                <dt>Median KS, top-15 features</dt>
                <dd>{fmt(d.median_top15_ks, 4)}</dd>
              </div>
              <div className="kv">
                <dt>Transferred PR-AUC</dt>
                <dd className="tone-crit">{fmt(d.transferred_pr_auc_drift, 4)}</dd>
              </div>
              <div className="kv">
                <dt>
                  Oracle upper bound
                  <span className="hint block">in-window CV, not deployable</span>
                </dt>
                <dd className="tone-ok">{fmt(d.in_window_cv_pr_auc_drift, 4)}</dd>
              </div>
            </dl>
            <p className="hint" style={{ marginTop: 12, lineHeight: 1.6 }}>
              Verdict: {d.verdict}. Top drifting features: {d.top15_features.slice(0, 5).join(", ").replace(/_feature_/g, " ")}.
            </p>
          </Card>

          <Card title="Degradation summary" sub="PR-AUC, steps 35–42 → 43–49">
            {MODELS.map((m) => {
              const a = w(m.name, "35-42")?.pr_auc ?? null;
              const b = w(m.name, "43-49")?.pr_auc ?? null;
              const change = a && b !== null ? (b / a - 1) * 100 : null;
              return (
                <div className="bar-row" key={m.name}>
                  <div className="bar-row__top">
                    <span className="bar-row__name">{m.name}</span>
                    <span className="bar-row__val tone-crit">{signedPct(change)}</span>
                  </div>
                  <div className="bar" role="img" aria-label={`${m.name}: ${fmt(a, 2)} before, ${fmt(b, 2)} after`}>
                    <div className="bar__fill" style={{ width: `${(b ?? 0) * 100}%`, background: m.color, minWidth: 3 }} />
                  </div>
                  <div className="bar-row__sub">
                    <span>{fmt(a, 3)} pre</span>
                    <span>{fmt(b, 3)} post</span>
                  </div>
                </div>
              );
            })}
            {decision && (
              <div className="notice notice--crit" style={{ marginTop: 16 }}>
                <div>
                  <strong>
                    Monitoring decision (step {decision.time_step}): {ACTION_LABEL[decision.action]}
                  </strong>
                  {decision.primary_reason}
                </div>
              </div>
            )}
          </Card>
        </div>
      </div>

      <Card
        title="Benchmark breakdown"
        sub="Precision, recall and F1 are at each model's saved operating threshold"
        aside={<span className="hint mono">{rows.length} model × window rows</span>}
      >
        <div className="table-wrap">
          <table>
            <caption className="sr-only">Benchmark metrics per model and evaluation window</caption>
            <thead>
              <tr>
                <th scope="col">Window</th>
                <th scope="col">Model</th>
                <th scope="col" className="r">PR-AUC</th>
                <th scope="col" className="r">ROC-AUC</th>
                <th scope="col" className="r">F1</th>
                <th scope="col" className="r">Precision</th>
                <th scope="col" className="r">Recall</th>
                <th scope="col" className="r">Random baseline</th>
                <th scope="col" className="c">Regime</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const drift = r.window === "43-49";
                return (
                  <tr key={`${r.window}-${r.model}`} className={drift ? "row--crit" : ""}>
                    <td className="num">Steps {r.window.replace("-", "–")}</td>
                    <td style={{ fontWeight: 600, color: "var(--ink)" }}>{r.model}</td>
                    <td className="r num" style={{ fontWeight: 600, color: drift ? "var(--crit)" : "var(--accent)" }}>
                      {fmt(r.pr_auc, 3)}
                    </td>
                    <td className="r num">{fmt(r.roc_auc, 3)}</td>
                    <td className="r num">{fmt(r.f1, 3)}</td>
                    <td className="r num">{fmt(r.precision, 3)}</td>
                    <td className="r num">{fmt(r.recall, 3)}</td>
                    <td className="r num">{pct(r.prevalence, 2)}</td>
                    <td className="c">
                      <Badge tone={drift ? "crit" : r.window === "35-42" ? "ok" : "neutral"} plain={r.window === "35-49"}>
                        {drift ? "Drift" : r.window === "35-42" ? "Pre-drift" : "Combined"}
                      </Badge>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </>
  );
}
