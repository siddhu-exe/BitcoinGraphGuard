import { useMemo, useState } from "react";
import { Badge, Card, CardSkeleton, ErrorState, Kpi, Legend } from "../components/ui";
import { StepLines } from "../components/charts";
import { loadMonitoring } from "../lib/data";
import { ACTION_LABEL, CHANNEL_LABEL, DRIFT_START, fmt, pct, signedPct } from "../lib/format";
import { useStatic } from "../lib/useStatic";
import type { ChannelState, LagSafeChannelKey, LagSafeRow, MonitoringData, TriggerAction } from "../lib/types";

type Filter = "ALL" | TriggerAction;
const FILTERS: { id: Filter; label: string }[] = [
  { id: "ALL", label: "All" },
  { id: "NO_ACTION", label: "No action" },
  { id: "RECALIBRATE_ONLY", label: "Recalibrate" },
  { id: "RETRAIN", label: "Retrain" },
];
const tone = (a: string) => (a === "RETRAIN" ? "crit" : a === "RECALIBRATE_ONLY" ? "warn" : "ok");
const statusTone = (s: string) => (/ALERT|CRITICAL|SIGNIFICANT/.test(s) ? "crit" : /MODERATE|WARNING/.test(s) ? "warn" : "ok");

export default function Monitoring() {
  const mon = useStatic(loadMonitoring);
  return (
    <>
      <div className="page-head">
        <div>
          <h1>Drift monitoring</h1>
          <p>
            Retrospective backtest of the monitoring system over steps 35–49. Each step uses only data available at that step, with labels
            arriving one step late.
          </p>
        </div>
      </div>
      {mon.error ? (
        <Card>
          <ErrorState title="Monitoring data failed to load" error={mon.error} onRetry={mon.retry} />
        </Card>
      ) : !mon.data ? (
        <>
          <div className="grid grid--kpi">
            {[0, 1, 2, 3].map((i) => (
              <CardSkeleton key={i} h={72} />
            ))}
          </div>
          <CardSkeleton h={300} title="chart" />
        </>
      ) : (
        <Loaded data={mon.data} />
      )}
    </>
  );
}

function Loaded({ data }: { data: MonitoringData }) {
  const { steps, summary, latest_decision: latest } = data;
  const lastStep = steps[steps.length - 1];
  const first = steps[0];
  const counts = summary.action_counts;
  const adversarial = steps.filter((s) => s.adversarial_auc !== null);

  return (
    <>
      <div className="grid grid--kpi">
        <Kpi
          label="Time horizon"
          value={`Steps ${first.step}–${lastStep.step}`}
          footLeft={`${summary.n_steps} evaluated steps`}
          footRight={data.metadata.generated}
        />
        <Kpi
          crit
          label="Prevalence shift"
          badge={<Badge tone="crit" plain>{lastStep.prevalence_status}</Badge>}
          value={signedPct(lastStep.prevalence_rel_change_pct)}
          tone="crit"
          footLeft={`Rolling vs baseline, step ${lastStep.step}`}
          footRight={`Rolling ${pct(lastStep.rolling_prevalence, 2)}`}
        />
        <Kpi
          label="Oracle F1 · steps 43–49"
          value={<span className="num">{fmt(summary.drift_window_mean_oracle_f1, 3)}</span>}
          footLeft="Mean of per-step oracle thresholds"
          footRight={`Pre-drift frozen ${fmt(summary.pre_drift_mean_frozen_f1, 2)}`}
        />
        <Kpi
          crit={latest.action === "RETRAIN"}
          label="Pipeline state"
          badge={<Badge tone={tone(latest.action)}>{latest.severity}</Badge>}
          value={ACTION_LABEL[latest.action]}
          tone={tone(latest.action)}
          footLeft={`Decision at step ${latest.time_step}`}
          footRight={`${counts.RETRAIN ?? 0} of ${summary.n_steps} steps retrain`}
        />
      </div>

      <div className="grid grid--8-4">
        <Card
          title="Frozen vs adaptive vs oracle F1"
          sub="Illicit-class F1 per step at three thresholds (frozen τ* = 0.435, rolling-adaptive, per-step oracle)"
          aside={
            <Legend
              label="Series"
              items={[
                { name: "Frozen threshold", color: "#64748b", dash: "6 3" },
                { name: "Adaptive threshold", color: "#0d9488" },
                { name: "Oracle (upper bound)", color: "#6366f1", dash: "2 3" },
              ]}
            />
          }
          foot={
            <>
              <span>
                Pre-drift frozen mean: <b className="tone-ok">{fmt(summary.pre_drift_mean_frozen_f1, 3)}</b>
              </span>
              <span>
                Drift frozen: <b className="tone-crit">{fmt(summary.drift_window_mean_frozen_f1, 3)}</b>
              </span>
              <span>
                Drift adaptive: <b className="tone-accent">{fmt(summary.drift_window_mean_adaptive_f1, 3)}</b>
              </span>
            </>
          }
        >
          <StepLines
            data={steps}
            lines={[
              { key: "frozen_f1", name: "Frozen", color: "#64748b", dash: "6 3" },
              { key: "adaptive_f1", name: "Adaptive", color: "#0d9488", width: 2.75 },
              { key: "oracle_f1", name: "Oracle", color: "#6366f1", dash: "2 3" },
            ]}
            summary="Line chart of F1 per step for frozen, adaptive and oracle thresholds. F1 is high through step 42 and collapses from step 43 for all three."
          />
        </Card>

        <Card tone="crit" title="Threshold alert" aside={<Badge tone="crit">Retrain</Badge>}>
          <h3 style={{ fontSize: 18, lineHeight: "24px" }}>Threshold tuning cannot fix this</h3>
          <p style={{ marginTop: 8, lineHeight: 1.6 }}>
            Even choosing the best threshold at every step (oracle), F1 averages {fmt(summary.drift_window_mean_oracle_f1, 3)} over steps
            43–49. The decision engine escalates to retraining.
          </p>
          <dl className="kvs" style={{ marginTop: 16 }}>
            <div className="kv">
              <dt>Oracle F1 (mean)</dt>
              <dd className="tone-crit">{fmt(summary.drift_window_mean_oracle_f1, 3)}</dd>
            </div>
            <div className="kv">
              <dt>Adaptive F1 (mean)</dt>
              <dd>{fmt(summary.drift_window_mean_adaptive_f1, 3)}</dd>
            </div>
            <div className="kv">
              <dt>Frozen F1 (mean)</dt>
              <dd>{fmt(summary.drift_window_mean_frozen_f1, 3)}</dd>
            </div>
          </dl>
          <p className="hint" style={{ marginTop: 12 }}>
            Latest reason: {latest.primary_reason}
          </p>
        </Card>
      </div>

      <LagSafeTable rows={data.lag_safe} delay={latest.label_delay_assumption} />

      <div className="grid grid--3">
        <Card title="Prevalence monitor" sub={`Step ${lastStep.step}`}>
          <dl className="kvs">
            <div className="kv">
              <dt>Step prevalence</dt>
              <dd>{pct(lastStep.prevalence, 2)}</dd>
            </div>
            <div className="kv">
              <dt>Rolling prevalence</dt>
              <dd>{pct(lastStep.rolling_prevalence, 2)}</dd>
            </div>
            <div className="kv">
              <dt>Change vs baseline</dt>
              <dd className="tone-crit">{signedPct(lastStep.prevalence_rel_change_pct)}</dd>
            </div>
          </dl>
        </Card>
        <Card title="Adversarial validation" sub="Domain-classifier AUC at periodic steps">
          <dl className="kvs">
            {adversarial.map((s) => (
              <div className="kv" key={s.step}>
                <dt>Step {s.step}</dt>
                <dd>{fmt(s.adversarial_auc, 4)}</dd>
              </div>
            ))}
          </dl>
        </Card>
        <Card title="Component statuses" sub={`Decision at step ${latest.time_step}`}>
          <dl className="kvs">
            {Object.entries(latest.component_statuses).map(([k, v]) => (
              <div className="kv" key={k}>
                <dt>{k.replace(/_/g, " ")}</dt>
                <dd>
                  <Badge tone={statusTone(v)}>{v}</Badge>
                </dd>
              </div>
            ))}
          </dl>
        </Card>
      </div>
    </>
  );
}

const CHANNEL_COLUMNS: { key: LagSafeChannelKey; label: string; hint?: string }[] = [
  { key: "score_shift", label: "Score shift" },
  { key: "performance", label: "Performance" },
  { key: "prevalence", label: "Prevalence" },
  { key: "feature_shift", label: "Feature shift", hint: "warning only" },
];

function ChannelCell({ state }: { state: ChannelState }) {
  if (state === "CRITICAL") return <Badge tone="crit">Critical</Badge>;
  if (state === "WARNING") return <Badge tone="warn">Warning</Badge>;
  if (state === "OK") return <span className="hint">OK</span>;
  return <span className="hint">n/a</span>;
}

function LagSafeTable({ rows, delay }: { rows: LagSafeRow[]; delay: string }) {
  const [filter, setFilter] = useState<Filter>("ALL");
  const shown = useMemo(() => rows.filter((r) => filter === "ALL" || r.action === filter), [rows, filter]);

  const exportCsv = () => {
    const head = "step,channels_fired,score_psi,score_shift,performance,worst_f1,prevalence,feature_shift,decision\n";
    const body = shown
      .map((r) =>
        [
          r.step,
          r.critical_channels.join("+"),
          r.score_psi ?? "",
          r.channels.score_shift.status,
          r.channels.performance.status,
          r.worst_f1 ?? "",
          r.channels.prevalence.status,
          r.channels.feature_shift.status,
          r.action,
        ].join(","),
      )
      .join("\n");
    const url = URL.createObjectURL(new Blob([head + body + "\n"], { type: "text/csv" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = "monitoring_lag_safe_decisions.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Card
      title="Lag-safe decision table"
      sub={`Which channel fired at each step. ${delay}`}
      aside={
        <div className="row">
          <div className="seg" role="group" aria-label="Decision filter">
            {FILTERS.map((f) => (
              <button key={f.id} type="button" aria-pressed={filter === f.id} onClick={() => setFilter(f.id)}>
                {f.label}
              </button>
            ))}
          </div>
          <button type="button" className="btn btn--sm" onClick={exportCsv}>
            Export CSV
          </button>
        </div>
      }
      foot={
        <>
          <span>Performance reads labels of steps up to t − L only; step t&apos;s own F1 is never an input.</span>
          <span aria-live="polite">
            Showing {shown.length} of {rows.length}
          </span>
        </>
      }
    >
      <div className="notice notice--warn" role="note" style={{ marginBottom: 12 }}>
        <div>
          <strong>Validated on one drift event.</strong>
          <span>
            The channels were checked against a single regime change, the step-43 shift. Their thresholds were fixed before the run, but no
            second event has tested them.
          </span>
        </div>
      </div>
      <div className="table-wrap">
        <table className="lag-safe">
          <caption className="sr-only">Per-step monitoring channels, score PSI and decision under a one-step label delay</caption>
          <thead>
            <tr>
              <th scope="col" className="c">Step</th>
              <th scope="col">Channels fired</th>
              <th scope="col" className="r" title="Population stability index of the model's score distribution against the last 5 labelled steps. 0.25 or more is critical.">
                Score PSI
              </th>
              {CHANNEL_COLUMNS.map((c) => (
                <th scope="col" className="c" key={c.key}>
                  {c.label}
                  {c.hint && <span className="th-hint">{c.hint}</span>}
                </th>
              ))}
              <th scope="col" className="r" title="Worst per-step F1 over the labelled steps the performance channel read">
                Worst F1
              </th>
              <th scope="col" className="c">Decision</th>
            </tr>
          </thead>
          <tbody>
            {shown.length === 0 && (
              <tr>
                <td colSpan={9} className="c hint" style={{ padding: 24 }}>
                  No steps match this filter.
                </td>
              </tr>
            )}
            {shown.map((r) => (
              <tr key={r.step} className={r.step >= DRIFT_START ? "row--crit" : ""}>
                <td className="c num" style={{ fontWeight: 600, color: r.step >= DRIFT_START ? "var(--crit)" : "var(--accent)" }}>
                  #{r.step}
                </td>
                <td className="fired">
                  {r.critical_channels.length === 0 ? (
                    <span className="hint">None</span>
                  ) : (
                    <span className="chip-row" style={{ marginTop: 0 }}>
                      {r.critical_channels.map((n) => (
                        <Badge key={n} tone="crit" plain>
                          {CHANNEL_LABEL[n] ?? n}
                        </Badge>
                      ))}
                    </span>
                  )}
                </td>
                <td className="r num">{fmt(r.score_psi, 3)}</td>
                {CHANNEL_COLUMNS.map((c) => (
                  <td className="c" key={c.key}>
                    <ChannelCell state={r.channels[c.key].status} />
                  </td>
                ))}
                <td className="r num">{fmt(r.worst_f1, 3)}</td>
                <td className="c">
                  <Badge tone={tone(r.action)}>{ACTION_LABEL[r.action]}</Badge>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
