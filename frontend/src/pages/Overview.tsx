import { Link } from "react-router-dom";
import { Badge, Card, CardSkeleton, ErrorState, Kpi, Legend } from "../components/ui";
import { StepLines } from "../components/charts";
import StatusBanner, { toDecision } from "../components/StatusBanner";
import TriagePanel from "../components/TriagePanel";
import { loadMonitoring, loadPerformance } from "../lib/data";
import { ACTION_LABEL, fmt, pct, signedPct } from "../lib/format";
import { MODELS } from "../lib/models";
import { useApi } from "../lib/useApi";
import { useStatic } from "../lib/useStatic";
import type { HealthResponse, MonitoringStatusResponse, PerformanceData, WindowMetric } from "../lib/types";

const win = (d: PerformanceData, model: string, w: WindowMetric["window"]) =>
  d.windows.find((x) => x.model === model && x.window === w);

export default function Overview() {
  const perf = useStatic(loadPerformance);
  const mon = useStatic(loadMonitoring);
  const status = useApi<MonitoringStatusResponse>("/monitoring/status");
  const health = useApi<HealthResponse>("/health");

  const decision = toDecision(status, mon.data?.latest_decision);

  return (
    <>
      <h1 className="sr-only">Overview</h1>
      <StatusBanner live={status} decision={decision} generated={mon.data?.metadata.generated} onRetry={status.retry} />

      {perf.error ? (
        <Card>
          <ErrorState title="Benchmark data failed to load" error={perf.error} onRetry={perf.retry} />
        </Card>
      ) : (
        <KpiRow perf={perf.data} decision={decision} health={health} />
      )}

      <TriagePanel />

      <div className="grid grid--7-5">
        {perf.data ? <Benchmarks perf={perf.data} /> : <CardSkeleton h={260} title="benchmarks" />}
        {perf.data ? <Findings perf={perf.data} mon={mon.data} /> : <CardSkeleton h={260} title="findings" />}
      </div>

      <div className="grid grid--8-4">
        {perf.data ? <Trajectory perf={perf.data} /> : <CardSkeleton h={300} title="trajectory" />}
        {mon.data ? <DecisionLog steps={mon.data.steps.slice(-5).reverse()} /> : <CardSkeleton h={240} title="decisions" />}
      </div>
    </>
  );
}

function KpiRow({
  perf,
  decision,
  health,
}: {
  perf?: PerformanceData;
  decision?: ReturnType<typeof toDecision>;
  health: ReturnType<typeof useApi<HealthResponse>>;
}) {
  const full = perf && win(perf, "XGBoost", "35-49");
  const early = perf && win(perf, "XGBoost", "35-42");
  const late = perf && win(perf, "XGBoost", "43-49");
  const change = late?.pr_auc != null && early?.pr_auc ? (late.pr_auc / early.pr_auc - 1) * 100 : null;

  const apiUp = health.phase === "success" && health.data;
  const apiWaiting = health.phase === "loading" || health.phase === "waking" || health.phase === "idle";

  return (
    <div className="grid grid--kpi">
      <Kpi
        label="PR-AUC · steps 35–49"
        value={<span className="num">{fmt(full?.pr_auc, 3)}</span>}
        tone="accent"
        footLeft={`Random baseline ${pct(full?.prevalence, 1)}`}
        footRight="Frozen XGBoost"
      />
      <Kpi
        crit
        label={
          <>
            <span aria-hidden="true">● </span>PR-AUC · steps 43–49
          </>
        }
        badge={<Badge tone="crit" plain>Alert</Badge>}
        value={<span className="num">{fmt(late?.pr_auc, 3)}</span>}
        tone="crit"
        footLeft={`${signedPct(change)} vs steps 35–42`}
        footRight={`Random baseline ${pct(late?.prevalence, 1)}`}
      />
      <Kpi
        label="Monitoring decision"
        value={decision ? ACTION_LABEL[decision.action] ?? decision.action : "—"}
        tone={decision?.action === "RETRAIN" ? "crit" : decision?.action === "RECALIBRATE_ONLY" ? "warn" : "ok"}
        footLeft={decision ? `Step ${decision.step} · ${decision.severity}` : ""}
        footRight={decision?.source === "cached" ? "Cached" : decision ? "Live" : ""}
      />
      <Kpi
        label="API status"
        badge={
          apiUp ? (
            <Badge tone={health.data?.status === "healthy" ? "ok" : "warn"}>{health.data?.status === "healthy" ? "Online" : health.data?.status}</Badge>
          ) : apiWaiting ? (
            <Badge tone="info">{health.phase === "waking" ? "Waking" : "Checking"}</Badge>
          ) : (
            <Badge tone="crit">Offline</Badge>
          )
        }
        value={
          apiUp ? (
            <>
              <span className="num">{Math.round(health.ms ?? 0)}</span> <small>ms round trip</small>
            </>
          ) : apiWaiting ? (
            <small style={{ fontSize: 13 }}>{health.phase === "waking" ? `Waking up server… ${health.elapsed}s` : "Contacting API…"}</small>
          ) : (
            <small style={{ fontSize: 13 }}>Unreachable</small>
          )
        }
        footLeft={apiUp ? `${health.data?.n_features} features · τ* ${health.data?.operating_threshold}` : health.error?.kind === "config" ? "VITE_API_URL not set" : ""}
        footRight={
          apiUp && health.data?.using_fallback_metrics ? (
            <span className="tone-warn">fallback metrics</span>
          ) : !apiUp && !apiWaiting ? (
            <button type="button" className="btn btn--sm" onClick={health.retry}>
              Retry
            </button>
          ) : (
            ""
          )
        }
      />
    </div>
  );
}

function Benchmarks({ perf }: { perf: PerformanceData }) {
  const rows = MODELS.map((m) => ({
    ...m,
    full: win(perf, m.name, "35-49"),
    early: win(perf, m.name, "35-42"),
    late: win(perf, m.name, "43-49"),
  })).sort((a, b) => (b.full?.pr_auc ?? 0) - (a.full?.pr_auc ?? 0));
  return (
    <Card
      title="Model benchmark comparison"
      aside={<span className="hint mono">PR-AUC · steps 35–49</span>}
      foot={
        <>
          <span>Test set: {rows[0].full?.n.toLocaleString()} labelled transactions</span>
          <span>Random baseline: {pct(rows[0].full?.prevalence, 2)}</span>
        </>
      }
    >
      {rows.map((r, i) => (
        <div className="bar-row" key={r.name}>
          <div className="bar-row__top">
            <span className="bar-row__name">
              {r.name}
              {i === 0 && <span className="hint"> (best)</span>}
            </span>
            <span className="bar-row__val">{fmt(r.full?.pr_auc, 3)}</span>
          </div>
          <div
            className="bar"
            role="img"
            aria-label={`${r.name} PR-AUC ${fmt(r.full?.pr_auc, 3)} over steps 35 to 49`}
          >
            <div className="bar__fill" style={{ width: `${(r.full?.pr_auc ?? 0) * 100}%`, background: r.color }} />
          </div>
          <div className="bar-row__sub">
            <span>35–42: {fmt(r.early?.pr_auc, 3)}</span>
            <span className="tone-crit">43–49: {fmt(r.late?.pr_auc, 3)}</span>
          </div>
        </div>
      ))}
    </Card>
  );
}

function Findings({ perf, mon }: { perf: PerformanceData; mon?: import("../lib/types").MonitoringData }) {
  const d = perf.drift_diagnosis;
  const late = MODELS.map((m) => win(perf, m.name, "43-49")?.pr_auc ?? 0);
  const maxLate = Math.max(...late);
  const x = win(perf, "XGBoost", "35-42");
  const y = win(perf, "XGBoost", "43-49");
  const s = mon?.summary;
  return (
    <Card
      title="Drift findings"
      aside={<Badge tone="crit">Drift detected</Badge>}
      foot={
        <>
          <span className="tone-crit">
            Action: {mon ? ACTION_LABEL[mon.latest_decision.action] : "—"}
          </span>
          <Link to="/monitoring" style={{ color: "var(--accent)" }}>
            Full decision matrix →
          </Link>
        </>
      }
    >
      <div className="finding">
        <span className="dot dot--crit" aria-hidden="true" />
        <div>
          <h3>All four architectures collapse after step 43</h3>
          <p>
            Every model scores PR-AUC ≤ {fmt(maxLate, 3)} on steps 43–49 against a {pct(y?.prevalence, 2)} random baseline.
          </p>
        </div>
      </div>
      <div className="finding">
        <span className="dot dot--warn" aria-hidden="true" />
        <div>
          <h3>Covariate shift, not missing signal</h3>
          <p>
            Adversarial validation AUC is {fmt(d.adversarial_auc_train_vs_drift, 4)}; median KS of the top-15 drifting features is{" "}
            {fmt(d.median_top15_ks, 4)}. <strong>Oracle upper bound, not deployable:</strong> a model cross-validated on steps 43–49&apos;s
            own labels reaches PR-AUC {fmt(d.in_window_cv_pr_auc_drift, 3)}, which shows the signal exists but uses the very labels being
            predicted.
          </p>
        </div>
      </div>
      <div className="finding">
        <span className="dot dot--crit" aria-hidden="true" />
        <div>
          <h3>Illicit prevalence regime change</h3>
          <p>
            The illicit share of labelled transactions falls from {pct(x?.prevalence, 2)} (steps 35–42) to {pct(y?.prevalence, 2)} (steps
            43–49).
          </p>
        </div>
      </div>
      {s?.drift_window_mean_oracle_f1 != null && (
        <div className="finding">
          <span className="dot dot--warn" aria-hidden="true" />
          <div>
            <h3>Threshold tuning does not recover it</h3>
            <p>
              Mean per-step F1 over steps 43–49: frozen {fmt(s.drift_window_mean_frozen_f1, 3)}, adaptive{" "}
              {fmt(s.drift_window_mean_adaptive_f1, 3)}, even the oracle threshold only {fmt(s.drift_window_mean_oracle_f1, 3)}.
            </p>
          </div>
        </div>
      )}
    </Card>
  );
}

function Trajectory({ perf }: { perf: PerformanceData }) {
  const pre = perf.steps.filter((s) => s.step < 43 && s.xgboost != null);
  const lo = Math.min(...pre.map((s) => s.xgboost as number));
  const hi = Math.max(...pre.map((s) => s.xgboost as number));
  return (
    <Card
      title="PR-AUC by time step"
      sub="Frozen models on the test window, steps 35 to 49 (one step ≈ two weeks)"
      aside={<Legend label="Series" items={MODELS.map((m) => ({ name: m.name, color: m.hex, dash: m.dash }))} />}
      foot={
        <>
          <span>
            XGBoost PR-AUC in steps 35–42: {fmt(lo, 3)} to {fmt(hi, 3)}
          </span>
          <Link to="/performance" style={{ color: "var(--accent)" }}>
            Full benchmark →
          </Link>
        </>
      }
    >
      <StepLines
        data={perf.steps}
        lines={MODELS.map((m) => ({ key: m.key, name: m.name, color: m.hex, dash: m.dash }))}
        summary={`Line chart of PR-AUC per time step for four models. All models score well in steps 35 to 42 and fall sharply from step 43.`}
      />
    </Card>
  );
}

function DecisionLog({ steps }: { steps: import("../lib/types").MonitoringStep[] }) {
  return (
    <Card title="Recent monitoring decisions" sub={`Backtest, steps ${steps[steps.length - 1]?.step}–${steps[0]?.step}`}>
      <ul className="log">
        {steps.map((s) => (
          <li key={s.step}>
            <span>
              Step <span className="mono">{s.step}</span>
            </span>
            <Badge tone={s.action === "RETRAIN" ? "crit" : s.action === "RECALIBRATE_ONLY" ? "warn" : "ok"}>
              {ACTION_LABEL[s.action]}
            </Badge>
          </li>
        ))}
      </ul>
    </Card>
  );
}
