import { useMemo, useState } from "react";
import { loadWalkforward } from "../lib/data";
import { fmt, pct } from "../lib/format";
import { useStatic } from "../lib/useStatic";
import { noiseMap } from "../lib/walkforward";
import type { Strategy, WalkforwardData } from "../lib/types";
import { StepLines } from "./charts";
import { Badge, Card, CardSkeleton, ErrorState, Legend } from "./ui";

type Window = "43-49" | "35-42";
const WINDOWS: { id: Window; label: string }[] = [
  { id: "43-49", label: "Drift · 43–49" },
  { id: "35-42", label: "Pre-drift · 35–42" },
];
const STATIC = { name: "Static", color: "#64748b", dash: "6 3" };
const EXPANDING = { name: "Expanding", color: "#0d9488", dash: undefined };
const CEILING = { name: "Max achievable", color: "#6366f1", dash: "2 3" };

/** Primary operating view: precision@K / recall@K for a fixed analyst alert budget. */
export default function TriagePanel() {
  const wf = useStatic(loadWalkforward);
  if (wf.error) {
    return (
      <Card>
        <ErrorState title="Triage data failed to load" error={wf.error} onRetry={wf.retry} />
      </Card>
    );
  }
  if (!wf.data) return <CardSkeleton h={560} title="triage" />;
  return <Loaded wf={wf.data} />;
}

function Loaded({ wf }: { wf: WalkforwardData }) {
  const { lags, ks } = wf.metadata;
  const [window, setWindow] = useState<Window>("43-49");
  const [lag, setLag] = useState<number>(lags[0]);
  const [k, setK] = useState<number>(ks[ks.length - 2] ?? ks[0]);
  const noise = useMemo(() => noiseMap(wf), [wf]);

  const pooled = (strategy: Strategy, kk: number) =>
    wf.triage.pooled.find((r) => r.strategy === strategy && r.lag === lag && r.window === window && r.k === kk);
  const validation = (strategy: Strategy) =>
    wf.triage.validation_threshold.find((r) => r.strategy === strategy && r.lag === lag && r.window === window);

  const recallRows = useMemo(
    () =>
      wf.steps.map((s) => {
        const at = (strategy: Strategy) =>
          wf.triage.steps.find((r) => r.strategy === strategy && r.lag === lag && r.k === k && r.step === s.step);
        return {
          step: s.step,
          static: at("static")?.recall ?? null,
          expanding: at("expanding")?.recall ?? null,
          ceiling: at("static")?.ceiling_recall ?? null,
        };
      }),
    [wf, lag, k],
  );
  const thresholdRows = useMemo(
    () =>
      wf.steps.map((s) => {
        const at = (strategy: Strategy) => wf.per_step.find((r) => r.strategy === strategy && r.lag === lag && r.step === s.step);
        return { step: s.step, static: at("static")?.threshold ?? null, expanding: at("expanding")?.threshold ?? null };
      }),
    [wf, lag],
  );

  const range = (key: "static" | "expanding") => {
    const v = thresholdRows.filter((r) => r.step >= 43).map((r) => r[key]).filter((x): x is number => x !== null);
    return v.length ? { lo: Math.min(...v), hi: Math.max(...v) } : undefined;
  };
  const exRange = range("expanding");
  const stRange = range("static");
  const zero =
    wf.triage.validation_threshold.find((r) => r.strategy === "expanding" && r.lag === lag && r.window === "43-49")
      ?.steps_with_zero_alerts ?? 0;

  const metricCells = (strategy: Strategy, kk: number) => {
    const r = pooled(strategy, kk);
    return (
      <>
        <td className="r num">{pct(r?.precision, 1)}</td>
        <td className="r num" style={{ fontWeight: 600, color: "var(--ink)" }}>
          {pct(r?.recall, 1)}
        </td>
      </>
    );
  };

  return (
    <Card
      title="Triage · top-K alerts per step"
      sub={`Analysts review the K highest-scored transactions each step. Pooled over steps ${window.replace("-", "–")}, labelled transactions only, label delay L = ${lag}.`}
      aside={
        <div className="row">
          <div className="seg" role="group" aria-label="Evaluation window">
            {WINDOWS.map((w) => (
              <button key={w.id} type="button" aria-pressed={window === w.id} onClick={() => setWindow(w.id)}>
                {w.label}
              </button>
            ))}
          </div>
          <div className="seg" role="group" aria-label="Label delay">
            {lags.map((l) => (
              <button key={l} type="button" aria-pressed={lag === l} onClick={() => setLag(l)}>
                L = {l}
              </button>
            ))}
          </div>
        </div>
      }
      foot={
        <>
          <span>Max achievable recall = Σ min(K, positives in step) ÷ total positives</span>
          <span>Static is trained through step {wf.retrains.find((r) => r.strategy === "static" && r.lag === lag)?.trained_through}</span>
        </>
      }
    >
      <div className="table-wrap">
        <table>
          <caption className="sr-only">
            Precision and recall at K alerts per step, static versus expanding model, with the maximum achievable recall
          </caption>
          <thead>
            <tr>
              <th scope="col" rowSpan={2}>Alert budget</th>
              <th scope="colgroup" colSpan={2} className="c grp">Static</th>
              <th scope="colgroup" colSpan={2} className="c grp">Expanding</th>
              <th scope="col" rowSpan={2} className="r">Max achievable recall</th>
            </tr>
            <tr>
              <th scope="col" className="r">Precision</th>
              <th scope="col" className="r">Recall</th>
              <th scope="col" className="r">Precision</th>
              <th scope="col" className="r">Recall</th>
            </tr>
          </thead>
          <tbody>
            {ks.map((kk) => (
              <tr key={kk}>
                <th scope="row" className="rowhead num">Top {kk} / step</th>
                {metricCells("static", kk)}
                {metricCells("expanding", kk)}
                <td className="r num tone-accent" style={{ fontWeight: 600 }}>{pct(pooled("static", kk)?.max_recall, 1)}</td>
              </tr>
            ))}
            <tr className="row--secondary">
              <th scope="row" className="rowhead">
                Validation threshold <Badge tone="neutral" plain>Secondary</Badge>
                <span className="hint block">
                  {fmt(validation("static")?.alerts_per_step, 1)} vs {fmt(validation("expanding")?.alerts_per_step, 1)} alerts per step
                </span>
              </th>
              <td className="r num">{pct(validation("static")?.precision, 1)}</td>
              <td className="r num">{pct(validation("static")?.recall, 1)}</td>
              <td className="r num">{pct(validation("expanding")?.precision, 1)}</td>
              <td className="r num">{pct(validation("expanding")?.recall, 1)}</td>
              <td className="r hint">n/a, volume is not fixed</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="triage-charts">
        <div>
          <div className="row-between chart-head">
            <h3>Recall at top {k} per step</h3>
            <div className="seg" role="group" aria-label="Alert budget K">
              {ks.map((kk) => (
                <button key={kk} type="button" aria-pressed={k === kk} onClick={() => setK(kk)}>
                  K = {kk}
                </button>
              ))}
            </div>
          </div>
          <Legend label="Series" items={[STATIC, EXPANDING, CEILING]} />
          <div style={{ height: 8 }} />
          <StepLines
            data={recallRows}
            lines={[
              { key: "ceiling", name: CEILING.name, color: CEILING.color, dash: CEILING.dash, width: 1.75 },
              { key: "static", name: STATIC.name, color: STATIC.color, dash: STATIC.dash },
              { key: "expanding", name: EXPANDING.name, color: EXPANDING.color },
            ]}
            noise={noise}
            height="chart--md"
            formatter={(v) => `${(v * 100).toFixed(1)}%`}
            summary={`Line chart of recall at the top ${k} alerts per step for static and expanding models against the maximum achievable recall, steps 35 to 49, label delay ${lag}.`}
          />
        </div>

        <div>
          <div className="row-between chart-head">
            <h3>Validation threshold per step</h3>
            <Badge tone="neutral" plain>Secondary</Badge>
          </div>
          <Legend label="Series" items={[STATIC, EXPANDING]} />
          <div style={{ height: 8 }} />
          <StepLines
            data={thresholdRows}
            lines={[
              { key: "static", name: STATIC.name, color: STATIC.color, dash: STATIC.dash },
              { key: "expanding", name: EXPANDING.name, color: EXPANDING.color },
            ]}
            noise={noise}
            height="chart--md"
            bandLabel={false}
            summary={`Line chart of the validation-selected decision threshold per step for static and expanding models, label delay ${lag}.`}
          />
          {exRange && (
            <p className="hint chart-note">
              Expanding&apos;s threshold swings {fmt(exRange.hi, 3)} to {fmt(exRange.lo, 3)} over steps 43–49
              {stRange && stRange.lo === stRange.hi ? `; static holds ${fmt(stRange.lo, 3)}` : ""}
              {zero > 0 ? `, and it raised no alert at ${zero} step${zero > 1 ? "s" : ""}` : ""}. That instability is why a fixed alert
              budget is the primary rule.
            </p>
          )}
        </div>
      </div>
    </Card>
  );
}
