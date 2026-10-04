import { Badge, Card, CardSkeleton, ErrorState, Legend } from "../components/ui";
import { LagCurve, StepLines, type LagPoint } from "../components/charts";
import { loadWalkforward } from "../lib/data";
import { fmt } from "../lib/format";
import { useStatic } from "../lib/useStatic";
import { noiseMap } from "../lib/walkforward";
import type { Strategy, WalkforwardData, WalkforwardPooled } from "../lib/types";

const SERIES = [
  { key: "static", name: "Static", color: "#64748b", dash: "6 3" },
  { key: "expanding", name: "Expanding", color: "#0d9488", dash: undefined },
] as const;

const ci = (p?: WalkforwardPooled) => (p?.ci_lo != null && p.ci_hi != null ? `[${fmt(p.ci_lo, 3)}, ${fmt(p.ci_hi, 3)}]` : "—");

export default function Retraining() {
  const wf = useStatic(loadWalkforward);
  return (
    <>
      <div className="page-head">
        <div>
          <h1>Retraining</h1>
          <p>
            Walk-forward test of whether refitting the model as labels arrive recovers performance after the step-43 regime shift. Static
            keeps the model trained on steps 1–34; expanding refits on every labelled step available at decision time.
          </p>
        </div>
      </div>

      {wf.error ? (
        <Card>
          <ErrorState title="Walk-forward data failed to load" error={wf.error} onRetry={wf.retry} />
        </Card>
      ) : !wf.data ? (
        <>
          <CardSkeleton h={120} title="summary" />
          <div className="grid grid--2">
            <CardSkeleton h={300} title="per-step chart" />
            <CardSkeleton h={300} title="label-delay chart" />
          </div>
        </>
      ) : (
        <Loaded wf={wf.data} />
      )}
    </>
  );
}

function Loaded({ wf }: { wf: WalkforwardData }) {
  const noise = noiseMap(wf);
  const noisy = Object.entries(noise);

  // (a) per-step PR-AUC at L = 1
  const perStep = wf.steps.map((s) => {
    const at = (strategy: Strategy) => wf.per_step.find((r) => r.strategy === strategy && r.lag === 1 && r.step === s.step);
    return { step: s.step, static: at("static")?.pr_auc ?? null, expanding: at("expanding")?.pr_auc ?? null };
  });
  const retrainSteps = wf.retrains.filter((r) => r.strategy === "expanding" && r.lag === 1 && r.retrained).map((r) => r.step);
  const staticTrainedThrough = wf.retrains.find((r) => r.strategy === "static" && r.lag === 1)?.trained_through;

  // (b) pooled 43-49 PR-AUC against label delay
  const pooled = (strategy: Strategy, lag: number) =>
    wf.pooled.find((p) => p.strategy === strategy && p.lag === lag && p.window === "43-49");
  const lagRows = wf.metadata.lags.map((lag) => ({ lag, st: pooled("static", lag), ex: pooled("expanding", lag) }));
  const curve: LagPoint[] = lagRows.map(({ lag, st, ex }) => ({
    lag,
    static: st?.pr_auc ?? null,
    static_ci: st?.ci_lo != null && st.ci_hi != null ? [st.ci_lo, st.ci_hi] : null,
    expanding: ex?.pr_auc ?? null,
    expanding_ci: ex?.ci_lo != null && ex.ci_hi != null ? [ex.ci_lo, ex.ci_hi] : null,
  }));
  const nIllicit = lagRows[0]?.ex?.n_illicit;

  return (
    <>
      <div className="notice notice--warn callout" role="note">
        <div>
          <strong>Retraining works only when labels arrive within about 1–2 steps. The real delay is unknown.</strong>
          <span>
            Every result here assumes a fixed label delay L: the labels of step s become usable at step s + L. L = 1 is the best case
            tested.
          </span>
        </div>
      </div>

      <div className="grid grid--2" style={{ alignItems: "start" }}>
        <Card
          title="PR-AUC by time step · label delay L = 1"
          sub="Per-step PR-AUC, static vs expanding, steps 35–49"
          aside={<Legend label="Series" items={SERIES.map((s) => ({ name: s.name, color: s.color, dash: s.dash }))} />}
          foot={
            <>
              <span>
                Static: trained through step {staticTrainedThrough ?? "—"}, never refit
              </span>
              <span>
                Expanding: refit at steps {retrainSteps[0]}–{retrainSteps[retrainSteps.length - 1]} ({retrainSteps.length} refits)
              </span>
            </>
          }
        >
          <StepLines
            data={perStep}
            lines={SERIES.map((s) => ({ key: s.key, name: s.name, color: s.color, dash: s.dash }))}
            noise={noise}
            summary="Line chart of per-step PR-AUC for the static and expanding models over steps 35 to 49 at label delay 1. Expanding stays high while static collapses from step 43."
          />
          {noisy.length > 0 && (
            <p className="hint chart-note">
              <Badge tone="warn">Noisy</Badge>
              Steps {noisy.map(([s]) => s).join(", ")}: too few positives ({noisy.map(([, n]) => n).join(", ")}), hollow markers. Per-step
              PR-AUC there is not interpreted.
            </p>
          )}
        </Card>

        <Card
          title="Pooled PR-AUC vs label delay · steps 43–49"
          sub={`Pooled over ${nIllicit ?? "—"} illicit labels in the drift window; bands are 95% cluster-bootstrap intervals over time steps`}
          aside={
            <Legend
              label="Series"
              items={[
                { name: "Expanding", color: "#0d9488" },
                { name: "Static", color: "#64748b", dash: "6 3" },
              ]}
            />
          }
          foot={
            <>
              <span>Shaded bands: 95% CI</span>
              <span>Only 7 steps in the window, so intervals are approximate and wide</span>
            </>
          }
        >
          <LagCurve
            data={curve}
            summary="Pooled PR-AUC on steps 43 to 49 for the expanding and static models at label delays 1, 2, 3 and 5, with confidence bands. Expanding is highest at delay 1 and falls toward static as the delay grows. A table below lists every value."
          />
          <div className="table-wrap" style={{ marginTop: 12 }}>
            <table>
              <caption className="sr-only">Pooled PR-AUC on steps 43–49 by label delay, with 95% cluster-bootstrap intervals</caption>
              <thead>
                <tr>
                  <th scope="col">Label delay</th>
                  <th scope="col" className="r">Static [95% CI]</th>
                  <th scope="col" className="r">Expanding [95% CI]</th>
                </tr>
              </thead>
              <tbody>
                {lagRows.map(({ lag, st, ex }) => (
                  <tr key={lag}>
                    <td className="num">L = {lag}</td>
                    <td className="r num">
                      {fmt(st?.pr_auc, 3)} <span className="hint ci">{ci(st)}</span>
                    </td>
                    <td className="r num" style={{ fontWeight: 600, color: "var(--ink)" }}>
                      {fmt(ex?.pr_auc, 3)} <span className="hint ci">{ci(ex)}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </>
  );
}
