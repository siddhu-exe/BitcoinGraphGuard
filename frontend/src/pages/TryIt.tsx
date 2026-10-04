import { useCallback, useMemo, useRef, useState } from "react";
import { Badge, Card, ErrorState, Icon, Skeleton } from "../components/ui";
import { API_URL } from "../lib/api";
import { loadSamples } from "../lib/data";
import { DRIFT_START, fmt, sci } from "../lib/format";
import { useApiAction } from "../lib/useApi";
import { useStatic } from "../lib/useStatic";
import type { PredictionOutput, SampleTx } from "../lib/types";

const N_FEATURES = 165;

interface Parsed {
  payload?: { tx_id: string | number | null; time_step: number; features: Record<string, number> };
  error?: string;
}

function parseInput(text: string): Parsed {
  if (text.trim() === "") return { error: "Load a sample transaction or paste a JSON payload." };
  let obj: unknown;
  try {
    obj = JSON.parse(text);
  } catch (e) {
    return { error: `Invalid JSON: ${(e as Error).message}` };
  }
  const o = obj as { tx_id?: string | number; time_step?: unknown; features?: Record<string, unknown> };
  if (typeof o !== "object" || o === null) return { error: "Payload must be a JSON object." };
  if (!Number.isInteger(o.time_step)) return { error: "`time_step` must be an integer." };
  if (!o.features || typeof o.features !== "object") return { error: "`features` must be an object of feature name → number." };
  const bad = Object.entries(o.features).filter(([, v]) => typeof v !== "number" || !Number.isFinite(v));
  if (bad.length) return { error: `Non-numeric feature value: ${bad[0][0]}${bad.length > 1 ? ` and ${bad.length - 1} more` : ""}.` };
  const count = Object.keys(o.features).length;
  if (count !== N_FEATURES) return { error: `Expected ${N_FEATURES} features, found ${count}.` };
  return {
    payload: { tx_id: o.tx_id ?? null, time_step: o.time_step as number, features: o.features as Record<string, number> },
  };
}

const reliabilityTone = (r: string) => (r === "reliable" ? "ok" : r === "degraded" ? "warn" : "info");

export default function TryIt() {
  const samples = useStatic(loadSamples);
  const predict = useApiAction<PredictionOutput>();
  const [text, setText] = useState("");
  const [loaded, setLoaded] = useState<SampleTx | null>(null);
  const lastBody = useRef<unknown>(null);

  const parsed = useMemo(() => parseInput(text), [text]);
  const step = parsed.payload?.time_step ?? loaded?.time_step ?? 35;
  const [touched, setTouched] = useState(false);

  const load = useCallback(
    (s: SampleTx) => {
      setLoaded(s);
      setTouched(false);
      predict.reset();
      setText(JSON.stringify({ tx_id: s.tx_id, time_step: s.time_step, features: s.features }, null, 2));
    },
    [predict],
  );

  const setStep = (n: number) => {
    if (!parsed.payload) return;
    setText(JSON.stringify({ ...parsed.payload, time_step: n }, null, 2));
  };

  const analyze = () => {
    setTouched(true);
    if (!parsed.payload) return;
    lastBody.current = parsed.payload;
    void predict.post("/predict", parsed.payload);
  };

  const busy = predict.phase === "loading" || predict.phase === "waking";
  const res = predict.data;
  const matchesSample = loaded && res && String(res.tx_id) === loaded.tx_id && res.time_step === loaded.time_step;

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Transaction inference & reliability check</h1>
          <p>
            Score a transaction with the live frozen XGBoost model. The API also reports how far to trust the score at that time step.
          </p>
        </div>
        <div className="row" role="group" aria-label="Sample transactions">
          <span className="hint">Presets:</span>
          {samples.data?.samples.map((s) => (
            <button key={s.tx_id} type="button" className="btn btn--sm" onClick={() => load(s)} title={s.title}>
              {s.y_true ? "Illicit" : "Licit"} · step {s.time_step}
              {s.y_true && s.recorded_score < (samples.data?.threshold ?? 0.435) ? " (missed)" : ""}
            </button>
          ))}
          {samples.loading && <Skeleton h={32} w={220} />}
        </div>
      </div>

      <div className="grid grid--2">
        <Card title="Transaction input" aside={<span className="hint mono">{N_FEATURES} features + time step</span>}>
          <div className="stack">
            <div className="well">
              <div className="row-between">
                <label className="label" htmlFor="step-slider">
                  Time step
                </label>
                <span className="mono" style={{ fontWeight: 700, color: "var(--ink)" }} aria-live="polite">
                  {step}
                </span>
              </div>
              <input
                id="step-slider"
                type="range"
                min={35}
                max={49}
                value={Math.min(49, Math.max(35, step))}
                disabled={!parsed.payload}
                onChange={(e) => setStep(Number(e.target.value))}
                aria-describedby="step-note"
              />
              <p id="step-note" className="hint" style={{ minHeight: 18 }}>
                {step >= DRIFT_START
                  ? `Step ${step} is in the drift regime (≥ ${DRIFT_START}): this model is documented as unreliable here.`
                  : "Steps 35–42: the stable test window. Slide to 43+ to see the reliability context change."}
              </p>
            </div>

            <div className="field">
              <div className="row-between">
                <label htmlFor="payload">Payload (JSON)</label>
                {loaded && (
                  <span className="hint mono">
                    tx {loaded.tx_id} · {loaded.title}
                  </span>
                )}
              </div>
              <textarea
                id="payload"
                className="textarea"
                spellCheck={false}
                value={text}
                onChange={(e) => {
                  setText(e.target.value);
                  setTouched(false);
                }}
                placeholder={'{\n  "tx_id": "…",\n  "time_step": 35,\n  "features": { "Local_feature_1": 0.1, … 165 values }\n}'}
                aria-invalid={touched && !parsed.payload}
                aria-describedby="payload-msg"
              />
              <p id="payload-msg" className={`hint ${touched && parsed.error ? "hint--err" : ""}`} role={touched && parsed.error ? "alert" : undefined}>
                {touched && parsed.error ? parsed.error : parsed.payload ? `Valid: ${N_FEATURES} numeric features.` : parsed.error}
              </p>
            </div>

            <div className="row">
              <button type="button" className="btn" disabled={!samples.data?.samples.length} onClick={() => samples.data && load(samples.data.samples[0])}>
                Load sample transaction
              </button>
              <button type="button" className="btn btn--primary" onClick={analyze} disabled={busy}>
                <Icon name={busy ? "refresh" : "bolt"} size={14} className={busy ? "spin" : undefined} />
                {busy ? "Analyzing…" : "Analyze transaction"}
              </button>
            </div>
            {samples.error && (
              <p className="hint hint--err" role="alert">
                Sample data failed to load: {samples.error.message}{" "}
                <button type="button" className="btn btn--sm" onClick={samples.retry}>
                  Retry
                </button>
              </p>
            )}
            {predict.phase === "waking" || (busy && predict.elapsed >= 2) ? (
              <div className="notice notice--wait" role="status" aria-live="polite">
                <Icon name="refresh" className="spin" />
                <div>
                  <strong>Waking up server, first request can take ~50s</strong>
                  Waiting {predict.elapsed}s of 65s before timing out.
                </div>
              </div>
            ) : null}
          </div>
        </Card>

        <Card
          title="Inference result"
          aside={res ? <span className="hint mono">step {res.time_step}{res.tx_id ? ` · tx ${res.tx_id}` : ""}</span> : undefined}
          foot={
            predict.ms !== undefined && predict.phase === "success" ? (
              <>
                <span>Round trip: {Math.round(predict.ms)} ms (includes network{predict.ms > 5000 ? " and cold start" : ""})</span>
                <span>{API_URL.replace(/^https?:\/\//, "")}</span>
              </>
            ) : undefined
          }
        >
          <div style={{ minHeight: 360 }}>
            {predict.phase === "error" && predict.error ? (
              <ErrorState
                title={
                  predict.error.kind === "timeout"
                    ? "The server did not respond in time"
                    : predict.error.kind === "config"
                      ? "API URL not configured"
                      : predict.error.kind === "http"
                        ? "The API rejected the request"
                        : "API unreachable"
                }
                error={predict.error}
                onRetry={predict.error.kind === "config" ? undefined : () => void predict.post("/predict", lastBody.current)}
              />
            ) : res ? (
              <div className="stack">
                <div className="well">
                  <div className="result-grid">
                    <div>
                      <div className="hint">Illicit probability</div>
                      <div className="prob" aria-live="polite">
                        {res.probability >= 0.001 ? res.probability.toFixed(4) : sci(res.probability)}
                      </div>
                      <div className="hint mono">threshold τ* = {res.threshold}</div>
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <Badge tone={res.binary_classification === 1 ? "crit" : "ok"} large>
                        {res.label}
                      </Badge>
                    </div>
                  </div>
                </div>

                <div className={`notice notice--${res.confidence_context.model_reliability === "reliable" ? "ok" : res.confidence_context.model_reliability === "degraded" ? "warn" : "wait"}`}>
                  <Icon name={res.confidence_context.model_reliability === "reliable" ? "check" : "warning"} />
                  <div>
                    <strong>
                      Reliability: <Badge tone={reliabilityTone(res.confidence_context.model_reliability)} plain>{res.confidence_context.model_reliability}</Badge>
                    </strong>
                    <p style={{ marginTop: 6 }}>{res.confidence_context.reason}</p>
                    <p style={{ marginTop: 4, opacity: 0.85 }}>{res.confidence_context.recommendation}</p>
                  </div>
                </div>

                <dl className="kvs">
                  <div className="kv">
                    <dt>Regime</dt>
                    <dd>{res.confidence_context.regime}</dd>
                  </div>
                  <div className="kv">
                    <dt>Historical PR-AUC at this step</dt>
                    <dd>{fmt(res.confidence_context.historical_pr_auc, 4)}</dd>
                  </div>
                  <div className="kv">
                    <dt>Historical F1 at this step</dt>
                    <dd>{fmt(res.confidence_context.historical_f1, 4)}</dd>
                  </div>
                  {matchesSample && loaded && (
                    <>
                      <div className="kv">
                        <dt>Recorded benchmark score</dt>
                        <dd>{sci(loaded.recorded_score)}</dd>
                      </div>
                      <div className="kv">
                        <dt>Dataset ground truth</dt>
                        <dd>{loaded.y_true ? "illicit" : "licit"}</dd>
                      </div>
                    </>
                  )}
                </dl>
              </div>
            ) : busy ? (
              <div className="stack" aria-busy="true">
                <Skeleton h={120} />
                <Skeleton h={88} />
                <Skeleton h={96} />
              </div>
            ) : (
              <div className="state" style={{ minHeight: 360 }}>
                <Icon name="info" size={22} />
                <p style={{ maxWidth: "40ch" }}>
                  Load a sample transaction, then choose <strong>Analyze transaction</strong>. The score, label and reliability context come from the live API.
                </p>
              </div>
            )}
          </div>
        </Card>
      </div>

      <section className="card row-between" aria-label="Temporal partition guide">
        <div>
          <h2 className="card__title">Temporal partition guide</h2>
          <p className="card__sub">Models are fit on early steps and tested strictly on later ones. Class 3 (unknown) transactions are never used as labels.</p>
        </div>
        <div className="row">
          <Badge plain>Train 1–34</Badge>
          <Badge tone="ok" plain>Test, stable 35–42</Badge>
          <Badge tone="warn" plain>Test, drift 43–49</Badge>
        </div>
      </section>
    </>
  );
}
