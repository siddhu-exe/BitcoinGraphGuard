import { useCallback, useMemo, useState } from "react";
import { loadSamples } from "../lib/data";
import { prob } from "../lib/format";
import type { PredictionOutput, SampleTx } from "../lib/types";
import { useApiAction } from "../lib/useApi";
import { useStatic } from "../lib/useStatic";
import { AsyncState, Card } from "../components/ui";

type Tone = "ok" | "retrain" | "warn" | "neutral";

const TONE_BY_STATE: Record<string, Tone> = {
  reliable: "ok",
  degraded: "retrain",
  training_reference: "neutral",
  unknown: "warn",
};

const LABEL_BY_STATE: Record<string, string> = {
  reliable: "Reliable",
  degraded: "Degraded",
  training_reference: "Training data",
  unknown: "Unknown",
};

const BADGE_BY_TONE: Record<Tone, string> = {
  ok: "badge badge--ok",
  retrain: "badge badge--retrain",
  warn: "badge badge--warn",
  neutral: "badge badge--neutral",
};

const reliabilityTone = (r: string): Tone => TONE_BY_STATE[r] ?? "neutral";

const reliabilityLabel = (r: string): string => LABEL_BY_STATE[r] ?? "Unknown";

export default function TryIt() {
  const samples = useStatic(loadSamples);
  const predict = useApiAction<PredictionOutput>();
  const [loaded, setLoaded] = useState<SampleTx | null>(null);
  const [step, setStep] = useState(35);

  const stepValid = Number.isInteger(step) && step >= 1 && step <= 60;

  const loadSample = useCallback((s: SampleTx) => {
    setLoaded(s);
    setStep(s.time_step);
    predict.reset();
  }, [predict]);

  const submit = useCallback(() => {
    if (!loaded || !stepValid) return;
    void predict.post("/predict", { tx_id: loaded.tx_id, time_step: step, features: loaded.features });
  }, [loaded, stepValid, step, predict]);

  const busy = predict.phase === "loading" || predict.phase === "waking";
  const res = predict.data;
  const tone: Tone | null = useMemo(
    () => (res ? reliabilityTone(res.confidence_context?.model_reliability ?? "") : null),
    [res],
  );

  if (samples.loading) return <AsyncState loading label="the sample transaction" />;
  if (samples.error || !samples.data) return <AsyncState loading={false} error={samples.error} label="the sample transaction" />;

  const sample = samples.data.samples[0];

  return (
    <div className="page page--tight">
      <section className="head-block">
        <h1 className="h1">Score a transaction</h1>
        <p className="lede">
          Calls the live FastAPI service with all {samples.data.n_features} features. Every answer carries a
          reliability flag for its time step.
        </p>
      </section>

      <section className="split" style={{ alignItems: "flex-start" }}>
        <form
          className="card card--snug"
          style={{ flex: "1 1 360px" }}
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
        >
          <div className="field">
            <label className="field__label" htmlFor="step">
              Time step
            </label>
            <input
              id="step"
              className="input"
              type="number"
              min={1}
              max={60}
              value={step}
              onChange={(e) => setStep(Number(e.target.value))}
              aria-describedby="step-hint"
            />
            <span className="field__hint" id="step-hint">
              35–42 reliable · 43–49 degraded · 50+ unmonitored
            </span>
          </div>

          <div className="field">
            <span className="field__label" id="features-label">
              Features
            </span>
            <div className="sample-box" aria-labelledby="features-label">
              <span
                className={`sample-box__text ${loaded ? "sample-box__text--ok" : "sample-box__text--muted"}`}
              >
                {loaded ? `${samples.data.n_features} features · tx ${loaded.tx_id}` : "No sample loaded"}
              </span>
              <button type="button" className="btn" onClick={() => loadSample(sample)}>
                Load sample
              </button>
            </div>
          </div>

          <button type="submit" className="btn btn--primary" disabled={!loaded || !stepValid || busy}>
            {busy ? "Predicting…" : "Predict"}
          </button>
        </form>

        <Card className="card--wide prediction">
          <div aria-live="polite" style={{ display: "flex", flexDirection: "column", gap: 20, flex: 1 }}>
            {!res && !busy && !predict.error && (
              <div className="prediction__center">
                <strong>No prediction yet</strong>
                <span style={{ fontSize: 14 }}>Load the sample transaction, then press Predict.</span>
              </div>
            )}

            {busy && (
              <div className="waking">
                <div className="waking__track">
                  <div className="waking__fill" />
                </div>
                <strong>Waking up the server</strong>
                <span style={{ fontSize: 14, color: "var(--muted)" }}>
                  The free host sleeps when idle. The first request can take about 50 seconds.
                </span>
              </div>
            )}

            {predict.error && !busy && (
              <div className="error-box" role="alert">
                <span className="error-box__title">
                  {predict.error.kind === "timeout" ? "Request timed out" : "Prediction failed"}
                </span>
                <span>{predict.error.message}</span>
                {predict.error.kind !== "config" && (
                  <button type="button" className="btn" onClick={submit}>
                    Retry
                  </button>
                )}
              </div>
            )}

            {res && !busy && !predict.error && (
              <>
                <div className="prediction__top">
                  <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                    <span style={{ fontSize: 13, color: "var(--muted)" }}>Illicit probability</span>
                    <span className="prediction__prob">{prob(res.probability)}</span>
                  </div>
                  <div className="prediction__side">
                    <span className={res.binary_classification === 1 ? "badge badge--retrain" : "badge badge--ok"}>
                      {res.label}
                    </span>
                    <span className="prediction__meta">
                      threshold {res.threshold} · tx {res.tx_id ?? "—"}
                    </span>
                  </div>
                </div>
                <div className={`reliability reliability--${tone ?? "neutral"}`}>
                  <div className="reliability__top">
                    <span className={BADGE_BY_TONE[tone ?? "neutral"]}>
                      {reliabilityLabel(res.confidence_context?.model_reliability ?? "unknown")}
                    </span>
                    <span style={{ fontSize: 13, color: "var(--muted-strong)" }}>step {res.time_step}</span>
                  </div>
                  <p className="reliability__reason">{res.confidence_context?.reason ?? "No reliability note returned."}</p>
                </div>
                <p className="footnote">
                  Scores rank transactions for review. Treat them as a triage queue, not a verdict.
                </p>
              </>
            )}
          </div>
        </Card>
      </section>
    </div>
  );
}
