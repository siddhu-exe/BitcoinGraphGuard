import type { StepColumn } from "../lib/derived";
import { maxPositives } from "../lib/derived";

export default function PerStepColumns({ columns }: { columns: StepColumn[] }) {
  const maxPos = maxPositives(columns) || 1;
  const summary = columns
    .filter((c) => c.noisy)
    .map((c) => `step ${c.step} has only ${c.positives} illicit labels`)
    .join("; ");
  return (
    <div className="steps-scroll">
      <div
        className="steps-grid"
        role="group"
        aria-label={`PR-AUC and illicit labels per step 35 to 49. ${summary ? summary + "." : ""}`}
      >
        {columns.map((c) => (
          <div className="step" key={c.step}>
            <span className="step__pr">{c.scored ? (c.noisy ? "—" : (c.pr?.toFixed(2) ?? "n/a")) : ""}</span>
            <div className={`step__track${c.scored ? "" : " step__track--unscored"}`}>
              {c.scored && !c.noisy && c.pr !== null && (
                <div className="step__prbar" style={{ height: `${Math.min(1, c.pr) * 100}%` }} />
              )}
            </div>
            <div className="step__posbox">
              <div className="step__posbar" style={{ height: `${Math.max(3, (c.positives / maxPos) * 100)}%` }} />
            </div>
            <span className="step__pos">{c.positives}</span>
            <span className="step__num">{c.step}</span>
            {c.noisy && <span className="badge badge--noisy">noisy</span>}
          </div>
        ))}
      </div>
    </div>
  );
}
