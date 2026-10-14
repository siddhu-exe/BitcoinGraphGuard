import { Link } from "react-router-dom";
import { actionLabel } from "../lib/derived";
import { useMonitoringStatus } from "../lib/status";

const dotColor = (action: string) =>
  action === "RETRAIN" ? "var(--drift)" : action === "NO_ACTION" ? "var(--badge-ok-ink)" : "var(--muted)";

export default function StatusStrip() {
  const s = useMonitoringStatus();
  const retrain = s.action === "RETRAIN";

  return (
    <section className="status-strip" aria-label="Live system status">
      <div className="status-strip__left">
        <span className="status-dot" style={{ background: dotColor(s.action) }} aria-hidden="true" />
        {s.step !== null ? (
          <span style={{ fontWeight: 500 }}>
            Live monitor: {actionLabel(s.action).toLowerCase()} recommended
          </span>
        ) : (
          <span style={{ fontWeight: 500 }}>Live monitor: checking status…</span>
        )}
        <span className="status-strip__note">
          {retrain
            ? "The latest data no longer matches what the model learned from."
            : "The monitor is reading the live signals."}
        </span>
      </div>
      <Link to="/monitoring" className="status-strip__link">
        How the alarm works →
      </Link>
    </section>
  );
}
