import { Link } from "react-router-dom";
import { API_URL } from "../lib/api";
import { ACTION_LABEL, CHANNEL_LABEL, channelTone } from "../lib/format";
import type { ApiState } from "../lib/useApi";
import type { CachedDecision, ChannelStatus, MonitoringStatusResponse } from "../lib/types";
import { Badge, Icon } from "./ui";

export interface Decision {
  step: number;
  action: string;
  severity: string;
  reason: string;
  source: "live" | "cached";
  /** Present only when the source reports it (older API deployments omit these). */
  labelDelay?: string;
  channels?: ChannelStatus[];
}

export function toDecision(
  live: ApiState<MonitoringStatusResponse>,
  cached: CachedDecision | undefined,
): Decision | undefined {
  if (live.phase === "success" && live.data) {
    return {
      step: live.data.latest_monitored_step,
      action: live.data.trigger_action,
      severity: live.data.severity,
      reason: live.data.primary_reason,
      source: "live",
      labelDelay: live.data.label_delay_assumption ?? undefined,
      channels: live.data.channel_breakdown
        ? Object.entries(live.data.channel_breakdown).map(([name, c]) => ({ name, ...c }))
        : undefined,
    };
  }
  if (cached) {
    return {
      step: cached.time_step,
      action: cached.action,
      severity: cached.severity,
      reason: cached.primary_reason,
      source: "cached",
      labelDelay: cached.label_delay_assumption,
      channels: cached.channels,
    };
  }
  return undefined;
}

const TITLE: Record<string, string> = {
  RETRAIN: "Model retrain required",
  RECALIBRATE_ONLY: "Threshold recalibration recommended",
  NO_ACTION: "No action required",
};

export default function StatusBanner({
  live,
  decision,
  generated,
  onRetry,
}: {
  live: ApiState<MonitoringStatusResponse>;
  decision?: Decision;
  generated?: string;
  onRetry: () => void;
}) {
  const waiting = live.phase === "loading" || live.phase === "waking" || live.phase === "idle";

  if (waiting) {
    const waking = live.phase === "waking";
    return (
      <div className="banner banner--wait" role="status" aria-live="polite">
        <div className="banner__main">
          <span className="banner__icon">
            <Icon name="refresh" className={waking ? "spin" : undefined} />
          </span>
          <div>
            <h2>{waking ? "Waking up server, first request can take ~50s" : "Checking live monitoring status…"}</h2>
            <p>
              {waking
                ? `Waiting ${live.elapsed}s. The API runs on a free tier that sleeps when idle.`
                : "Contacting the inference service."}
              {decision && ` Last saved decision: ${ACTION_LABEL[decision.action] ?? decision.action} at step ${decision.step}.`}
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (!decision) {
    return (
      <div className="banner banner--warn" role="alert">
        <div className="banner__main">
          <span className="banner__icon">
            <Icon name="warning" />
          </span>
          <div>
            <h2>Monitoring status unavailable</h2>
            <p>{live.error?.message}</p>
          </div>
        </div>
        <button type="button" className="btn btn--sm" onClick={onRetry}>
          <Icon name="refresh" size={14} /> Retry
        </button>
      </div>
    );
  }

  const tone = decision.action === "RETRAIN" ? "crit" : decision.action === "RECALIBRATE_ONLY" ? "warn" : "ok";
  const cached = decision.source === "cached";
  return (
    <div className={`banner banner--${tone}`} role={tone === "crit" ? "alert" : "status"}>
      <div className="banner__main">
        <span className="banner__icon">
          <Icon name={tone === "ok" ? "check" : "warning"} />
        </span>
        <div>
          <h2>
            {TITLE[decision.action] ?? decision.action} · step {decision.step}
          </h2>
          <p>{decision.reason}</p>
          <ChannelMeta decision={decision} />
          {cached && (
            <p className="hint" style={{ marginTop: 4 }}>
              <strong>Cached decision</strong> from the {generated ?? "saved"} backtest report.{" "}
              {live.error?.kind === "config"
                ? `Set VITE_API_URL to read live status.`
                : `Live API unreachable${API_URL ? ` (${API_URL.replace(/^https?:\/\//, "")})` : ""}: ${live.error?.message ?? ""}`}
            </p>
          )}
        </div>
      </div>
      <div className="banner__actions">
        <Badge tone={cached ? "neutral" : tone === "crit" ? "crit" : tone} plain={cached}>
          {cached ? "Cached" : `Live · ${decision.severity}`}
        </Badge>
        {cached && (
          <button type="button" className="btn btn--sm" onClick={onRetry}>
            <Icon name="refresh" size={14} /> Retry
          </button>
        )}
        <Link to="/monitoring" className="btn btn--sm">
          View details
        </Link>
      </div>
    </div>
  );
}

/** Label-delay assumption and the channels currently firing; renders nothing when the source omits them. */
function ChannelMeta({ decision }: { decision: Decision }) {
  const { labelDelay, channels } = decision;
  if (!labelDelay && !channels) return null;
  const firing = (channels ?? [])
    .filter((c) => c.status === "CRITICAL" || c.status === "WARNING")
    .sort((a, b) => Number(b.status === "CRITICAL") - Number(a.status === "CRITICAL"));
  return (
    <div className="banner__meta">
      {labelDelay && (
        <p>
          <strong>Label delay:</strong> {labelDelay}
        </p>
      )}
      {channels && (
        <div className="chip-row" role="group" aria-label="Active monitoring channels">
          <span className="hint" style={{ alignSelf: "center" }}>
            Active channels:
          </span>
          {firing.length === 0 ? (
            <Badge tone="ok" plain>None firing</Badge>
          ) : (
            firing.map((c) => (
              <Badge key={c.name} tone={channelTone(c.status)}>
                {CHANNEL_LABEL[c.name] ?? c.name} · {c.status === "CRITICAL" ? "Critical" : "Warning"}
              </Badge>
            ))
          )}
        </div>
      )}
    </div>
  );
}
