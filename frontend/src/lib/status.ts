/**
 * Overview status strip: read /monitoring/status from the live API, and fall
 * back to the bundled monitoring.json when the API is unreachable or cold.
 */
import { loadMonitoring } from "./data";
import { CHANNEL_LABEL } from "./derived";
import type {MonitoringStatusResponse, TriggerAction } from "./types";
import { useApi } from "./useApi";
import { useStatic } from "./useStatic";

/** Channels that can drive a retrain decision, in display order. */
const TRIGGER_KEYS = ["performance", "prevalence", "score_shift", "triad"];

function channelsText(statuses: Record<string, string>): string {
  const hits = TRIGGER_KEYS.filter((k) => statuses[k] === "CRITICAL" || statuses[k] === "ALERT");
  const names = hits.map((k) => (CHANNEL_LABEL[k] ?? k).toLowerCase());
  if (names.length === 0) return "no trigger channels";
  if (names.length === 1) return `${names[0]} channel`;
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]} channels`;
}

export interface StatusView {
  step: number | null;
  action: TriggerAction;
  severity: string;
  channelsText: string;
  labelDelay: number | null;
  source: "live" | "cached" | "loading";
}

export function useMonitoringStatus(): StatusView {
  const api = useApi<MonitoringStatusResponse>("/monitoring/status");
  const cached = useStatic(loadMonitoring);

  if (api.phase === "success" && api.data) {
    const d = api.data;
    return {
      step: d.latest_monitored_step ?? null,
      action: d.trigger_action,
      severity: d.severity,
      channelsText: channelsText(d.component_statuses ?? {}),
      labelDelay: d.label_delay_steps ?? null,
      source: "live",
    };
  }

  if (cached.data) {
    const d = cached.data.latest_decision;
    return {
      step: d.time_step ?? null,
      action: d.action,
      severity: d.severity,
      channelsText: channelsText(d.component_statuses ?? {}),
      labelDelay: d.label_delay_steps ?? null,
      source: "cached",
    };
  }

  return { step: null, action: "NO_ACTION", severity: "", channelsText: "", labelDelay: null, source: "loading" };
}
