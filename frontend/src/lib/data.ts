import type { MonitoringData, PerformanceData, SamplesData, WalkforwardData } from "./types";

const cache = new Map<string, Promise<unknown>>();

/** Fetch a bundled static JSON file once and memoise the promise. */
function loadStatic<T>(name: string): Promise<T> {
  let hit = cache.get(name) as Promise<T> | undefined;
  if (!hit) {
    hit = fetch(`/data/${name}`).then((res) => {
      if (!res.ok) throw new Error(`Could not load /data/${name} (HTTP ${res.status})`);
      if (!(res.headers.get("content-type") ?? "").includes("json")) {
        throw new Error(`/data/${name} is missing. Run: python scripts/export_dashboard_data.py`);
      }
      return res.json() as Promise<T>;
    });
    hit.catch(() => cache.delete(name));
    cache.set(name, hit);
  }
  return hit;
}

export const loadPerformance = () => loadStatic<PerformanceData>("performance.json");
export const loadMonitoring = () => loadStatic<MonitoringData>("monitoring.json");
export const loadSamples = () => loadStatic<SamplesData>("samples.json");
export const loadWalkforward = () => loadStatic<WalkforwardData>("walkforward.json");
