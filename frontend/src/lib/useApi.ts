import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, WAKING_AFTER_MS, apiRequest } from "./api";

export type Phase = "idle" | "loading" | "waking" | "success" | "error";

export interface ApiState<T> {
  phase: Phase;
  data?: T;
  error?: ApiError;
  /** Round-trip time of the last successful call, in ms. */
  ms?: number;
  /** Seconds since the in-flight request started. */
  elapsed: number;
}

function useMachine<T>() {
  const [state, setState] = useState<ApiState<T>>({ phase: "idle", elapsed: 0 });
  const abortRef = useRef<AbortController | null>(null);
  const startedRef = useRef(0);

  const run = useCallback(async (call: (signal: AbortSignal) => Promise<{ data: T; ms: number }>) => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    startedRef.current = Date.now();
    setState((s) => ({ phase: "loading", data: s.data, elapsed: 0 }));
    try {
      const { data, ms } = await call(ctrl.signal);
      if (ctrl.signal.aborted) return;
      setState({ phase: "success", data, ms, elapsed: 0 });
    } catch (err) {
      if (ctrl.signal.aborted) return;
      const error = err instanceof ApiError ? err : new ApiError("network", "Unexpected error.");
      setState((s) => ({ phase: "error", error, data: s.data, elapsed: 0 }));
    }
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setState({ phase: "idle", elapsed: 0 });
  }, []);

  const inFlight = state.phase === "loading" || state.phase === "waking";
  useEffect(() => {
    if (!inFlight) return;
    const id = window.setInterval(() => {
      const elapsedMs = Date.now() - startedRef.current;
      setState((s) =>
        s.phase === "loading" || s.phase === "waking"
          ? { ...s, phase: elapsedMs >= WAKING_AFTER_MS ? "waking" : "loading", elapsed: Math.floor(elapsedMs / 1000) }
          : s,
      );
    }, 500);
    return () => window.clearInterval(id);
  }, [inFlight]);

  useEffect(() => () => abortRef.current?.abort(), []);

  return { state, run, reset };
}

/** GET a JSON endpoint on mount; exposes retry(). */
export function useApi<T>(path: string) {
  const { state, run } = useMachine<T>();
  const load = useCallback(() => run((signal) => apiRequest<T>(path, {}, signal)), [path, run]);
  useEffect(() => {
    void load();
  }, [load]);
  return { ...state, retry: load };
}

/** POST on demand (Try It). */
export function useApiAction<T>() {
  const { state, run, reset } = useMachine<T>();
  const post = useCallback(
    (path: string, body: unknown) =>
      run((signal) =>
        apiRequest<T>(
          path,
          { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) },
          signal,
        ),
      ),
    [run],
  );
  return { ...state, post, reset };
}
