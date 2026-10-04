export const API_URL: string = ((import.meta.env.VITE_API_URL as string | undefined) ?? "").replace(/\/+$/, "");

/** Render free tier cold starts take up to ~50s; give it headroom before giving up. */
export const API_TIMEOUT_MS = 65_000;
/** After this long without a response we tell the user the server is probably waking up. */
export const WAKING_AFTER_MS = 2_500;

export type ApiErrorKind = "config" | "timeout" | "network" | "http";

export class ApiError extends Error {
  kind: ApiErrorKind;
  status?: number;
  constructor(kind: ApiErrorKind, message: string, status?: number) {
    super(message);
    this.kind = kind;
    this.status = status;
  }
}

async function readError(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { message?: string; detail?: unknown; details?: string[] };
    if (body.details?.length) return `${body.message ?? "Validation failed"} ${body.details.slice(0, 3).join("; ")}`;
    if (typeof body.detail === "string") return body.detail;
    if (body.message) return body.message;
  } catch {
    /* fall through */
  }
  return `The API answered with HTTP ${res.status}.`;
}

export async function apiRequest<T>(
  path: string,
  init: RequestInit = {},
  signal?: AbortSignal,
): Promise<{ data: T; ms: number }> {
  if (!API_URL) {
    throw new ApiError("config", "VITE_API_URL is not set, so the live API cannot be reached.");
  }
  const ctrl = new AbortController();
  let timedOut = false;
  const timer = window.setTimeout(() => {
    timedOut = true;
    ctrl.abort();
  }, API_TIMEOUT_MS);
  const onAbort = () => ctrl.abort();
  signal?.addEventListener("abort", onAbort);

  const started = performance.now();
  try {
    const res = await fetch(`${API_URL}${path}`, { ...init, signal: ctrl.signal });
    if (!res.ok) throw new ApiError("http", await readError(res), res.status);
    const data = (await res.json()) as T;
    return { data, ms: performance.now() - started };
  } catch (err) {
    if (err instanceof ApiError) throw err;
    if (timedOut) {
      throw new ApiError("timeout", `No response after ${API_TIMEOUT_MS / 1000}s. The server may still be starting.`);
    }
    if (signal?.aborted) throw err;
    throw new ApiError("network", "Could not reach the API. It may be offline, or blocked by CORS or the network.");
  } finally {
    window.clearTimeout(timer);
    signal?.removeEventListener("abort", onAbort);
  }
}
