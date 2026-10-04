import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { DRIFT_START } from "../lib/format";

const AXIS = { fontFamily: "JetBrains Mono, monospace", fontSize: 10, fill: "#5b6b82" } as const;

export interface LineSpec {
  key: string;
  name: string;
  color: string;
  dash?: string;
  width?: number;
}

interface TipProps {
  active?: boolean;
  label?: number | string;
  payload?: { name: string; value: number | null; color: string }[];
  formatter?: (v: number) => string;
  /** step -> illicit labels, for steps too sparse for a per-step metric to mean anything. */
  noise?: Record<number, number>;
}

function Tip({ active, label, payload, formatter, noise }: TipProps) {
  if (!active || !payload?.length) return null;
  const f = formatter ?? ((v: number) => v.toFixed(3));
  const sparse = noise?.[Number(label)];
  return (
    <div className="chart-tip">
      <div className="chart-tip__title">
        Step {label}
        {Number(label) >= DRIFT_START ? " · drift regime" : ""}
      </div>
      {sparse !== undefined && (
        <div className="chart-tip__warn">Too few positives ({sparse}), noisy</div>
      )}
      {payload.map((p) => (
        <div className="chart-tip__row" key={p.name}>
          <span style={{ color: "var(--ink-2)" }}>
            <span style={{ color: p.color }}>●</span> {p.name}
          </span>
          <b>{p.value === null || p.value === undefined ? "—" : f(p.value)}</b>
        </div>
      ))}
    </div>
  );
}

interface DotProps {
  cx?: number;
  cy?: number;
  payload?: { step?: number };
}

/** Group sorted step numbers into [first, last] runs of consecutive steps. */
function contiguousRuns(steps: number[]): [number, number][] {
  const runs: [number, number][] = [];
  for (const st of steps) {
    const run = runs[runs.length - 1];
    if (run && st === run[1] + 1) run[1] = st;
    else runs.push([st, st]);
  }
  return runs;
}

export function StepLines({
  data,
  lines,
  summary,
  height = "chart--lg",
  domain = [0, 1],
  showDriftBand = true,
  bandLabel = true,
  yTicks,
  yFormat,
  formatter,
  noise,
}: {
  data: object[];
  lines: LineSpec[];
  summary: string;
  height?: string;
  domain?: [number, number | "auto"];
  showDriftBand?: boolean;
  /** Print "Drift regime" inside the band (turn off on stacked secondary charts). */
  bandLabel?: boolean;
  yTicks?: number[];
  yFormat?: (v: number) => string;
  formatter?: (v: number) => string;
  noise?: Record<number, number>;
}) {
  const steps = (data as { step: number }[]).map((d) => d.step);
  const last = steps[steps.length - 1];
  const hasDrift = showDriftBand && last >= DRIFT_START && steps[0] <= last;
  const noiseRuns = contiguousRuns(steps.filter((st) => noise?.[st] !== undefined));
  return (
    <div className={`chart ${height}`} role="img" aria-label={summary}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
          <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" vertical={false} />
          {hasDrift && (
            <ReferenceArea
              x1={Math.max(DRIFT_START, steps[0])}
              x2={last}
              fill="#fff1f2"
              fillOpacity={0.9}
              stroke="#fda4af"
              strokeDasharray="4 3"
              ifOverflow="visible"
              label={
                bandLabel
                  ? { value: "Drift regime (43–49)", position: "insideTopRight", fill: "#be123c", fontSize: 11, fontFamily: "Space Grotesk, sans-serif", fontWeight: 600 }
                  : undefined
              }
            />
          )}
          {noiseRuns.map(([a, b]) => (
            <ReferenceArea
              key={`noise-${a}`}
              x1={a}
              x2={b}
              fill="#94a3b8"
              fillOpacity={0.28}
              stroke="#64748b"
              strokeDasharray="2 2"
              ifOverflow="visible"
            />
          ))}
          <XAxis dataKey="step" tick={AXIS} tickLine={false} axisLine={{ stroke: "#cbd5e1" }} interval="preserveStartEnd" minTickGap={8} />
          <YAxis domain={domain} ticks={yTicks} tick={AXIS} tickLine={false} axisLine={false} width={44} tickFormatter={yFormat ?? ((v: number) => v.toFixed(v % 1 ? 2 : 1))} />
          <Tooltip content={<Tip formatter={formatter} noise={noise} />} cursor={{ stroke: "#94a3b8", strokeDasharray: "3 3" }} isAnimationActive={false} />
          {lines.map((l) => (
            <Line
              key={l.key}
              dataKey={l.key}
              name={l.name}
              stroke={l.color}
              strokeWidth={l.width ?? 2.25}
              strokeDasharray={l.dash}
              dot={(p: DotProps) => {
                const sparse = noise?.[p.payload?.step ?? -1] !== undefined;
                return (
                  <circle
                    key={`${l.key}-${p.payload?.step}`}
                    cx={p.cx}
                    cy={p.cy}
                    r={sparse ? 3.5 : 2.5}
                    fill={sparse ? "#fff" : l.color}
                    stroke={sparse ? l.color : "#fff"}
                    strokeWidth={sparse ? 1.5 : 1}
                  />
                );
              }}
              activeDot={{ r: 5, stroke: "#fff", strokeWidth: 2 }}
              connectNulls
              isAnimationActive={false}
              type="linear"
            />
          ))}
          {hasDrift && <ReferenceLine x={DRIFT_START} stroke="#f43f5e" strokeDasharray="4 3" />}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function PrevalenceBars({
  data,
  summary,
}: {
  data: { step: number; prevalence: number | null }[];
  summary: string;
}) {
  return (
    <div className="chart chart--md" role="img" aria-label={summary}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
          <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="step" tick={AXIS} tickLine={false} axisLine={{ stroke: "#cbd5e1" }} interval={0} />
          <YAxis tick={AXIS} tickLine={false} axisLine={false} width={44} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} />
          <Tooltip
            content={<Tip formatter={(v) => `${(v * 100).toFixed(2)}%`} />}
            cursor={{ fill: "rgba(148,163,184,0.12)" }}
            isAnimationActive={false}
          />
          <Bar dataKey="prevalence" name="Illicit share" radius={[4, 4, 0, 0]} isAnimationActive={false}>
            {data.map((d) => (
              <Cell key={d.step} fill={d.step >= DRIFT_START ? "#fda4af" : "#5eead4"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export interface LagPoint {
  lag: number;
  static: number | null;
  static_ci: [number, number] | null;
  expanding: number | null;
  expanding_ci: [number, number] | null;
}

function LagTip({ active, payload }: { active?: boolean; payload?: { payload: LagPoint }[] }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  const row = (name: string, color: string, v: number | null, ci: [number, number] | null) => (
    <div className="chart-tip__row">
      <span style={{ color: "var(--ink-2)" }}>
        <span style={{ color }}>●</span> {name}
      </span>
      <b>
        {v === null ? "—" : v.toFixed(3)}
        {ci && <span style={{ fontWeight: 400, color: "var(--ink-3)" }}> [{ci[0].toFixed(2)}, {ci[1].toFixed(2)}]</span>}
      </b>
    </div>
  );
  return (
    <div className="chart-tip">
      <div className="chart-tip__title">Label delay L = {p.lag}</div>
      {row("Expanding", "#0d9488", p.expanding, p.expanding_ci)}
      {row("Static", "#64748b", p.static, p.static_ci)}
    </div>
  );
}

/** Pooled PR-AUC against label delay, with 95% cluster-bootstrap bands. */
export function LagCurve({ data, summary }: { data: LagPoint[]; summary: string }) {
  const lags = data.map((d) => d.lag);
  return (
    <div className="chart chart--lg" role="img" aria-label={summary}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: -12 }}>
          <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="lag"
            type="number"
            domain={[Math.min(...lags) - 0.4, Math.max(...lags) + 0.4]}
            ticks={lags}
            tickFormatter={(v: number) => `L=${v}`}
            tick={AXIS}
            tickLine={false}
            axisLine={{ stroke: "#cbd5e1" }}
            allowDataOverflow
          />
          <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tick={AXIS} tickLine={false} axisLine={false} width={44} tickFormatter={(v: number) => v.toFixed(v % 1 ? 2 : 1)} />
          <Tooltip content={<LagTip />} cursor={{ stroke: "#94a3b8", strokeDasharray: "3 3" }} isAnimationActive={false} />
          <Area dataKey="expanding_ci" stroke="none" fill="#0d9488" fillOpacity={0.18} isAnimationActive={false} activeDot={false} legendType="none" />
          <Area dataKey="static_ci" stroke="none" fill="#64748b" fillOpacity={0.22} isAnimationActive={false} activeDot={false} legendType="none" />
          <Line dataKey="static" name="Static" stroke="#64748b" strokeWidth={2.25} strokeDasharray="6 3" dot={{ r: 3, fill: "#64748b", stroke: "#fff", strokeWidth: 1 }} isAnimationActive={false} type="linear" />
          <Line dataKey="expanding" name="Expanding" stroke="#0d9488" strokeWidth={2.5} dot={{ r: 3.5, fill: "#0d9488", stroke: "#fff", strokeWidth: 1 }} isAnimationActive={false} type="linear" />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
