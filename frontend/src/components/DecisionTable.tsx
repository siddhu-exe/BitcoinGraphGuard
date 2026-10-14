import type { MonitoringData } from "../lib/types";
import {
  actionBadgeClass,
  actionLabel,
  channelStateClass,
  criticalChannelsText,
  labelsRead,
  prevalenceStatus,
} from "../lib/derived";
import { num } from "../lib/format";

export default function DecisionTable({ mon, highlightStep = 43 }: { mon: MonitoringData; highlightStep?: number }) {
  return (
    <div className="table-scroll">
      <table className="table table--decisions">
        <thead>
          <tr>
            <th scope="col">Step</th>
            <th scope="col">Labels read</th>
            <th scope="col" className="num">
              Worst F1
            </th>
            <th scope="col" className="num">
              Score PSI
            </th>
            <th scope="col">Prevalence</th>
            <th scope="col">Critical channels</th>
            <th scope="col">Decision</th>
          </tr>
        </thead>
        <tbody>
          {mon.lag_safe.map((r) => {
            const psiCritical = r.channels.score_shift?.status === "CRITICAL";
            return (
              <tr key={r.step} className={r.step === highlightStep ? "row--highlight" : undefined}>
                <td className="mono cell-ink" style={{ fontWeight: 500 }}>
                  {r.step}
                </td>
                <td className="mono cell-muted">{labelsRead(r.perf_steps_read)}</td>
                <td className="num cell-ink">{num(r.worst_f1, 3)}</td>
                <td className={`num ${psiCritical ? channelStateClass("CRITICAL") : "cell-ink"}`}>
                  {num(r.score_psi, 4)}
                </td>
                <td className="cell-body">{prevalenceStatus(r)}</td>
                <td className="cell-body">{criticalChannelsText(r)}</td>
                <td>
                  <span className={actionBadgeClass(r.action)}>{actionLabel(r.action)}</span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
