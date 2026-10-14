import type { DelayRow } from "../lib/derived";
import { num } from "../lib/format";

function Bar({ row }: { row: DelayRow }) {
  const width = row.value !== null ? `${row.value * 100}%` : "0%";
  const hasCi = row.ciLo !== null && row.ciHi !== null;
  return (
    <div className="ldb__row">
      <span className={`ldb__label${row.isStatic ? " ldb__label--static" : ""}`}>{row.label}</span>
      <div
        className="ldb__track"
        role="img"
        aria-label={`${row.label}: PR-AUC ${num(row.value, 3)}${
          hasCi ? `, 95% interval ${num(row.ciLo, 2)} to ${num(row.ciHi, 2)}` : ""
        }`}
      >
        <div className={`ldb__bar${row.isStatic ? " ldb__bar--static" : ""}`} style={{ width }} />
        {hasCi && (
          <div
            className="ldb__ci"
            style={{ left: `${(row.ciLo as number) * 100}%`, width: `${((row.ciHi as number) - (row.ciLo as number)) * 100}%` }}
          />
        )}
      </div>
      <span className={`ldb__value${row.isStatic ? " ldb__value--drift" : ""}`}>{num(row.value, 3)}</span>
    </div>
  );
}

export default function LabelDelayBars({ rows, staticRow }: { rows: DelayRow[]; staticRow: DelayRow }) {
  return (
    <div className="ldb">
      {rows.map((r) => (
        <Bar key={r.label} row={r} />
      ))}
      <Bar row={staticRow} />
    </div>
  );
}
