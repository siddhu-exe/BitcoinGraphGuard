import type { TriageRow } from "../lib/derived";
import { NA, num } from "../lib/format";

export default function TriageTable({ rows }: { rows: TriageRow[] }) {
  return (
    <div className="table-scroll">
      <table className="table table--triage">
        <thead>
          <tr>
            <th scope="col">K per step</th>
            <th scope="col" className="num">
              Retrained precision
            </th>
            <th scope="col" className="num">
              Retrained recall
            </th>
            <th scope="col" className="num">
              Static recall
            </th>
            <th scope="col" className="num">
              Max recall
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.k}>
              <th
                scope="row"
                style={{ textAlign: "left", fontWeight: 500, fontSize: 15, color: "var(--ink)", padding: "12px 12px 12px 0", borderTop: "1px solid var(--border-soft)" }}
              >
                {r.k}
              </th>
              <td className="num cell-ink">{num(r.precision, 3)}</td>
              <td className="num cell-accent">{r.recall === null ? NA : num(r.recall, 3)}</td>
              <td className="num cell-drift">{num(r.staticRecall, 3)}</td>
              <td className="num cell-muted">{num(r.maxRecall, 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
