import type { ModelRow } from "../lib/derived";
import { num } from "../lib/format";

export default function ModelTable({ models }: { models: ModelRow[] }) {
  return (
    <div className="table-scroll">
      <table className="table">
        <thead>
          <tr>
            <th scope="col">Model</th>
            <th scope="col" className="num">
              Before
            </th>
            <th scope="col" className="num">
              After
            </th>
          </tr>
        </thead>
        <tbody>
          {models.map((m) => (
            <tr key={m.name}>
              <th scope="row" className="model-cell">
                <span className="model-name">{m.name}</span>
                <span className="model-kind">{m.kind}</span>
              </th>
              <td className="num cell-ink">{num(m.early, 3)}</td>
              <td className="num cell-drift">{num(m.late, 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
