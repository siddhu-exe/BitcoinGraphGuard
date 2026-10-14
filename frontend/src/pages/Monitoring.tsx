import { loadMonitoring } from "../lib/data";
import { channelCards } from "../lib/derived";
import { span, trim } from "../lib/format";
import { useStatic } from "../lib/useStatic";
import ChannelCards from "../components/ChannelCards";
import DecisionTable from "../components/DecisionTable";
import { AsyncState, Card } from "../components/ui";

export default function Monitoring() {
  const mon = useStatic(loadMonitoring);
  if (mon.loading) return <AsyncState loading label="monitoring backtest" />;
  if (mon.error || !mon.data) return <AsyncState loading={false} error={mon.error} label="monitoring backtest" />;

  const m = mon.data;
  const channels = m.latest_decision.channels;
  const delay = m.latest_decision.label_delay_steps;
  const f1Floor = channels.find((c) => c.name === "performance")?.threshold ?? null;
  const psiThreshold = channels.find((c) => c.name === "score_shift")?.threshold ?? null;

  return (
    <div className="page">
      <section className="head-block">
        <h1 className="h1">How the system knows it is wrong</h1>
        <p className="lede">
          The monitor watches four signals and recommends a retrain when they shift. Replayed on the real data, it
          stayed quiet while the model worked (steps {span(35, 42)}) and raised the alarm at step 43, the step the model
          broke. It only uses information that would have existed at the time.
        </p>
      </section>

      <ChannelCards cards={channelCards(m)} />

      <Card className="card--snug">
        <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", gap: "8px 24px" }}>
          <div className="card-head">
            <h2 className="h2">Decisions by step</h2>
            <p className="subtle">
              Label delay {delay} step · F1 floor {trim(f1Floor, 2)} · score PSI critical at {trim(psiThreshold, 2)}
            </p>
          </div>
          <span className="badge badge--neutral" style={{ alignSelf: "flex-start" }}>
            Validated on one drift event
          </span>
        </div>
        <DecisionTable mon={m} highlightStep={43} />
        <p className="footnote">
          Score PSI uses labelled transactions only. Steps 45–46 have under 10 positives, so the performance window at
          46–47 falls back to steps {span(43, 44)}.
        </p>
      </Card>
    </div>
  );
}
