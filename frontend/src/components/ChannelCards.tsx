import type { ChannelCard } from "../lib/derived";

/** Display tag for each channel: the raw kind, with the two special cases the monitor uses. */
const kindLabel = (c: ChannelCard): string =>
  c.key === "feature_shift" ? "warning only" : c.kind === "label_dependent" ? "lag-safe" : c.kind.replace(/_/g, "-");

export default function ChannelCards({ cards }: { cards: ChannelCard[] }) {
  return (
    <section className="channels" aria-label="Monitoring channels">
      {cards.map((c) => (
        <div className="channel" key={c.key}>
          <div className="channel__top">
            <span className="channel__name">{c.name}</span>
            <span className="channel__kind">{kindLabel(c)}</span>
          </div>
          <span className={`channel__first${c.firstCritical === null ? " channel__first--muted" : " channel__first--drift"}`}>
            {c.firstCritical === null ? "never critical" : `step ${c.firstCritical}`}
          </span>
          <span className="channel__note">{c.note}</span>
        </div>
      ))}
    </section>
  );
}
