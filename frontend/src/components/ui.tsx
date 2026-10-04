import type { ReactNode } from "react";
import type { ApiError } from "../lib/api";

type IconName = "warning" | "check" | "info" | "refresh" | "bolt" | "wifi-off" | "arrow-right";

const PATHS: Record<IconName, ReactNode> = {
  warning: (
    <>
      <path d="M12 3.5 2.5 20h19L12 3.5Z" />
      <path d="M12 10v4.5M12 17.5h.01" />
    </>
  ),
  check: <path d="m5 12.5 4.5 4.5L19 7.5" />,
  info: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v5M12 7.5h.01" />
    </>
  ),
  refresh: (
    <>
      <path d="M20 11a8 8 0 1 0-2.3 5.7" />
      <path d="M20 4v7h-7" />
    </>
  ),
  bolt: <path d="M13 3 5 13.5h6L10 21l8-10.5h-6L13 3Z" />,
  "wifi-off": (
    <>
      <path d="M3 3l18 18" />
      <path d="M5 12.5a10 10 0 0 1 4-2.3M19 12.5a10 10 0 0 0-3-1.9M8.5 16a5 5 0 0 1 7 0M12 19.5h.01" />
    </>
  ),
  "arrow-right": <path d="M5 12h14m-5-5 5 5-5 5" />,
};

export function Icon({ name, size = 16, className }: { name: IconName; size?: number; className?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      className={className}
    >
      {PATHS[name]}
    </svg>
  );
}

export type Tone = "ok" | "warn" | "crit" | "info" | "neutral";

export function Badge({
  tone = "neutral",
  children,
  large,
  plain,
}: {
  tone?: Tone;
  children: ReactNode;
  large?: boolean;
  plain?: boolean;
}) {
  const cls = ["badge", tone !== "neutral" && `badge--${tone}`, large && "badge--lg", plain && "badge--plain"]
    .filter(Boolean)
    .join(" ");
  return <span className={cls}>{children}</span>;
}

export function Card({
  title,
  sub,
  aside,
  children,
  foot,
  tone,
  as: Tag = "section",
  className = "",
}: {
  title?: ReactNode;
  sub?: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
  foot?: ReactNode;
  tone?: "crit";
  as?: "section" | "div" | "article";
  className?: string;
}) {
  return (
    <Tag className={`card ${tone ? `card--${tone}` : ""} ${className}`}>
      {(title || aside) && (
        <header className="card__head">
          <div>
            {title && <h2 className="card__title">{title}</h2>}
            {sub && <p className="card__sub">{sub}</p>}
          </div>
          {aside}
        </header>
      )}
      {children}
      {foot && <footer className="card__foot">{foot}</footer>}
    </Tag>
  );
}

export function Kpi({
  label,
  badge,
  value,
  footLeft,
  footRight,
  tone,
  crit,
}: {
  label: ReactNode;
  badge?: ReactNode;
  value: ReactNode;
  footLeft?: ReactNode;
  footRight?: ReactNode;
  tone?: "ok" | "warn" | "crit" | "accent";
  crit?: boolean;
}) {
  return (
    <article className={`card kpi ${crit ? "card--crit" : ""}`}>
      <div className="kpi__label">
        <span>{label}</span>
        {badge}
      </div>
      <div className={`kpi__value ${tone ? `tone-${tone}` : ""}`}>{value}</div>
      <div className="kpi__foot">
        <span>{footLeft}</span>
        <span>{footRight}</span>
      </div>
    </article>
  );
}

export function Skeleton({ h = 16, w = "100%" }: { h?: number; w?: number | string }) {
  return <div className="skeleton" style={{ height: h, width: w }} aria-hidden="true" />;
}

/** Reserves a card's space while static data loads, so nothing shifts. */
export function CardSkeleton({ h = 220, title }: { h?: number; title?: string }) {
  return (
    <section className="card" aria-busy="true" aria-label={title ? `Loading ${title}` : "Loading"}>
      <Skeleton h={18} w="40%" />
      <div style={{ height: 16 }} />
      <Skeleton h={h} />
    </section>
  );
}

export function ErrorState({
  title = "Something went wrong",
  error,
  onRetry,
}: {
  title?: string;
  error?: ApiError | Error;
  onRetry?: () => void;
}) {
  return (
    <div className="state" role="alert">
      <Icon name="warning" size={24} className="tone-crit" />
      <div>
        <strong style={{ color: "var(--ink)" }}>{title}</strong>
        {error && <p className="hint" style={{ marginTop: 4, maxWidth: 52 + "ch" }}>{error.message}</p>}
      </div>
      {onRetry && (
        <button type="button" className="btn btn--sm" onClick={onRetry}>
          <Icon name="refresh" size={14} /> Retry
        </button>
      )}
    </div>
  );
}

export function Legend({
  items,
  label,
}: {
  items: { name: string; color: string; dash?: string; swatch?: boolean }[];
  label: string;
}) {
  return (
    <ul className="legend" aria-label={label}>
      {items.map((it) => (
        <li key={it.name} style={{ color: "var(--ink-2)" }}>
          <i
            className={it.swatch ? "sw" : ""}
            style={{
              color: it.color,
              borderTopStyle: it.dash ? "dashed" : "solid",
              background: it.swatch ? it.color : undefined,
            }}
          />
          {it.name}
        </li>
      ))}
    </ul>
  );
}
