import type { ReactNode } from "react";

export function BrandMark() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
      <circle cx="4" cy="10" r="2.2" />
      <circle cx="16" cy="4" r="2.2" />
      <circle cx="16" cy="16" r="2.2" />
      <path d="M6 9l8-4M6 11l8 4" />
    </svg>
  );
}

export function InfoIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
      <circle cx="10" cy="10" r="8" />
      <path d="M10 9v5M10 6v.01" />
    </svg>
  );
}

export function AsyncState({
  loading,
  error,
  label,
}: {
  loading: boolean;
  error?: Error | null;
  label: string;
}) {
  if (loading) return <p className="page-fallback">Loading {label}…</p>;
  if (error) {
    return (
      <div className="error-box" role="alert">
        <span className="error-box__title">Could not load {label}</span>
        <span>{error.message}</span>
      </div>
    );
  }
  return null;
}

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <article className={`card ${className}`.trim()}>{children}</article>;
}
