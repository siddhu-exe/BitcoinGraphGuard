import { Suspense, useEffect, useRef } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { CardSkeleton } from "./ui";

const LINKS = [
  { to: "/", label: "Overview", title: "Overview" },
  { to: "/performance", label: "Performance", title: "Model Performance" },
  { to: "/retraining", label: "Retraining", title: "Retraining" },
  { to: "/monitoring", label: "Monitoring", title: "Drift Monitoring" },
  { to: "/try-it", label: "Try It", title: "Try It" },
];

export default function Layout() {
  const { pathname } = useLocation();
  const mainRef = useRef<HTMLElement>(null);
  const first = useRef(true);

  useEffect(() => {
    const link = LINKS.find((l) => l.to === pathname) ?? LINKS[0];
    document.title = `${link.title} · BitcoinGraphGuard`;
    // Move focus to the page on client-side navigation (not on first load) for keyboard / SR users.
    if (first.current) first.current = false;
    else mainRef.current?.focus({ preventScroll: true });
    window.scrollTo({ top: 0 });
  }, [pathname]);

  return (
    <div className="app">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="topbar">
        <div className="topbar__inner">
          <NavLink to="/" className="brand" aria-label="BitcoinGraphGuard home">
            <img className="brand__mark" src="/favicon.svg" alt="" width="28" height="28" />
            <span className="brand__name">BitcoinGraphGuard</span>
          </NavLink>
          <nav className="nav" aria-label="Primary">
            {LINKS.map((l) => (
              <NavLink key={l.to} to={l.to} end>
                {l.label}
              </NavLink>
            ))}
          </nav>
          <div className="topbar__meta">
            <span className="badge badge--info badge--plain">Elliptic++ · steps 35–49</span>
          </div>
        </div>
      </header>
      <main id="main" ref={mainRef} tabIndex={-1} className="page">
        <Suspense
          fallback={
            <div className="stack" aria-busy="true">
              <CardSkeleton h={60} />
              <CardSkeleton h={320} />
            </div>
          }
        >
          <Outlet />
        </Suspense>
      </main>
      <footer className="footer">
        <div className="footer__inner">
          <span>BitcoinGraphGuard · Temporal heterogeneous graph fraud detection</span>
          <span>Served model: frozen XGBoost · τ* = 0.435</span>
        </div>
      </footer>
    </div>
  );
}
