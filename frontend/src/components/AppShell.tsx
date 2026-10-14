import { NavLink, Outlet, useLocation } from "react-router-dom";
import { BrandMark } from "./ui";

const LINKS = [
  { to: "/", label: "Overview", end: true },
  { to: "/retraining", label: "Retraining", end: false },
  { to: "/monitoring", label: "Monitoring", end: false },
  { to: "/try-it", label: "Try it", end: false },
];

const GITHUB_URL = "https://github.com/siddhu-exe";

export default function AppShell() {
  const { pathname } = useLocation();
  const isLanding = pathname === "/";

  return (
    <div className="app">
      <header className="site-header">
        <nav className="nav" aria-label="Primary">
          <NavLink to="/" className="brand" aria-label="BitcoinGraphGuard home">
            <BrandMark />
            BitcoinGraphGuard
          </NavLink>
          <div className="nav-links">
            {LINKS.map((l) => (
              <NavLink key={l.to} to={l.to} end={l.end} className="nav-link">
                {l.label}
              </NavLink>
            ))}
          </div>
          {isLanding ? (
            <a className="nav-github" href={GITHUB_URL} target="_blank" rel="noreferrer">
              GitHub →
            </a>
          ) : (
            <p className="nav-meta">Elliptic++ · test steps 35–49</p>
          )}
        </nav>
      </header>
      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}
