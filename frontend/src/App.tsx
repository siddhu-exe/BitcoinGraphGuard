import { lazy, Suspense } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import AppShell from "./components/AppShell";

const Overview = lazy(() => import("./pages/Overview"));
const Retraining = lazy(() => import("./pages/Retraining"));
const Monitoring = lazy(() => import("./pages/Monitoring"));
const TryIt = lazy(() => import("./pages/TryIt"));

export default function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={<div className="page-fallback">Loading…</div>}>
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<Overview />} />
            <Route path="retraining" element={<Retraining />} />
            <Route path="monitoring" element={<Monitoring />} />
            <Route path="try-it" element={<TryIt />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}
