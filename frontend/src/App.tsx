import { lazy } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";

const Overview = lazy(() => import("./pages/Overview"));
const Performance = lazy(() => import("./pages/Performance"));
const Retraining = lazy(() => import("./pages/Retraining"));
const Monitoring = lazy(() => import("./pages/Monitoring"));
const TryIt = lazy(() => import("./pages/TryIt"));

function NotFound() {
  return (
    <div className="state">
      <h1>Page not found</h1>
      <a className="btn btn--sm" href="/">
        Back to Overview
      </a>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Overview />} />
          <Route path="performance" element={<Performance />} />
          <Route path="retraining" element={<Retraining />} />
          <Route path="monitoring" element={<Monitoring />} />
          <Route path="try-it" element={<TryIt />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
