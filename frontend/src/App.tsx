import { NavLink, Route, Routes } from "react-router-dom";
import { BrandMark } from "./components/BrandMark";
import { CaseListPage } from "./pages/CaseListPage";
import { MetricsPage } from "./pages/MetricsPage";
import { useHealth } from "./hooks/useCases";

function HealthPill() {
  const { data } = useHealth();
  if (!data) {
    return <span className="health-pill health-pill--unknown">API status unknown</span>;
  }
  const tone = data.status === "ok" && data.database ? "ok" : "degraded";
  return (
    <span className={`health-pill health-pill--${tone}`}>
      API {data.status} · DB {data.database ? "connected" : "unavailable"}
    </span>
  );
}

export default function App() {
  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar__brand">
          <BrandMark size={32} />
          <div>
            <h1>RevenueGuard</h1>
            <p className="topbar__subtitle">Revenue recovery operations console</p>
          </div>
        </div>
        <nav className="topbar__nav" aria-label="Primary">
          <NavLink to="/" end>
            Cases
          </NavLink>
          <NavLink to="/metrics">Metrics</NavLink>
        </nav>
        <HealthPill />
      </header>
      <main className="app-main">
        <Routes>
          <Route path="/" element={<CaseListPage />} />
          <Route path="/cases/:transactionId" element={<CaseListPage />} />
          <Route path="/metrics" element={<MetricsPage />} />
          <Route path="*" element={<div className="status status--notfound">Page not found.</div>} />
        </Routes>
      </main>
      <footer className="app-footer">
        <p className="text-muted">
          Predictions are model estimates. Recommendations are deterministic investigation outputs.
          Policy decisions are authoritative. No financial actions are executed by this application.
        </p>
      </footer>
    </div>
  );
}

