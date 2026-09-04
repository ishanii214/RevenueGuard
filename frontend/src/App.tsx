import { NavLink, Route, Routes } from "react-router-dom";
import { CaseDetailPage } from "./pages/CaseDetailPage";
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
      <header className="app-header">
        <div>
          <h1>RevenueGuard</h1>
          <p className="app-header__subtitle">Revenue recovery operations — prediction, investigation, policy</p>
        </div>
        <HealthPill />
      </header>
      <nav className="app-nav" aria-label="Primary">
        <NavLink to="/" end>
          Cases
        </NavLink>
        <NavLink to="/metrics">Metrics</NavLink>
      </nav>
      <main className="app-main">
        <Routes>
          <Route path="/" element={<CaseListPage />} />
          <Route path="/cases/:transactionId" element={<CaseDetailPage />} />
          <Route path="/metrics" element={<MetricsPage />} />
          <Route path="*" element={<div className="status status--notfound">Page not found.</div>} />
        </Routes>
      </main>
      <footer className="app-footer">
        <p className="text-muted">
          Predictions are model estimates. Policy decisions are authoritative. No financial
          actions are executed by this application.
        </p>
      </footer>
    </div>
  );
}
