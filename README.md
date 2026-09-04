# RevenueGuard

**AI-powered revenue recovery and failed-payment investigation platform.**

## About RevenueGuard

RevenueGuard is a planned platform that helps businesses recover revenue lost to failed payments. For each failed payment, it will predict how likely the payment is to be recovered, investigate the failure with an AI agent, and apply financial policy guardrails to decide whether the payment should be retried, sent for human review, or ignored. Recovery attempts are simulated, and their outcomes feed business metrics so the impact of every decision can be measured.

## The Problem It Solves

Failed payments represent a large, largely unmanaged revenue leak:

- **Recoverable revenue is lost silently** — legitimate customers' payments fail for temporary reasons (insufficient funds, expired cards, transient gateway issues) and are never retried effectively.
- **Retries are undifferentiated** — naive retry strategies treat every failed payment the same, wasting attempt budgets on unrecoverable payments while missing recoverable ones.
- **Investigation is manual** — understanding why a payment failed requires digging through payment and customer records by hand.
- **Outcomes are unmeasured** — teams rarely know how much revenue their recovery efforts actually recover.

RevenueGuard is being designed to turn failed-payment recovery into a predicted, investigated, policy-guarded, and measured process.

## Planned Architecture

RevenueGuard will process each failed payment through the following pipeline:

```mermaid
flowchart TD
    A["Failed Payment"] --> B["XGBoost Recovery Prediction"]
    B --> C["LangGraph Investigation Agent"]
    C --> D["Payment / Customer Investigation Tools"]
    D --> E["Financial Policy / Guardrails"]
    E --> F["RETRY / REVIEW / IGNORE"]
    F --> G["Simulated Recovery"]
    G --> H["Business Metrics"]
```

- **XGBoost Recovery Prediction** — a trained model estimates the probability that a failed payment can be successfully recovered.
- **LangGraph Investigation Agent** — an agent orchestrates the investigation of each failed payment, deciding which checks to run and in what order.
- **Payment / Customer Investigation Tools** — focused tools the agent calls to inspect payment and customer context.
- **Financial Policy / Guardrails** — hard rules and limits that constrain what the agent is allowed to decide.
- **RETRY / REVIEW / IGNORE** — the three possible outcomes: retry the payment automatically, escalate for human review, or take no action.
- **Simulated Recovery** — recovery attempts are simulated so decision quality can be evaluated safely before touching real money flows.
- **Business Metrics** — recovered amount, recovery rate, and related KPIs quantify the system's impact.

## Planned Technology Stack

The project is Python-oriented. The current plan:

- **Language:** Python 3.x
- **Recovery prediction:** XGBoost
- **Agent orchestration:** LangGraph
- **Agent tracing and evaluation:** LangSmith (later)
- **Application persistence:** PostgreSQL (later)
- **Backend API:** FastAPI (later)
- **Analyst dashboard:** React + TypeScript (later)
- **Tool interface:** MCP (later — exposes the existing investigation tools through a standardized tool interface)
- **Version control and collaboration:** Git / GitHub

Dependencies will be introduced only as each phase of the project actually requires them.

## Project Structure

```text
RevenueGuard/
├── README.md
├── .gitignore
├── .env.example
├── requirements.txt
├── data/
├── scripts/
├── agent/
│   ├── __init__.py
│   ├── graph.py
│   ├── state.py
│   ├── tools.py
│   └── prompts.py
└── tests/
```

## Project Status

RevenueGuard is being developed **incrementally**. The repository currently contains project scaffolding only. The following components are **planned and not yet implemented**:

- [ ] XGBoost recovery prediction model
- [ ] LangGraph investigation agent
- [ ] Payment / customer investigation tools
- [ ] Financial policy / guardrails
- [ ] Simulated recovery and business metrics
- [ ] LangSmith tracing and evaluation
- [ ] PostgreSQL persistence
- [ ] FastAPI backend
- [ ] React + TypeScript analyst dashboard
- [ ] MCP tool interface
