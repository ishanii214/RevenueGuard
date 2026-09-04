"""RevenueGuard investigation agent package.

Phase 3: a fully deterministic LangGraph investigation workflow. It produces
investigative recommendations only — it never executes or authorizes a
financial action. The later deterministic policy/guardrail layer controls
whether any recommended action is permitted.

LLM-backed narration/reasoning is deliberately deferred to a later phase;
schemas, tools and graph state are designed so an LLM node can be added
without redesign (see prompts.py).
"""
