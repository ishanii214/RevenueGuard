"""Prompt templates for the investigation agent.

Reserved for LLM-backed evidence narration and reasoning in a later phase.
Phase 3 is fully deterministic and uses no LLM. When an LLM node is added it
must slot into the existing graph without changing schemas, tools, or graph
state, and its output must remain non-authoritative: recommendations stay
investigative, never financially authorized.
"""
