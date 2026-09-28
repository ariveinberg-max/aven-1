# ADR-0001: Record architecture decisions

- **Status:** Accepted
- **Date:** 2026-09-26

## Context
Much of the code will be written with AI coding agents (Codex, Claude Code). Agents are fast but have no memory of *why* things are the way they are. Without written decisions, they will "helpfully" undo them, and the codebase drifts into a pile of generated code.

## Decision
Record significant decisions as numbered ADRs in `docs/adr/`. `AGENTS.md` instructs agents to read the relevant ADRs and never to contradict an accepted ADR without proposing a new one.

## Consequences
There is a small writing overhead per decision. In exchange, decisions become reviewable and agents stay consistent.

## Alternatives considered
- Wiki pages: they drift from the code and agents cannot see them.
- Code comments only: too scattered to capture cross-cutting decisions.
