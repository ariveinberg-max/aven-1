# Working with AI coding agents (Codex, Claude Code)

AI agents are fast implementers and poor architects. This repository is set up so that you do the architecture and they do the implementing, one small, testable work package at a time.

## The loop

1. **Pick one WP** from [work-packages.md](../plan/work-packages.md) whose dependencies are done.
2. **Create a branch** named `wp/<id>-<slug>`.
3. **Prompt the agent** using the template below.
4. **Review the diff yourself**, using the checklist below. Run `make check` and `make smoke` locally.
5. **Merge** only when CI is green and you understand every changed line in `core/`, `data/catalog.py` and `evaluation/`.

## Prompt template

```
Implement work package WP-<id> from docs/plan/work-packages.md.

Read AGENTS.md first and follow it strictly. Relevant contracts:
- docs/architecture/data-model.md
- docs/architecture/evaluation-protocol.md (do not modify src/neurolayer/evaluation/)
- ADR-<nnnn> (list the relevant ones)

Scope: exactly the WP's scope. Put anything else under "Follow-ups" in the PR description.
Tests: add the tests listed in the WP; use synthetic data only (no downloads in default tests).
Done when: `make check` passes and the WP's acceptance criteria are met. Show the command output.
```

## Review checklist

- [ ] The diff only touches what the WP needs.
- [ ] There are new tests, and they would fail without the change. Temporarily revert a line and check.
- [ ] No test assertions were loosened to make things pass.
- [ ] Nothing in `evaluation/` or catalog `license` blocks changed (unless an ADR is attached).
- [ ] No new dependency without a reason, and heavy dependencies are extras.
- [ ] No hard-coded paths, no unseeded randomness, no silent `except:`.
- [ ] No data, weights or secrets (the hooks catch most of this; still look).
- [ ] Docstrings and types present; units (volts) and channel names (canonical) respected.

## Red flags that mean "stop and look closely"

- Accuracy suddenly jumps. Suspect leakage first (target subject in `fit`, test-window statistics, a label hidden in metadata).
- The agent "simplified" a validation or removed an error.
- The agent edited `.pre-commit-config.yaml`, CI, `pyproject.toml` tool settings or `AGENTS.md` without being asked.
- The agent cites a paper, dataset fact or license term you cannot find.
