# Contributing

1. **Start from a work package** ([docs/plan/work-packages.md](docs/plan/work-packages.md)) or an issue created from a template.
2. **Branch:** `wp/<id>-<slug>`, `fix/<slug>`, `exp/<slug>` or `docs/<slug>`. Never commit directly to `main`.
3. **Commits:** [Conventional Commits](https://www.conventionalcommits.org/). Examples:
   - `feat(data): add MOABB adapter`
   - `fix(eval): …` (needs an ADR)
   - `docs: …`
   - `test: …`
   - `chore: …`
4. **Before pushing:** `make check` (and `make web-check` if you touched `apps/web`).
5. **PR:** fill in the template, keep it small (aim for < 400 changed lines excluding lockfiles), and link the WP or issue.
6. **Architecture changes:** add an ADR (`docs/adr/0000-template.md`) in the same PR.
7. **Experiments:** add a card (`docs/experiments/TEMPLATE.md`) and a config. Only official runs go into the results ledger.

AI agents follow [AGENTS.md](AGENTS.md). Human reviewers use the checklist in [working with AI agents](docs/guides/working-with-ai-agents.md).
