# Agent Instructions

Whenever the user asks to add, change, or remove a feature — or fix a
bug — the agent MUST follow this gated workflow. Skipping a gate is a
process failure even if the code works.

## Workflow

### Phase 1 — PRD

1. Write or update `docs/PRDs/v<X.Y.Z>-<slug>.md` covering
   **Problem · Goal · Non-goals · Design · Acceptance**.
2. Show the PRD (or the relevant diff) to the user and **wait for an
   explicit accept gate** before any code change. Do not edit source,
   tests, configs, or run a release on the strength of an inferred
   "yes."
3. Once the PRD is accepted, **update `docs/roadmap.html`** to
   reflect the new or revised entry (slug, theme, status). The
   roadmap is the user-visible map of where the project is going;
   it must stay in lock-step with the PRD set. If the change
   renumbers or removes other releases, update those entries too.

### Phase 2 — Implementation

4. Only after the PRD is accepted AND `docs/roadmap.html` is in
   sync, develop the code so it matches the PRD.
5. If during implementation a constraint forces a **scope change** (a
   feature isn't supported by the underlying library, the design
   doesn't survive contact with reality, etc.), **stop coding**, go
   back to Phase 1, update the PRD AND the roadmap, get a fresh
   accept gate, then resume. Do not silently drop or add scope in
   the code.

### Phase 3 — User acceptance for release

6. Present the implementation to the user — running the user-based
   acceptance test from the PRD, or surfacing the diff plus a
   manual-test plan — and **wait for an explicit accept-or-reject
   gate**.
7. If the user **accepts**: write or update `CHANGELOG.md`, flip the
   roadmap entry's status to shipped, commit, tag `vX.Y.Z`
   (annotated), push branch + tag. This is the release.
8. If the user **rejects**: do NOT commit. Treat the feedback as
   driving a return to Phase 1 — update the PRD AND the roadmap with
   the new constraints, re-gate, re-implement.

## Principles

Design and implementation follow **SOLID** and **YAGNI** —
keep abstractions earned by use, keep features earned by the PRD.

## Notes on autonomous mode

If the user has set a `/goal` and walked away (e.g. an unattended
overnight implementation), the agent may proceed through Phase 2 and
declare Phase 3 ready, but **must not** ship the tag without an
explicit accept gate from a human — leave the diff staged with the
CHANGELOG drafted and surface the gate question in the next user
turn.
