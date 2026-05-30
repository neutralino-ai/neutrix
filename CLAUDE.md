# Agent Instructions

Whenever the user asks to add, change, or remove a feature — or fix a
bug — the agent MUST follow this gated workflow. Skipping a gate is a
process failure even if the code works.

## Workflow

### Phase 1 — Splits → PRD

1. **Survey Claude Code first** for the analogous behavior. CC's
   TypeScript source lives at `/datafs/users/dingxf/anthropic/cc2/src`.
   Use the `learn-from-claude` skill or dispatch an `Explore`
   subagent with concrete keywords. Spend 10–20 minutes; longer
   means the scope is probably too broad and worth narrowing.
2. **Enumerate every design split point** — each place where there
   is more than one defensible choice (signal/semantics, history
   shape, key bindings, ordering, error surfacing, …). A list with
   zero split points is almost certainly under-surveyed; push back
   on yourself.
3. **Write the splits to
   `docs/splits/v<X.Y.Z>-<slug>.html`**. One HTML file per upcoming
   release. The user opens it in a browser and reads each split
   point with: what CC does (file:line citation), the alternatives,
   and the agent's recommendation. The user marks each split with
   a decision label (Follow CC / Alternative / User-directed
   different / No CC analog). Format/styling matches
   `docs/roadmap.html`. HTML — not Markdown — because the user
   reads it in a browser tab while deciding.
4. **Discuss with the user, turn by turn**, until every split has a
   decision the user is comfortable with. The HTML is the durable
   reference; the chat is where the back-and-forth happens. The
   user may add splits the agent missed.
5. **Only after the splits are settled**, write or update
   `docs/PRDs/v<X.Y.Z>-<slug>.md` covering
   **Problem · Goal · Non-goals · Claude Code reference · Design ·
   Acceptance**. The **Claude Code reference** section embeds the
   split-point decisions reached in step 4, one sub-block per
   split point. Link to the splits HTML for the full survey.
6. Show the PRD to the user and **wait for an explicit accept
   gate**. At this stage the user is confirming the writeup of
   decisions already taken; the gate is fast.
7. Once the PRD is accepted, **update `docs/roadmap.html`** to
   reflect the new or revised entry (slug, theme, status). The
   roadmap is the user-visible map of where the project is going;
   it must stay in lock-step with the PRD set. If the change
   renumbers or removes other releases, update those entries too.

### Phase 2 — Implementation

8. Only after the PRD is accepted AND `docs/roadmap.html` is in
   sync, develop the code so it matches the PRD.
9. If during implementation a **new split point** surfaces (a
   choice nobody discussed at Phase 1), **stop coding**, add the
   split to `docs/splits/v<X.Y.Z>-<slug>.html`, surface it to the
   user the same way, update the PRD with the decision, and re-gate.
10. If during implementation a constraint forces a **scope change**
    (a feature isn't supported by the underlying library, the design
    doesn't survive contact with reality, etc.), **stop coding**, go
    back to Phase 1, update the PRD AND the roadmap, get a fresh
    accept gate, then resume. Do not silently drop or add scope in
    the code.

### Phase 3 — User acceptance for release

11. Present the implementation to the user — running the user-based
    acceptance test from the PRD, or surfacing the diff plus a
    manual-test plan — and **wait for an explicit accept-or-reject
    gate**.
12. If the user **accepts**: write or update `CHANGELOG.md`, flip the
    roadmap entry's status to shipped, commit, tag `vX.Y.Z`
    (annotated), push branch + tag. This is the release.
13. If the user **rejects**: do NOT commit. Treat the feedback as
    driving a return to Phase 1 — update the PRD AND the roadmap with
    the new constraints, re-gate, re-implement.

## Principles

Design and implementation follow **SOLID** and **YAGNI** —
keep abstractions earned by use, keep features earned by the PRD.

## Living documents — keep `docs/roadmap.html` up to date (required workflow)

`docs/roadmap.html` is the project's single user-visible living document: the
roadmap (where the project is going) and — as the v2 surface settles — the
current architecture and the per-feature design decisions. **Keeping it current
is a required workflow step, not optional housekeeping:**

- **On PRD acceptance** (Phase 1, step 7): add or revise the release's roadmap
  entry.
- **On ship** (Phase 3, step 12): flip the entry to shipped, and if the release
  changed a subsystem's design or added / removed / deferred a capability, update
  the architecture / design-decision sections to match.
- **Never let it contradict reality** — a reversed or descoped decision is
  corrected in the same change that makes it true.

A shipped release whose design isn't reflected in `docs/roadmap.html` is an
incomplete release.

## Notes on autonomous mode

If the user has set a `/goal` and walked away (e.g. an unattended
overnight implementation), the agent may proceed through Phase 2 and
declare Phase 3 ready, but **must not** ship the tag without an
explicit accept gate from a human — leave the diff staged with the
CHANGELOG drafted and surface the gate question in the next user
turn.
