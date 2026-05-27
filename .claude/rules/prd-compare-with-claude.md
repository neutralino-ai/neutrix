# Discuss design split points with the user BEFORE drafting the PRD

## Rule

For any behavior-change feature, the workflow is:

1. **Survey first.** Before writing a single line of the PRD, the agent reads Claude Code's analogous subsystem under `/datafs/users/dingxf/anthropic/cc2/src` and enumerates every **design split point** in the planned change — each place where there is more than one defensible choice.
2. **Write the splits to `docs/splits/v<X.Y.Z>-<slug>.html`.** One HTML file per upcoming release. The user opens it in a browser. Each split block records: name, what CC does (with file:line citation), the alternatives considered, the agent's recommendation, and a decision label slot the user fills in. Format/styling matches `docs/roadmap.html` (same CSS palette and entry shape). HTML — not Markdown — because the user reads it in a browser tab while deciding.
3. **Discuss with the user, turn by turn.** The HTML is the durable reference; the chat is where the back-and-forth happens. Present each split conversationally — "Split point #N: <what>. CC does <X>. Alternatives: <Y, Z>. Recommendation: <…>. Your call?" — so the user can redirect, introduce splits the agent missed, and reframe ones it got wrong.
4. **Draft the PRD only after the splits are settled.** The PRD contains a `## Claude Code reference` section between **Non-goals** and **Design** that records the decisions reached, one sub-block per split point, and **links to the splits HTML** for the full survey.

The splits HTML is the **durable record** of the discussion; the PRD's Claude Code reference section is the executive summary. If either reads like a list of conclusions the agent reached alone, the rule was skipped.

Decision label for each split point is exactly one of:
- **Follow CC** — same semantics as Claude Code.
- **Alternative** — we diverge on purpose, with the reason in the same sentence.
- **User-directed different** — the user explicitly chose a different path (cite the user instruction verbatim).
- **No CC analog** — CC has nothing comparable (state the design rationale).

In scope: any behavior-change feature — cancellation, conversation-history mutation, key bindings, transcript rendering, tool dispatch and result handling, slash commands, streaming, error surfaces. Not in scope: pure internal refactors with no behavior change, and features that genuinely have no CC analog (the YAML slot-switching commands, the onboarding TUI).

## Why

Drafting the PRD before the discussion forces the user into accept/reject on a proposal the agent already shaped — which means the agent's first guess locks in the decisions on the unsurfaced split points. Discussing the split points conversationally, in plain language, lets the user redirect cheaply at the exact moment of choice.

Concrete prior incident: v0.9.2 cancellation had at least four split points (kill signal, stream-close vs task-cancel, history rollback vs append marker, single broadcaster vs distributed AbortSignal). The agent drafted the PRD with all four pre-decided, the user accepted the PRD, and then v0.9.3 had to spend a whole release switching the history split point to CC's append-marker semantics. A 10-minute conversational survey *before* PRD drafting would have surfaced the choice and let the user decide it once.

## How to apply

1. **Survey CC.** Use the `learn-from-claude` skill or dispatch an `Explore` subagent with concrete keywords. Spend 10–20 minutes. A list with zero split points is almost certainly an under-survey; push back on yourself.

2. **Write `docs/splits/v<X.Y.Z>-<slug>.html`** with the split points. Re-use the CSS palette and entry shape from `docs/roadmap.html` so the file fits the existing UI style. One `<div class="split">` per split point, each carrying:
   - **Name** — one-line.
   - **CC behavior** — what CC does, with file:line citation.
   - **Alternatives** — list every defensible choice including CC's.
   - **Recommendation** — the agent's pick.
   - **Decision** — a placeholder for the user's choice (filled in during step 3). One of: Follow CC, Alternative — <reason>, User-directed different — <user quote>, No CC analog — <rationale>.

3. **Discuss with the user, turn by turn.** Surface each split conversationally: "Split point #N: <what>. CC does <X>. Alternatives: <Y, Z>. Recommendation: <…>. Your call?" The user reads the HTML alongside the chat, accepts or redirects, may add splits the agent missed. Update the HTML's Decision field as each is settled.

4. **Only then draft the PRD.** The `## Claude Code reference` section uses one sub-block per split point and **links to the splits HTML**:

   ```
   See `docs/splits/v<X.Y.Z>-<slug>.html` for the full survey.

   ### Split point: <one-line name>
   - **CC**: <what CC does, with file:line citation>
   - **Decision**: Follow CC / Alternative — <reason> / User-directed different — <user quote> / No CC analog — <rationale>
   ```

5. **Present the PRD for the Phase-1 accept gate** (per `release-workflow.md`). At this stage the user is confirming the writeup of decisions already taken, not making them for the first time. The gate is fast.

6. **Mid-implementation discovery.** If a new split point surfaces during coding, stop, **add it to the splits HTML**, surface it to the user the same way (step 3), update the PRD with the new sub-block, and re-gate. Same shape as the scope-change rule in `release-workflow.md`.

## Out of scope

- **1:1 behavior parity with CC.** The rule requires conscious choice on each split point, not mimicry.
- **Comparison with other reference codebases** (Codex, Aider, Continue, cursor). Voluntary additions under a split point are welcome but only CC is mandatory.
- **Implementation parity.** Match *semantics*, not code shape.
- **PRDs for the legacy `tui.py` Textual app.** Standing non-goal across the v0.9.x line.

## Stronger enforcement (optional)

This is a thinking-and-talking rule. No hook can enforce "you actually had the discussion." If a post-ship realization of "wait, that was a split point we never discussed" surfaces, treat it as a process bug and call it out by name in the next PRD's reference section so future readers learn from the miss.
