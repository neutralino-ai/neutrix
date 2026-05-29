# LLM-view and user-view differ only by fold, never by content

## Rule

Every byte the LLM receives in a round must appear in the user-visible
transcript for that round. LLM-view and user-view differ by **fold/expand
only** — never by omission, summarization, or truncation.

Channels in scope:
- System prompt
- Tool schemas (the `tools=` parameter)
- All `messages[*]` content (user, assistant, tool)
- Injected `<system-reminder>` turns
- Any future input channel (RAG snippets, cached prefixes, multimodal blobs, …)

## Why

Without parity, a user reading the transcript cannot reproduce or audit what the
LLM actually saw. Bugs that only manifest with the hidden context — wrong tool
schema, stale system prompt, accidental reminder injection — become invisible.
The transcript stops being a trustworthy record and becomes an editorial summary.

This invariant also constrains a recurring temptation in agent loops: "the LLM
needs this, but it's noisy for the user, so let's hide it." That trade always
favours the developer over the user. Folding is the right answer — same data,
less visual weight, reachable on demand.

## How to apply

1. **Single source of truth.** The renderer derives from the same
   `(system, messages, tools)` the LLM is sent (`ContextManager.round_bundle()`),
   not a parallel partial walk.
2. **One render hook per LLM-bound channel.** Adding a new input channel to the
   LLM in a PR must add the matching render in the same PR.
3. **Folding via expand-by-append, not omission.** The transcript is append-only
   scrollback (v0.6.8): a block cannot be collapsed/re-rendered in place. So a
   long channel renders a one-line **folded summary** with a byte/line count, and
   its full content is reachable by an **expand command that re-prints it below**
   (`/show <what>`, `/tool N`). "Folded" means a summary is present — never that
   the channel is absent.
4. **Label injected turns.** `<system-reminder>` and any loop-injected turn
   renders with a distinct prefix/style, not as a typed user turn.
5. **Invariant test.** A test asserts that for a recorded round, every populated
   channel of the `LLMRoundBundle` produced ≥1 render call — iterating the
   bundle's fields dynamically so a future channel trips it.

## Out of scope

- Layout choices (color, prefix string, fold threshold) — PRD-level design.
- Cosmetic asymmetries (renderer prefixes `<- ` to tool results; the LLM sees
  raw JSON) — presentation, not content.
- An in-place fold/expand **toggle** — would require reversing the append-only
  renderer (full-screen TUI). Expansion is by re-printing below.
- `--verbose`/debug modes that show *more* than the LLM saw — fine; parity is a
  lower bound, not a ceiling.
