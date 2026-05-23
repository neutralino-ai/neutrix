# Every change requires PRD + CHANGELOG entry + new SemVer tag

## Rule

Every change to this project — code, packaging, user-facing docs — must:

1. **Have a PRD** at `docs/PRDs/v<X.Y.Z>-<slug>.md`, written or updated **before** implementation.
2. **Update `CHANGELOG.md`** with a new `## [vX.Y.Z] — YYYY-MM-DD` section listing the change.
3. **Ship as a new SemVer version** via an annotated git tag `vX.Y.Z` on the commit.

No commit that lands on `main` without all three. WIP / experiments stay on feature branches and consolidate into one release commit at merge time.

## Why

This repo uses `setuptools_scm`, so the published version **is** the git tag — there's no separate `pyproject.toml` version to bump. Anchoring every release commit to `(PRD, CHANGELOG entry, tag)` keeps the version line, the human-readable changelog, and the design rationale in lock-step. A user running `neutrix --version` can trace `vX.Y.Z` → CHANGELOG entry → PRD without archaeology. Skip the rule once and the chain breaks for good.

## How to apply

1. **At task start, pick the next SemVer:**
   - `patch` (e.g. `0.1.0 → 0.1.1`): bug fix, no user-visible behavior change.
   - `minor` (`0.1.0 → 0.2.0`): new feature, backwards-compatible.
   - `major` (`0.1.0 → 1.0.0`): breaking change. (Pre-1.0, breaking changes may bump `minor` instead — call it out in the PRD.)

2. **Write the PRD** at `docs/PRDs/v<X.Y.Z>-<short-slug>.md` (create the dir on first use). Minimum sections: **Problem · Goal · Non-goals · Design · Acceptance**. Keep updating it if scope shifts during implementation.

3. **Before committing**, prepend a new block to `CHANGELOG.md` (Keep-a-Changelog style):
   ```markdown
   ## [vX.Y.Z] — YYYY-MM-DD
   ### Added | Changed | Fixed | Removed
   - <one line per change>

   See [docs/PRDs/vX.Y.Z-slug.md](docs/PRDs/vX.Y.Z-slug.md).
   ```

4. **Commit + tag + push as a unit:**
   ```bash
   git add <files> CHANGELOG.md docs/PRDs/vX.Y.Z-*.md
   git commit -m "vX.Y.Z: <short subject>"
   git tag -a vX.Y.Z -m "vX.Y.Z — <short subject>"
   git push origin <branch>
   git push origin vX.Y.Z
   ```

5. **Done check:** after `pip install -e .` the tag is live; `neutrix --version` reports `X.Y.Z`; CHANGELOG and PRD files exist for that tag.

## Out of scope

- WIP commits on feature branches that will be squashed before merge.
- Changes to `.gitignore`, CI YAML, or other purely-internal plumbing that never affects users — these can ride along with the next user-facing release.
- Internal tooling (developer-only scripts) that doesn't ship in the wheel.

## Stronger enforcement (optional)

The "CHANGELOG entry exists for the new tag" and "tag is annotated, not lightweight" checks can be wired as a pre-push hook so the model literally can't push a non-conforming release. The PRD step is judgment work and can't be automated. To wire the mechanical checks, ask for `update-config` to add a PreToolUse hook.
