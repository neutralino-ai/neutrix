"""`neutrix` CLI entry point: load config, build Agent, launch TUI."""
from __future__ import annotations

import argparse
import sys

from loguru import logger

from neutrix import __version__
from neutrix.agent import Agent
from neutrix.config import (
    CONFIG_PATH,
    ConfigError,
    bootstrap_config,
    load_config,
    resolve_initial_slot,
)
from neutrix.session import load as session_load


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neutrix",
        description=(
            "A multi-provider TUI agent (DeepSeek, GLM, Claude via IHEP). "
            "Configure providers and the fast/strong slots in "
            f"{CONFIG_PATH}."
        ),
    )
    p.add_argument(
        "--load", metavar="PATH",
        help="load a saved session JSON file",
    )
    p.add_argument(
        "--no-tools", action="store_true",
        help="disable tool calling for this session",
    )
    p.add_argument(
        "--no-markdown", action="store_true",
        help="render assistant replies as plain text instead of markdown",
    )
    p.add_argument(
        "-V", "--version", action="version", version=f"neutrix {__version__}",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not CONFIG_PATH.exists():
        path = bootstrap_config()
        print(
            f"neutrix: created default config at {path}\n"
            f"edit it to add at least one provider api_key, then re-run `neutrix`.",
            file=sys.stderr,
        )
        return 0

    try:
        config = load_config()
    except ConfigError as e:
        print(f"neutrix: {e}", file=sys.stderr)
        return 1

    fast_slot, strong_slot = resolve_initial_slot(config)
    if fast_slot is None and strong_slot is None:
        from neutrix.onboard import run_onboarding

        launched = run_onboarding(config)
        if not launched:
            print("neutrix: onboarding cancelled.", file=sys.stderr)
            return 0
        try:
            config = load_config()
        except ConfigError as e:
            print(f"neutrix: {e}", file=sys.stderr)
            return 1
        fast_slot, strong_slot = resolve_initial_slot(config)
        if fast_slot is None and strong_slot is None:
            print(
                "neutrix: no slot is usable after onboarding; check "
                f"{config.path}.",
                file=sys.stderr,
            )
            return 1

    slot = fast_slot or strong_slot
    assert slot is not None  # for type-checker; guarded above

    agent = Agent(slot=slot, use_tools=not args.no_tools)

    if args.load:
        try:
            payload = session_load(args.load)
            agent.messages = payload["messages"]
            logger.info("loaded session: {} msgs", len(agent.messages))
        except Exception as e:
            print(f"neutrix: error loading session: {e}", file=sys.stderr)
            return 1

    from neutrix.tui import NeutrixApp

    app = NeutrixApp(agent, config=config, render_markdown=not args.no_markdown)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
