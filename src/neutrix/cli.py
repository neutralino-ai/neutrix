"""`neutrix` CLI entry point: parse args, build Agent, launch TUI."""
from __future__ import annotations

import argparse
import sys

from loguru import logger

from neutrix import __version__
from neutrix.agent import Agent
from neutrix.config import (
    PROVIDERS,
    default_model_for,
    default_provider_name,
    get_provider,
    load_env,
)
from neutrix.session import load as session_load


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neutrix",
        description="A simple multi-provider TUI agent (DeepSeek, GLM, Claude).",
    )
    p.add_argument(
        "-p", "--provider",
        choices=sorted(PROVIDERS),
        help="LLM provider (default: $NEUTRIX_PROVIDER or 'deepseek')",
    )
    p.add_argument(
        "-m", "--model",
        help="model name (default: provider's default or $NEUTRIX_MODEL)",
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
    load_env()
    args = build_parser().parse_args(argv)

    provider_name = args.provider or default_provider_name()
    try:
        provider = get_provider(provider_name)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    model = args.model or default_model_for(provider)

    try:
        agent = Agent(
            provider=provider,
            model=model,
            use_tools=not args.no_tools,
        )
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.load:
        try:
            payload = session_load(args.load)
        except Exception as e:
            print(f"error loading session: {e}", file=sys.stderr)
            return 1
        loaded_provider = get_provider(payload["provider"])
        agent.switch(loaded_provider, payload["model"])
        agent.messages = payload["messages"]
        logger.info("loaded session: {} msgs", len(agent.messages))

    # Import here so optional textual dep failures surface only when running TUI.
    from neutrix.tui import NeutrixApp

    app = NeutrixApp(agent, render_markdown=not args.no_markdown)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
