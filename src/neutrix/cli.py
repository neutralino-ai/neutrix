"""`neutrix` CLI entry point: load config, build ContextManager, launch chat."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from loguru import logger

from neutrix import __version__, transcript
from neutrix.config import (
    CONFIG_PATH,
    ConfigError,
    bootstrap_config,
    load_config,
    resolve_initial_slot,
)
from neutrix.context_files import compose_system_prompt
from neutrix.context_manager import DEFAULT_SYSTEM_PROMPT, ContextManager
from neutrix.executor import Executor
from neutrix.llm import OpenAIChatLLM
from neutrix.permissions import load_policy
from neutrix.store import ChatStore


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neutrix",
        description=(
            "A multi-provider terminal agent (DeepSeek, GLM, Claude via IHEP). "
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

    slot = strong_slot or fast_slot
    assert slot is not None  # for type-checker; guarded above

    llm = OpenAIChatLLM(slot)
    store = ChatStore()
    executor = Executor(store=store)
    # v1.4.0: load permission rules from .claude/settings(.local).json (CC-compat).
    executor.policy = load_policy(os.getcwd())

    # v1.2.0: compose CLAUDE.md/AGENTS.md (+ user ~/.claude/CLAUDE.md) into the
    # system prompt so the agent has project memory (.claude/-compatible).
    effective_system_prompt = compose_system_prompt(DEFAULT_SYSTEM_PROMPT, os.getcwd())
    seed_messages: list[dict] = [{"role": "system", "content": effective_system_prompt}]
    loaded_tasks: list = []
    if args.load:
        try:
            loaded_store, metadata = transcript.load(args.load)
            seed_messages = list(metadata["raw_messages"]) or seed_messages
            loaded_tasks = list(loaded_store.tasks)
            logger.info(
                "loaded transcript: {} msgs, {} tasks",
                len(seed_messages),
                len(loaded_tasks),
            )
        except Exception as e:
            print(f"neutrix: error loading transcript: {e}", file=sys.stderr)
            return 1

    ctx = ContextManager(
        slot=slot,
        llm=llm,
        executor=executor,
        store=store,
        system_prompt=effective_system_prompt,
        use_tools=not args.no_tools,
        messages=seed_messages,
    )
    if loaded_tasks:
        ctx.store.replace_tasks(loaded_tasks)

    _configure_chat_logging()

    from neutrix.terminal_chat import TerminalChat

    chat = TerminalChat(ctx, config=config, render_markdown=not args.no_markdown)
    chat.run()
    return 0


def _configure_chat_logging() -> None:
    """Keep operational logs out of the interactive terminal surface."""
    log_path = Path("~/.cache/neutrix/neutrix.log").expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(log_path, rotation="1 MB", retention=3, level="INFO")


if __name__ == "__main__":
    raise SystemExit(main())
