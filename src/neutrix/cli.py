"""`neutrix` CLI entry point: load config, build ContextManager, launch chat."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
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
        "--continue", "-c", dest="continue_session", action="store_true",
        help="resume the most recent session in this directory",
    )
    p.add_argument(
        "--resume", metavar="ID", dest="resume_id",
        help="resume a specific session by id (prefix ok)",
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
    resume_session_id: str | None = None
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
    elif args.continue_session or args.resume_id:
        # v1.5.2: resume an auto-persisted CC-compatible session for this cwd.
        from neutrix.session_store import list_sessions, load_session, most_recent

        if args.resume_id:
            info = next(
                (s for s in list_sessions(os.getcwd())
                 if s.session_id.startswith(args.resume_id)),
                None,
            )
        else:
            info = most_recent(os.getcwd())
        if info is None:
            print("neutrix: no session to resume in this directory", file=sys.stderr)
            return 1
        raw_messages, _records, tasks = load_session(info.path)
        if raw_messages:
            seed_messages = list(raw_messages)
        loaded_tasks = list(tasks)
        resume_session_id = info.session_id
        logger.info(
            "resuming session {}: {} msgs, {} tasks",
            info.session_id, len(seed_messages), len(loaded_tasks),
        )

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

    # v1.5.0 diagnosability: log WHICH code this process actually loaded. A
    # running Python process freezes its imported source at startup; the
    # setuptools_scm __version__ is install-frozen and can lag the live editable
    # source. The module path + its mtime answer "what code is this, how fresh"
    # — the recurring stale-process confusion.
    import neutrix as _nx

    _src = os.path.dirname(_nx.__file__)
    try:
        _mtime = datetime.fromtimestamp(
            os.path.getmtime(os.path.join(_src, "context_manager.py"))
        ).strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        _mtime = "?"
    logger.info("neutrix {} starting — src={} (loaded {})", __version__, _src, _mtime)

    from neutrix.terminal_chat import TerminalChat

    chat = TerminalChat(ctx, config=config, render_markdown=not args.no_markdown)
    # v1.5.2: on resume, append to the resumed session file (skip its records).
    if resume_session_id is not None:
        chat._resume_session_id = resume_session_id
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
