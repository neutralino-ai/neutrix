"""Shared test fixtures."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_session_home(tmp_path, monkeypatch):
    """v1.5.2: redirect neutrix's session logs to a per-test tmp dir.

    ``session_store.session_dir`` honors ``$NEUTRIX_SESSION_HOME``; pointing it at
    the test's ``tmp_path`` guarantees no test ever writes a session JSONL to the
    real ``~/.cache/neutrix`` (and certainly not ``~/.claude``), no matter how the
    chat under test is constructed.
    """
    monkeypatch.setenv("NEUTRIX_SESSION_HOME", str(tmp_path))
