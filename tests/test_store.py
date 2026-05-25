"""Unit tests for :mod:`neutrix.store`.

These tests run the store in isolation — no terminal, no agent, no LLM.
They cover the boundary contract that future renderers will rely on:
read-side returns immutable tuples; mutators notify subscribers;
multiple mutations between yields coalesce; OpenAI bridge round-trips.
"""
from __future__ import annotations

import asyncio

from neutrix.store import (
    ChatStore,
    MessageRecord,
    PendingToolCall,
    QueuedUserMessage,
    Task,
    openai_to_record,
    record_to_openai,
)


def test_messages_returns_tuple_not_internal_list():
    store = ChatStore()
    store.append_message(MessageRecord(role="user", content="hi"))
    msgs = store.messages
    assert isinstance(msgs, tuple)
    assert len(msgs) == 1
    # mutating the returned tuple is not possible (tuple is immutable),
    # but mutating the underlying list via a stale reference should not
    # affect future reads — re-read confirms the store still owns it.
    assert store.messages == msgs
    store.append_message(MessageRecord(role="assistant", content="hello"))
    assert len(store.messages) == 2
    assert len(msgs) == 1  # the earlier snapshot is unchanged


def test_enqueue_and_dequeue_user_messages():
    store = ChatStore()
    a = store.enqueue_user("first")
    b = store.enqueue_user("second")
    assert isinstance(a, QueuedUserMessage)
    assert store.queued_user_messages == (a, b)
    assert store.dequeue_user() == a
    assert store.queued_user_messages == (b,)
    assert store.dequeue_user() == b
    assert store.queued_user_messages == ()
    assert store.dequeue_user() is None


def test_assistant_stream_lifecycle_appends_message():
    store = ChatStore()
    assert store.pending_assistant_text is None
    store.start_assistant_stream()
    assert store.pending_assistant_text == ""
    store.extend_assistant_stream("hel")
    store.extend_assistant_stream("lo")
    assert store.pending_assistant_text == "hello"
    record = store.finish_assistant_stream()
    assert record is not None
    assert record.role == "assistant"
    assert record.content == "hello"
    assert store.pending_assistant_text is None
    assert store.messages[-1] == record


def test_finish_assistant_stream_without_start_returns_none():
    store = ChatStore()
    assert store.finish_assistant_stream() is None
    assert store.messages == ()


def test_pending_tool_call_lifecycle():
    store = ChatStore()
    call_a = store.add_pending_tool_call("list_dir", '{"path": "."}')
    call_b = store.add_pending_tool_call("read_file", '{"path": "x"}')
    assert isinstance(call_a, PendingToolCall)
    assert store.pending_tool_calls == (call_a, call_b)
    removed = store.remove_pending_tool_call("read_file")
    assert removed == call_b
    assert store.pending_tool_calls == (call_a,)
    store.clear_pending_tool_calls()
    assert store.pending_tool_calls == ()


def test_remove_pending_tool_call_falls_back_to_first_when_name_unknown():
    store = ChatStore()
    a = store.add_pending_tool_call("list_dir", "{}")
    store.add_pending_tool_call("read_file", "{}")
    removed = store.remove_pending_tool_call("not-a-tool")
    assert removed == a
    assert len(store.pending_tool_calls) == 1


def test_reset_clears_everything_and_optionally_seeds_system_prompt():
    store = ChatStore()
    store.append_message(MessageRecord(role="user", content="hi"))
    store.enqueue_user("queued")
    store.add_pending_tool_call("t", "{}")
    store.start_assistant_stream()
    store.extend_assistant_stream("partial")
    store.add_task("task")

    store.reset(system_prompt="be brief")
    assert len(store.messages) == 1
    assert store.messages[0].role == "system"
    assert store.messages[0].content == "be brief"
    assert store.queued_user_messages == ()
    assert store.pending_tool_calls == ()
    assert store.pending_assistant_text is None
    assert store.tasks == ()
    # After reset the next id starts from 1 again.
    fresh = store.add_task("first")
    assert fresh.id == "1"


# ---- tasks ------------------------------------------------------------------


def test_add_task_assigns_monotonic_string_ids():
    store = ChatStore()
    a = store.add_task("first")
    b = store.add_task("second", description="detail")
    assert isinstance(a, Task)
    assert (a.id, a.subject, a.status) == ("1", "first", "pending")
    assert (b.id, b.subject, b.description) == ("2", "second", "detail")
    assert store.tasks == (a, b)


def test_update_task_changes_fields_and_refreshes_updated_at():
    store = ChatStore()
    task = store.add_task("first")
    original_updated_at = task.updated_at

    updated = store.update_task(task.id, status="in_progress")
    assert updated is not None
    assert updated.status == "in_progress"
    assert updated.subject == "first"
    assert updated.updated_at >= original_updated_at
    assert store.tasks[0] == updated


def test_update_task_no_change_returns_existing_unchanged():
    store = ChatStore()
    task = store.add_task("first")
    same = store.update_task(task.id, subject="first")
    assert same == task


def test_update_task_unknown_id_returns_none():
    store = ChatStore()
    store.add_task("first")
    assert store.update_task("99", status="completed") is None


def test_remove_task_returns_removed_record_or_none():
    store = ChatStore()
    a = store.add_task("first")
    b = store.add_task("second")
    removed = store.remove_task(a.id)
    assert removed == a
    assert store.tasks == (b,)
    assert store.remove_task("99") is None


def test_replace_tasks_seeds_next_id_from_max_id_not_length():
    """Deleted-then-saved tasks leave gaps in the id sequence; the next
    id must be max(loaded_ids) + 1, not len(loaded) + 1, otherwise a
    fresh add_task would collide with an existing id."""
    store = ChatStore()
    loaded = [
        Task(id="2", subject="two"),
        Task(id="5", subject="five"),
    ]
    store.replace_tasks(loaded)
    assert store.tasks == tuple(loaded)
    fresh = store.add_task("new")
    assert fresh.id == "6"


def test_replace_tasks_with_empty_list_resets_counter_to_one():
    store = ChatStore()
    store.add_task("a")
    store.add_task("b")
    store.replace_tasks([])
    assert store.tasks == ()
    assert store.add_task("c").id == "1"


async def test_task_mutations_notify_subscribers():
    store = ChatStore()
    woke = asyncio.Event()
    yielded = 0

    async def watcher() -> None:
        nonlocal yielded
        async for _ in store.changes():
            yielded += 1
            woke.set()
            if yielded == 3:
                return

    task = asyncio.create_task(watcher())
    await asyncio.sleep(0)

    store.add_task("a")
    await asyncio.wait_for(woke.wait(), timeout=1.0)
    woke.clear()

    only = store.tasks[0]
    store.update_task(only.id, status="in_progress")
    await asyncio.wait_for(woke.wait(), timeout=1.0)
    woke.clear()

    store.remove_task(only.id)
    await asyncio.wait_for(task, timeout=1.0)
    assert yielded == 3


async def test_changes_yields_on_every_subscribed_mutation():
    store = ChatStore()
    seen: list[int] = []

    async def watcher() -> None:
        count = 0
        async for _ in store.changes():
            count += 1
            seen.append(count)
            if count == 3:
                return

    task = asyncio.create_task(watcher())
    # Give the watcher a chance to subscribe.
    await asyncio.sleep(0)
    store.append_message(MessageRecord(role="user", content="a"))
    await asyncio.sleep(0)
    store.enqueue_user("queued")
    await asyncio.sleep(0)
    store.append_message(MessageRecord(role="assistant", content="b"))
    await asyncio.sleep(0)
    await asyncio.wait_for(task, timeout=1.0)
    assert seen == [1, 2, 3]


async def test_changes_coalesces_consecutive_mutations_between_yields():
    store = ChatStore()
    woke = asyncio.Event()
    proceed = asyncio.Event()
    yields = 0

    async def watcher() -> None:
        nonlocal yields
        async for _ in store.changes():
            yields += 1
            woke.set()
            await proceed.wait()
            proceed.clear()
            if yields == 2:
                return

    task = asyncio.create_task(watcher())
    await asyncio.sleep(0)

    # Burst three mutations before the watcher gets to clear its event.
    store.append_message(MessageRecord(role="user", content="a"))
    store.append_message(MessageRecord(role="user", content="b"))
    store.append_message(MessageRecord(role="user", content="c"))
    await asyncio.wait_for(woke.wait(), timeout=1.0)
    woke.clear()
    # All three mutations collapsed into the single first yield.
    assert yields == 1

    # Let the watcher loop around; while it is awaiting again, a single
    # mutation should produce exactly one more yield.
    proceed.set()
    store.append_message(MessageRecord(role="user", content="d"))
    await asyncio.wait_for(woke.wait(), timeout=1.0)
    assert yields == 2
    proceed.set()
    await asyncio.wait_for(task, timeout=1.0)


async def test_changes_supports_independent_subscribers():
    store = ChatStore()
    counts = {"a": 0, "b": 0}

    async def watcher(name: str) -> None:
        async for _ in store.changes():
            counts[name] += 1
            if counts[name] == 2:
                return

    task_a = asyncio.create_task(watcher("a"))
    task_b = asyncio.create_task(watcher("b"))
    await asyncio.sleep(0)
    store.append_message(MessageRecord(role="user", content="x"))
    await asyncio.sleep(0)
    store.append_message(MessageRecord(role="user", content="y"))
    await asyncio.wait_for(asyncio.gather(task_a, task_b), timeout=1.0)
    assert counts == {"a": 2, "b": 2}


async def test_aclose_unsubscribes_the_iterator():
    store = ChatStore()
    iterator = store.changes()

    # Kick the iterator so the generator body runs through subscription.
    pull = asyncio.create_task(iterator.__anext__())
    await asyncio.sleep(0)
    assert len(store._subscribers) == 1

    store.append_message(MessageRecord(role="user", content="hi"))
    await asyncio.wait_for(pull, timeout=1.0)

    await iterator.aclose()
    # The subscriber's event is removed by the finally clause; further
    # mutations have no one to notify on this iterator.
    assert store._subscribers == set()


# ---- OpenAI bridge ----------------------------------------------------------


def test_openai_round_trip_user_message():
    raw = {"role": "user", "content": "hello"}
    record = openai_to_record(raw)
    assert record.role == "user"
    assert record.content == "hello"
    assert record_to_openai(record) == raw


def test_openai_round_trip_assistant_with_tool_calls_preserves_extras():
    raw = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "function": {"name": "list_dir", "arguments": "{}"}}
        ],
    }
    record = openai_to_record(raw)
    assert record.role == "assistant"
    assert record.content is None
    assert record.extra is not None
    assert record.extra.get("tool_calls") == raw["tool_calls"]
    assert record_to_openai(record) == raw


def test_openai_round_trip_tool_message():
    raw = {"role": "tool", "tool_call_id": "c1", "content": "ok"}
    record = openai_to_record(raw)
    assert record.role == "tool"
    assert record.tool_call_id == "c1"
    assert record.content == "ok"
    assert record_to_openai(record) == raw


def test_openai_to_record_normalizes_unknown_role():
    record = openai_to_record({"role": "weird", "content": "x"})
    assert record.role == "system"


def test_openai_to_record_stringifies_non_string_content():
    record = openai_to_record({"role": "user", "content": 42})
    assert record.content == "42"


