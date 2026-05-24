"""Tests for tool_loop_guard.LoopGuard."""

from __future__ import annotations

import pytest

from tool_loop_guard import LoopDetectedError, LoopGuard, default_key_fn


# ---- key function ---------------------------------------------------------


def test_default_key_includes_tool_name():
    a = default_key_fn("search", {"q": "x"})
    b = default_key_fn("fetch", {"q": "x"})
    assert a != b


def test_default_key_canonicalizes_arg_order():
    a = default_key_fn("t", {"a": 1, "b": 2})
    b = default_key_fn("t", {"b": 2, "a": 1})
    assert a == b


def test_default_key_none_args_is_empty_dict():
    assert default_key_fn("t", None) == default_key_fn("t", {})


def test_default_key_falls_back_to_repr_on_unserializable():
    class _NotJson:
        def __repr__(self):
            return "NotJson()"

    k = default_key_fn("t", _NotJson())
    assert "NotJson()" in k[1]


# ---- happy path ----------------------------------------------------------


def test_records_under_threshold_does_not_raise():
    guard = LoopGuard(window=4, threshold=3)
    guard.record("search", {"q": "x"})
    guard.record("search", {"q": "x"})  # second hit, still ok
    assert len(guard) == 2


def test_third_match_raises_loop_detected():
    guard = LoopGuard(window=4, threshold=3)
    guard.record("search", {"q": "x"})
    guard.record("search", {"q": "x"})
    with pytest.raises(LoopDetectedError) as exc:
        guard.record("search", {"q": "x"})
    assert exc.value.tool_name == "search"
    assert exc.value.count == 3
    assert exc.value.window == 4
    assert exc.value.threshold == 3


def test_different_args_do_not_count():
    guard = LoopGuard(window=4, threshold=3)
    guard.record("search", {"q": "a"})
    guard.record("search", {"q": "b"})
    guard.record("search", {"q": "c"})  # different args, no raise


def test_different_tools_do_not_count():
    guard = LoopGuard(window=4, threshold=3)
    guard.record("search", {"q": "x"})
    guard.record("fetch", {"q": "x"})
    guard.record("write", {"q": "x"})  # different tools, no raise


# ---- window eviction ------------------------------------------------------


def test_eviction_keeps_unrelated_loops_safe():
    guard = LoopGuard(window=4, threshold=3)
    guard.record("search", {"q": "x"})  # match #1 for search, window has 1 entry
    guard.record("fill_a", {})
    guard.record("fill_b", {})
    guard.record("fill_c", {})  # window now full: [search, fill_a, fill_b, fill_c]
    guard.record("search", {"q": "x"})  # search evicted, this is "#1 again"
    assert len(guard) == 4  # buffer stays at window size


def test_old_match_evicted_before_threshold():
    guard = LoopGuard(window=3, threshold=2)
    guard.record("search", {"q": "x"})  # 1st search
    guard.record("filler", {"k": 1})
    guard.record("filler", {"k": 2})    # window: [search, filler-1, filler-2]
    # The first search is now the oldest. Another search should evict it
    # (since the window holds 3, appending will drop the oldest).
    guard.record("search", {"q": "x"})  # would have been 2nd search, but
                                        # the first search was evicted first,
                                        # so this is the 1st again. No raise.
    assert len(guard) == 3


# ---- would_raise ----------------------------------------------------------


def test_would_raise_predicts_next_record_outcome():
    guard = LoopGuard(window=4, threshold=3)
    guard.record("search", {"q": "x"})
    guard.record("search", {"q": "x"})
    assert guard.would_raise("search", {"q": "x"}) is True
    assert guard.would_raise("search", {"q": "y"}) is False
    # would_raise must NOT mutate buffer
    assert len(guard) == 2


def test_would_raise_accounts_for_full_window_eviction():
    # Window holds 3. After 3 distinct calls the buffer is full.
    # A 4th call evicts the oldest before counting matches, so a project
    # that fills the buffer with unrelated calls then asks "would_raise"
    # for a fresh tool should say False.
    guard = LoopGuard(window=3, threshold=2)
    guard.record("a", None)
    guard.record("b", None)
    guard.record("c", None)
    assert guard.would_raise("fresh", None) is False


# ---- reset ---------------------------------------------------------------


def test_reset_clears_buffer():
    guard = LoopGuard(window=4, threshold=3)
    guard.record("search", {"q": "x"})
    guard.record("search", {"q": "x"})
    guard.reset()
    assert len(guard) == 0
    # safe to record again
    guard.record("search", {"q": "x"})
    guard.record("search", {"q": "x"})  # still 2 matches, under threshold


# ---- recent_keys ----------------------------------------------------------


def test_recent_keys_returns_oldest_first():
    guard = LoopGuard(window=4, threshold=4)
    guard.record("a", None)
    guard.record("b", None)
    guard.record("c", None)
    keys = guard.recent_keys
    assert [k[0] for k in keys] == ["a", "b", "c"]


# ---- custom key_fn -------------------------------------------------------


def test_custom_key_fn_can_ignore_noisy_field():
    def stable_key(tool_name, args):
        # ignore request_id when computing the key
        cleaned = {k: v for k, v in (args or {}).items() if k != "request_id"}
        return default_key_fn(tool_name, cleaned)

    guard = LoopGuard(window=4, threshold=3, key_fn=stable_key)
    guard.record("search", {"q": "x", "request_id": "r1"})
    guard.record("search", {"q": "x", "request_id": "r2"})
    with pytest.raises(LoopDetectedError):
        guard.record("search", {"q": "x", "request_id": "r3"})


# ---- argument validation -------------------------------------------------


@pytest.mark.parametrize("window", [-1, 0, 1])
def test_invalid_window_rejected(window):
    with pytest.raises(ValueError):
        LoopGuard(window=window, threshold=2)


@pytest.mark.parametrize("threshold", [-1, 0, 1])
def test_invalid_threshold_rejected(threshold):
    with pytest.raises(ValueError):
        LoopGuard(window=5, threshold=threshold)


def test_threshold_larger_than_window_rejected():
    with pytest.raises(ValueError):
        LoopGuard(window=3, threshold=4)


# ---- error info ----------------------------------------------------------


def test_loop_detected_error_carries_full_context():
    guard = LoopGuard(window=3, threshold=2)
    guard.record("search", {"q": "x"})
    with pytest.raises(LoopDetectedError) as exc_info:
        guard.record("search", {"q": "x"})
    exc = exc_info.value
    assert exc.tool_name == "search"
    assert exc.tool_args == {"q": "x"}
    assert exc.count == 2
    assert exc.window == 3
    assert exc.threshold == 2
    assert "search" in str(exc)
