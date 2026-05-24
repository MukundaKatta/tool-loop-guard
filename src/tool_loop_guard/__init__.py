"""tool-loop-guard - detect repeated identical tool calls in an agent loop.

When an LLM agent gets confused, the most common failure mode is calling the
same tool with the same args over and over. A circuit breaker won't catch
this (no errors), a deadline won't catch it (single calls are fast), but
the wall clock keeps ticking and the bill keeps growing.

`LoopGuard` watches a sliding window of recent tool calls and raises
`LoopDetectedError` the moment a (tool_name, args) pair repeats more than
`threshold` times within the last `window` calls.

    from tool_loop_guard import LoopGuard, LoopDetectedError

    guard = LoopGuard(window=8, threshold=3)

    try:
        for step in agent_loop():
            guard.record(step.tool_name, step.tool_args)
            run_tool(step)
    except LoopDetectedError as exc:
        print(f"agent looping on {exc.tool_name}({exc.args})")
        print(f"  occurred {exc.count} times in the last {exc.window} calls")

The default key is `(tool_name, canonical_json(args))`. For more exotic
matching (e.g. ignore a `request_id` field in args) pass a custom
`key_fn(tool_name, args) -> hashable`.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Hashable

__version__ = "0.1.0"
__all__ = [
    "LoopGuard",
    "LoopDetectedError",
    "default_key_fn",
]


KeyFn = Callable[[str, Any], Hashable]


def default_key_fn(tool_name: str, args: Any) -> Hashable:
    """Canonical (tool_name, json) key. Sorted-key JSON of args.

    `args` may be `None` (treated as `{}`), a dict, a list, or any
    JSON-serializable value. For non-JSON-serializable values we fall
    back to `repr()` for stability.
    """
    if args is None:
        canon = "{}"
    else:
        try:
            canon = json.dumps(args, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            canon = repr(args)
    return (tool_name, canon)


class LoopDetectedError(RuntimeError):
    """Raised when the same (tool_name, args) pair fires too often.

    Attributes:
        tool_name: name of the offending tool
        tool_args: the args object that was matched (last recorded form)
        count: how many times the key was seen inside the window
        window: configured window size
        threshold: configured threshold
    """

    def __init__(
        self,
        tool_name: str,
        tool_args: Any,
        count: int,
        window: int,
        threshold: int,
    ) -> None:
        self.tool_name = tool_name
        # `args` is reserved by BaseException, so we expose `tool_args`.
        self.tool_args = tool_args
        self.count = count
        self.window = window
        self.threshold = threshold
        super().__init__(
            f"tool {tool_name!r} called {count} times in the last {window} "
            f"calls (threshold={threshold})"
        )


@dataclass(frozen=True)
class _Recent:
    key: Hashable
    args: Any
    tool_name: str


class LoopGuard:
    """Sliding-window detector for repeated tool calls.

    Args:
        window: how many recent calls to look at. Must be >= 2.
        threshold: how many matches inside that window are allowed before
            `LoopDetectedError` is raised. Must satisfy `2 <= threshold <= window`.
        key_fn: optional function to produce a hashable key. Defaults to
            `default_key_fn`. Pass your own to ignore noisy fields (e.g.
            request_ids, timestamps) before comparison.
    """

    def __init__(
        self,
        window: int = 8,
        threshold: int = 3,
        *,
        key_fn: KeyFn | None = None,
    ) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        if threshold < 2:
            raise ValueError("threshold must be >= 2")
        if threshold > window:
            raise ValueError("threshold must be <= window")
        self._window = window
        self._threshold = threshold
        self._key_fn: KeyFn = key_fn or default_key_fn
        self._buf: deque[_Recent] = deque(maxlen=window)

    # ---- introspection ------------------------------------------------

    @property
    def window(self) -> int:
        return self._window

    @property
    def threshold(self) -> int:
        return self._threshold

    @property
    def recent_keys(self) -> list[Hashable]:
        """Recent keys, oldest first."""
        return [r.key for r in self._buf]

    def __len__(self) -> int:
        return len(self._buf)

    # ---- core ---------------------------------------------------------

    def record(self, tool_name: str, args: Any = None) -> None:
        """Record a call. Raises `LoopDetectedError` if threshold exceeded."""
        key = self._key_fn(tool_name, args)
        self._buf.append(_Recent(key=key, args=args, tool_name=tool_name))
        count = sum(1 for r in self._buf if r.key == key)
        if count >= self._threshold:
            raise LoopDetectedError(
                tool_name=tool_name,
                tool_args=args,
                count=count,
                window=self._window,
                threshold=self._threshold,
            )

    def would_raise(self, tool_name: str, args: Any = None) -> bool:
        """Return True if the *next* `record` for this key would raise.

        Lets callers preview the outcome without mutating the buffer.
        """
        key = self._key_fn(tool_name, args)
        # Simulate appending without overflow side-effects: if the buffer is
        # already full and the oldest entry has the same key, the projected
        # count loses one before gaining one. Otherwise it's len(matches) + 1.
        existing = [r.key for r in self._buf]
        if len(existing) == self._window:
            existing = existing[1:]  # the oldest would be evicted on append
        projected = sum(1 for k in existing if k == key) + 1
        return projected >= self._threshold

    def reset(self) -> None:
        """Forget all recorded calls."""
        self._buf.clear()
