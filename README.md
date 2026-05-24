# tool-loop-guard

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tool-loop-guard.svg)](https://pypi.org/project/tool-loop-guard/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Detect when an LLM agent gets stuck calling the same tool with the same args.** Zero deps. One class.

```python
from tool_loop_guard import LoopGuard, LoopDetectedError

guard = LoopGuard(window=8, threshold=3)

try:
    for step in agent_loop():
        guard.record(step.tool_name, step.tool_args)
        run_tool(step)
except LoopDetectedError as exc:
    print(f"agent looping on {exc.tool_name}({exc.tool_args})")
    print(f"  occurred {exc.count} times in the last {exc.window} calls")
    abort_or_replan()
```

## Why

The most common agent failure isn't a runtime error — it's a `search_web("anthropic prompt cache")` that returns nothing useful, an agent that decides to "try the same search one more time," and the loop running until your wallet notices.

`tool-loop-guard` is a hundred lines of sliding-window bookkeeping that throws when the same `(tool_name, args)` tuple shows up more than `threshold` times within the last `window` calls. That's the entire library.

A circuit breaker won't catch this (no errors). A deadline won't catch it (each call is fast). A budget will *eventually* catch it (after blowing through cost). The guard catches it on the third repeat.

## Install

```bash
pip install tool-loop-guard
```

## API

```python
guard = LoopGuard(
    window=8,        # how many recent calls to consider
    threshold=3,     # how many matches in that window are allowed
    key_fn=None,     # custom (tool_name, args) -> hashable
)

guard.record(tool_name, args)   # raises LoopDetectedError on the offending call
guard.would_raise(tool_name, args)  # peek without mutating
guard.reset()                   # clear the buffer
len(guard)                      # how many calls in the buffer
guard.recent_keys               # list of keys, oldest first
```

`record` raises `LoopDetectedError` with `.tool_name`, `.tool_args`, `.count`, `.window`, `.threshold` attributes for surface-area-friendly error handling. (The exception uses `tool_args` instead of `args` because `args` is reserved by `BaseException`.)

The default key is `(tool_name, canonical_json(args))`. To ignore noisy fields (e.g. `request_id`, `timestamp`) before comparison, pass your own `key_fn`:

```python
def stable_key(tool_name, args):
    cleaned = {k: v for k, v in (args or {}).items() if k not in {"request_id", "ts"}}
    return ("ext", tool_name, sorted(cleaned.items()))

guard = LoopGuard(window=10, threshold=3, key_fn=stable_key)
```

## Companion libraries

- [`llm-circuit-breaker`](https://github.com/MukundaKatta/llm-circuit-breaker) — opens on *provider* errors, not on agent confusion.
- [`agent-deadline`](https://github.com/MukundaKatta/agent-deadline) — cooperative per-task deadline. Pair with this guard: deadline catches "took too long," guard catches "looping fast."
- [`agentleash`](https://github.com/MukundaKatta/agentleash) — USD + call budget. Catches looping eventually; this guard catches it sooner.

## License

MIT
