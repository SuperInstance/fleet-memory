#!/usr/bin/env python3
"""living-minds-serialize.py — load-on-demand + idle-eviction Ollama model manager.

Keeps at most ONE model resident at a time, so a fleet GPU floor never pays
VRAM/RAM for more than one "living mind" at once — the single largest memory
win in the budget roadmap.

Policy
------
* `--load <name>` evicts the currently-resident model (`ollama stop`) BEFORE
  loading the requested one (`ollama run <name> --keepalive 5m ""`).
* Idle eviction is delegated to Ollama itself via `--keepalive` (default 5m):
  if nothing asks for the resident model within the window, Ollama unloads it
  automatically, returning the VRAM to the floor.
* `--load` refuses names outside the managed model list (the list IS the
  budget); extend the list with `--models` if a model should be loadable.
* `--status` prints the currently-resident model (or "none").
* `--stop` evicts the currently-resident model immediately.

Exit codes: 0 = success, 1 = runtime failure, 2 = usage error.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys

DEFAULT_MODELS = "granite3.1-dense:2b,llama3.2,qwen2.5:3b"
DEFAULT_KEEPALIVE = "5m"

# `ollama ps` table header (older clients lack `--format json`).
_PS_HEADER = "NAME"


def _run(argv: list[str], binary: str) -> subprocess.CompletedProcess:
    """Run an ollama subcommand, non-interactive (stdin from /dev/null)."""
    return subprocess.run(
        [binary, *argv],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )


def resident_models(binary: str) -> list[str]:
    """Return the names of all models currently resident in Ollama.

    Tries `ollama ps --format json` first (newer clients); falls back to
    parsing the plain-text table (NAME column) when the client lacks the
    flag. Never raises — a missing/broken ollama yields [].
    """
    # JSON path (best effort).
    p = _run(["ps", "--format", "json"], binary)
    if p.returncode == 0 and p.stdout.strip():
        try:
            data = json.loads(p.stdout)
            return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass  # fall through to table parse

    # Table path: `NAME ID SIZE PROCESSOR UNTIL`, header on line 1.
    p = _run(["ps"], binary)
    if p.returncode != 0:
        return []
    names: list[str] = []
    for line in p.stdout.splitlines():
        cols = line.split()
        if cols and cols[0] != _PS_HEADER:
            names.append(cols[0])
    return names


def current_resident(binary: str, managed: list[str]) -> str | None:
    """The resident model, preferring one inside the managed set."""
    resident = resident_models(binary)
    if not resident:
        return None
    for name in resident:
        if name in managed:
            return name
    return resident[0]


def cmd_status(binary: str, managed: list[str]) -> int:
    resident = resident_models(binary)
    if not resident:
        print("none")
        return 0
    print(resident[0])
    unmanaged = [n for n in resident if n not in managed]
    if unmanaged:
        sys.stderr.write(
            f"warning: resident model(s) outside managed list: {', '.join(unmanaged)}\n"
        )
    return 0


def cmd_stop(binary: str, managed: list[str]) -> int:
    resident = current_resident(binary, managed)
    if resident is None:
        print("none resident; nothing to stop")
        return 0
    p = _run(["stop", resident], binary)
    if p.returncode != 0:
        sys.stderr.write(f"error: failed to stop '{resident}': {p.stderr.strip()}\n")
        return 1
    print(f"stopped {resident}")
    return 0


def cmd_load(name: str, keepalive: str, binary: str, managed: list[str]) -> int:
    if name not in managed:
        sys.stderr.write(
            f"error: '{name}' is not in the managed model list ({', '.join(managed)})\n"
        )
        return 2

    resident = current_resident(binary, managed)

    if resident == name:
        # Already the resident — refresh keepalive only (idempotent, no eviction).
        p = _run(["run", name, "--keepalive", keepalive, ""], binary)
        if p.returncode != 0:
            sys.stderr.write(f"error: keepalive refresh failed: {p.stderr.strip()}\n")
            return 1
        print(f"{name} already resident (keepalive refreshed: {keepalive})")
        return 0

    # Evict before load — never allow two residents.
    if resident is not None:
        p = _run(["stop", resident], binary)
        if p.returncode != 0:
            sys.stderr.write(
                f"error: failed to evict '{resident}': {p.stderr.strip()} — aborting load\n"
            )
            return 1
        print(f"evicted {resident}")

    p = _run(["run", name, "--keepalive", keepalive, ""], binary)
    if p.returncode != 0:
        sys.stderr.write(f"error: failed to load '{name}': {p.stderr.strip()}\n")
        return 1
    print(f"loaded {name} (keepalive {keepalive})")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="living-minds-serialize.py",
        description=(
            "Load-on-demand + idle-eviction Ollama model manager. Keeps at most "
            "ONE model resident at a time."
        ),
        epilog=(
            "Idle eviction is handled by Ollama itself: every load sets "
            f"--keepalive (default {DEFAULT_KEEPALIVE}), so an untouched model "
            "is automatically unloaded when the window expires."
        ),
    )
    parser.add_argument(
        "--models",
        default=DEFAULT_MODELS,
        metavar="LIST",
        help=(
            "comma-separated managed model names (the VRAM budget). "
            f"default: {DEFAULT_MODELS}"
        ),
    )
    parser.add_argument(
        "--keepalive",
        default=DEFAULT_KEEPALIVE,
        metavar="DUR",
        help=(
            "idle-eviction window passed to `ollama run --keepalive`, "
            f"e.g. 5m, 1h (default: {DEFAULT_KEEPALIVE})"
        ),
    )
    parser.add_argument(
        "--ollama",
        default="ollama",
        metavar="PATH",
        help="path to the ollama binary (default: ollama on PATH)",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--load",
        metavar="NAME",
        help="evict the current resident, then load NAME (must be in --models)",
    )
    action.add_argument(
        "--status",
        action="store_true",
        help="print the currently-resident model (or 'none')",
    )
    action.add_argument(
        "--stop",
        action="store_true",
        help="evict the currently-resident model immediately",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if not shutil.which(args.ollama):
        sys.stderr.write(f"error: ollama binary not found: {args.ollama}\n")
        return 1

    managed = [m.strip() for m in args.models.split(",") if m.strip()]
    if not managed:
        sys.stderr.write("error: --models must contain at least one name\n")
        return 2

    if args.status:
        return cmd_status(args.ollama, managed)
    if args.stop:
        return cmd_stop(args.ollama, managed)
    if args.load:
        return cmd_load(args.load, args.keepalive, args.ollama, managed)

    # Unreachable: the mutually exclusive group is required.
    sys.stderr.write("error: one of --load/--status/--stop is required\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
