from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Iterator, TextIO

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows/non-POSIX fails closed at runtime.
    fcntl = None


def shadow_command_lock_path(state_path: str | Path, explicit_lock: str | Path | None = None) -> Path:
    """Return the lock path for a frozen shadow state.

    By default each state file gets its own sibling ``.lock`` file so independent
    frozen strategies do not block one another. Callers may provide an explicit
    shared lock path when they intentionally need a wider critical section.
    """

    if explicit_lock is not None and str(explicit_lock).strip():
        return Path(explicit_lock)
    return Path(str(Path(state_path)) + ".lock")


def paper_ledger_write_lock_path(ledger_path: str | Path) -> Path:
    """Return the lock path shared by ledger readers and the exclusive writer."""

    return Path(str(Path(ledger_path)) + ".lock")


def _write_owner_metadata(handle: TextIO) -> None:
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {
                "pid": os.getpid(),
                "acquired_at": datetime.now(timezone.utc).isoformat(),
            },
            sort_keys=True,
        )
        + "\n"
    )
    handle.flush()
    os.fsync(handle.fileno())


def _acquire_exclusive_lock(
    path: str | Path,
    *,
    unavailable_message: str,
    busy_message: str,
) -> Iterator[None]:
    if fcntl is None:
        raise ValueError(unavailable_message)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = target.open("a+", encoding="utf-8")
    acquired = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError as exc:
            raise ValueError(busy_message) from exc
        _write_owner_metadata(handle)
        yield
    finally:
        if acquired:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
        else:
            handle.close()


def _acquire_shared_lock(
    path: str | Path,
    *,
    unavailable_message: str,
) -> Iterator[None]:
    if fcntl is None:
        raise ValueError(unavailable_message)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = target.open("a+", encoding="utf-8")
    acquired = False
    try:
        # Intentionally blocking: the writer critical section is short and a verified
        # reader must observe either the state before the commit or the fully committed
        # ledger+checkpoint pair, never the transient interval between them.
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        acquired = True
        yield
    finally:
        if acquired:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
        else:
            handle.close()


@contextmanager
def exclusive_shadow_command_lock(path: str | Path) -> Iterator[None]:
    """Acquire a non-blocking process lock for one shadow command critical section.

    The operating system owns lock lifetime, so abnormal process termination releases
    the advisory lock automatically. The lock file itself is intentionally persistent;
    file existence is not interpreted as lock ownership.
    """

    yield from _acquire_exclusive_lock(
        path,
        unavailable_message="shadow command locking requires POSIX fcntl support",
        busy_message="another shadow command is already running for this state",
    )


@contextmanager
def exclusive_paper_ledger_write_lock(path: str | Path) -> Iterator[None]:
    """Serialize verify-and-commit operations for one paper ledger.

    The OS releases the writer lock on process termination, so a crash cannot leave a
    stale ownership marker behind. Verified readers coordinate through a shared lock.
    """

    yield from _acquire_exclusive_lock(
        path,
        unavailable_message="paper ledger writer locking requires POSIX fcntl support",
        busy_message="paper ledger is already being written",
    )


@contextmanager
def shared_paper_ledger_read_lock(path: str | Path) -> Iterator[None]:
    """Wait for any in-flight ledger writer, then hold a shared verified-read lock."""

    yield from _acquire_shared_lock(
        path,
        unavailable_message="paper ledger reader locking requires POSIX fcntl support",
    )
