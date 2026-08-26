"""Cross-platform, process-tree-bounded child execution.

This module is intentionally internal.  Callers translate its small exception
taxonomy into their public service-specific errors.
"""

from __future__ import annotations

import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path


class BoundedProcessError(RuntimeError):
    """Base error carrying only byte-capped child diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
    ) -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


class BoundedProcessTimeout(BoundedProcessError):
    """The absolute child-operation deadline expired."""


class BoundedProcessCancelled(BoundedProcessError):
    """The caller requested cancellation."""


class BoundedProcessOutputLimit(BoundedProcessError):
    """Combined stdout and stderr exceeded their byte ceiling."""


class BoundedProcessLaunchError(BoundedProcessError):
    """The child or its containment boundary could not be established."""


class BoundedProcessDrainError(BoundedProcessError):
    """The bounded pipe/process cleanup did not finish by the deadline."""


class _WindowsJob:
    """Kill-on-close Windows Job Object containing one subprocess tree."""

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        import ctypes
        from ctypes import wintypes

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        self._kernel32 = kernel32
        self._handle = handle
        try:
            information = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            information.BasicLimitInformation.LimitFlags = 0x00002000
            if not kernel32.SetInformationJobObject(
                handle,
                9,
                ctypes.byref(information),
                ctypes.sizeof(information),
            ):
                raise OSError(
                    ctypes.get_last_error(), "SetInformationJobObject failed"
                )
            process_handle = wintypes.HANDLE(int(process._handle))  # type: ignore[attr-defined]
            if not kernel32.AssignProcessToJobObject(handle, process_handle):
                raise OSError(
                    ctypes.get_last_error(), "AssignProcessToJobObject failed"
                )
        except BaseException:
            kernel32.CloseHandle(handle)
            self._handle = None
            raise

    def terminate(self) -> None:
        if self._handle is not None:
            self._kernel32.TerminateJobObject(self._handle, 1)

    def close(self) -> None:
        if self._handle is not None:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


class _ProcessTree:
    def __init__(
        self,
        process: subprocess.Popen[bytes],
        *,
        isolate_process_tree: bool,
    ) -> None:
        self.process = process
        self.isolate_process_tree = isolate_process_tree
        self.windows_job = (
            _WindowsJob(process)
            if isolate_process_tree and os.name == "nt"
            else None
        )

    def terminate(self) -> None:
        if not self.isolate_process_tree:
            try:
                self.process.kill()
            except (OSError, ProcessLookupError):
                pass
            return
        if self.windows_job is not None:
            self.windows_job.terminate()
            return
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def close(self) -> None:
        self.terminate()
        if self.windows_job is not None:
            self.windows_job.close()


_WINDOWS_JOB_LAUNCHER = """
import json
import subprocess
import sys

payload = json.load(sys.stdin)
process = subprocess.Popen(
    payload["command"],
    stdin=subprocess.DEVNULL,
    creationflags=payload["creationflags"],
)
raise SystemExit(process.wait())
"""


def _validate_arguments(
    command: Sequence[str],
    timeout_seconds: float | int,
    max_output_bytes: int,
) -> None:
    if (
        isinstance(command, (str, bytes))
        or not command
        or any(type(item) is not str or not item for item in command)
    ):
        raise TypeError("command must be a nonempty sequence of nonempty strings")
    if (
        type(timeout_seconds) not in (int, float)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be finite and positive")
    if type(max_output_bytes) is not int or max_output_bytes < 1:
        raise ValueError("max_output_bytes must be a positive integer")


def run_bounded_process(
    command: Sequence[str],
    *,
    timeout_seconds: float | int,
    max_output_bytes: int,
    cancel: Callable[[], bool] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    creationflags: int = 0,
    isolate_process_tree: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    """Run a byte-capped child within one absolute wall-clock deadline.

    The default creates and owns a process-tree boundary. Set
    ``isolate_process_tree=False`` only for a controlled leaf process which
    must remain inside an already-established outer boundary.
    """

    _validate_arguments(command, timeout_seconds, max_output_bytes)
    if type(isolate_process_tree) is not bool:
        raise TypeError("isolate_process_tree must be Boolean")
    started = time.monotonic()
    deadline = started + float(timeout_seconds)
    cleanup_reserve = min(
        0.25,
        max(0.005, float(timeout_seconds) * 0.1),
        float(timeout_seconds) * 0.5,
    )
    execution_deadline = deadline - cleanup_reserve
    platform_flags = creationflags
    if os.name == "nt" and isolate_process_tree:
        platform_flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        launch_command = [sys.executable, "-I", "-c", _WINDOWS_JOB_LAUNCHER]
        child_stdin = subprocess.PIPE
    else:
        launch_command = list(command)
        child_stdin = subprocess.DEVNULL
    try:
        process = subprocess.Popen(
            launch_command,
            stdin=child_stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(cwd) if cwd is not None else None,
            env=None if env is None else dict(env),
            creationflags=platform_flags,
            start_new_session=isolate_process_tree and os.name != "nt",
        )
    except (OSError, ValueError) as exc:
        raise BoundedProcessLaunchError(f"could not launch child process: {exc}") from exc

    try:
        tree = _ProcessTree(
            process,
            isolate_process_tree=isolate_process_tree,
        )
    except OSError as exc:
        process.kill()
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
        raise BoundedProcessLaunchError(
            f"could not establish child process-tree containment: {exc}"
        ) from exc

    if os.name == "nt" and isolate_process_tree:
        try:
            if process.stdin is None:
                raise OSError("contained launcher stdin is unavailable")
            payload = json.dumps(
                {
                    "command": list(command),
                    "creationflags": creationflags,
                },
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("ascii")
            process.stdin.write(payload)
            process.stdin.close()
        except (OSError, ValueError) as exc:
            tree.close()
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
            raise BoundedProcessLaunchError(
                f"could not start the contained child process: {exc}"
            ) from exc

    chunks: queue.Queue[tuple[str, bytes | None]] = queue.Queue(maxsize=8)
    stop_readers = threading.Event()

    def emit(item: tuple[str, bytes | None]) -> None:
        while not stop_readers.is_set():
            try:
                chunks.put(item, timeout=0.02)
                return
            except queue.Full:
                continue

    def read_stream(name: str, stream: object) -> None:
        try:
            while True:
                try:
                    data = stream.read(8_192)  # type: ignore[attr-defined]
                except (OSError, ValueError):
                    break
                if not data:
                    break
                emit((name, data))
        finally:
            emit((name, None))

    readers = [
        threading.Thread(
            target=read_stream,
            args=("stdout", process.stdout),
            daemon=True,
        ),
        threading.Thread(
            target=read_stream,
            args=("stderr", process.stderr),
            daemon=True,
        ),
    ]
    for reader in readers:
        reader.start()

    output = {"stdout": bytearray(), "stderr": bytearray()}
    finished_streams = 0
    failure: type[BoundedProcessError] | None = None
    failure_message = ""
    cleanup_deadline = deadline
    contained = False
    try:
        while finished_streams < 2:
            now = time.monotonic()
            if failure is None and not contained:
                if cancel is not None and cancel():
                    failure = BoundedProcessCancelled
                    failure_message = "bounded child process cancelled"
                    cleanup_deadline = min(deadline, now + cleanup_reserve)
                    tree.terminate()
                    contained = True
                elif now >= execution_deadline and process.poll() is None:
                    failure = BoundedProcessTimeout
                    failure_message = (
                        f"bounded child process timed out after {timeout_seconds:g}s"
                    )
                    tree.terminate()
                    contained = True
                elif process.poll() is not None:
                    tree.terminate()
                    contained = True

            if now >= cleanup_deadline:
                break
            try:
                name, data = chunks.get(
                    timeout=max(0.001, min(0.02, cleanup_deadline - now))
                )
            except queue.Empty:
                continue
            if data is None:
                finished_streams += 1
                continue

            captured = len(output["stdout"]) + len(output["stderr"])
            available = max_output_bytes - captured
            if len(data) > available:
                if available > 0:
                    output[name].extend(data[:available])
                if failure is None:
                    failure = BoundedProcessOutputLimit
                    failure_message = "bounded child output exceeded its byte budget"
                    cleanup_deadline = min(deadline, time.monotonic() + cleanup_reserve)
                    tree.terminate()
                    contained = True
            else:
                output[name].extend(data)

        if finished_streams < 2 and failure is None:
            failure = BoundedProcessDrainError
            failure_message = "bounded child streams did not close by the deadline"
        if process.poll() is None:
            if finished_streams == 2 and failure is None:
                try:
                    process.wait(
                        timeout=min(0.02, max(0.0, deadline - time.monotonic()))
                    )
                except subprocess.TimeoutExpired:
                    tree.terminate()
            else:
                tree.terminate()
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            if failure is None:
                failure = BoundedProcessDrainError
                failure_message = "bounded child process did not exit by the deadline"
    finally:
        stop_readers.set()
        tree.close()
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        join_deadline = deadline
        for reader in readers:
            reader.join(timeout=max(0.0, join_deadline - time.monotonic()))

    stdout = bytes(output["stdout"])
    stderr = bytes(output["stderr"])
    if failure is not None:
        raise failure(failure_message, stdout=stdout, stderr=stderr)
    returncode = process.poll()
    if returncode is None:
        raise BoundedProcessDrainError(
            "bounded child process has no terminal return code",
            stdout=stdout,
            stderr=stderr,
        )
    return subprocess.CompletedProcess(list(command), returncode, stdout, stderr)


__all__ = [
    "BoundedProcessCancelled",
    "BoundedProcessDrainError",
    "BoundedProcessError",
    "BoundedProcessLaunchError",
    "BoundedProcessOutputLimit",
    "BoundedProcessTimeout",
    "run_bounded_process",
]
