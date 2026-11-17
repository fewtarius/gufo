#!/usr/bin/env python3
"""Capture gufo's AQL queue state at the moment a request stops progressing.

    sudo python3 tools/prof/queue_watch.py

Three failures on 2026-11-16 left the compute queue making no progress: two
aborted with `Illegal opcode in command stream` and one hung silently with the
GPU idle and a thread parked in hipStreamSynchronize. The cores showed a
byte-perfect, structurally valid ring after the fact, so the packet that the
command processor choked on was never captured as it actually saw it, and the
silent hang left no dump at all.

This watches for the stall while the process is still alive and reads the ring
out of /proc/<pid>/mem. Two samples are taken a few seconds apart, so a frozen
queue is distinguishable from a busy one. Everything is written under
/tmp/gufo-wedge-<timestamp>/ and nothing is sent anywhere.

Detection is deliberately dumb and independent of gufo's logging tiers: a chat
request that was accepted and has not completed, plus a GPU that reports zero
busy for `--idle-seconds`. Readings come from /proc and sysfs only, so the
watch costs the server nothing while it waits.
"""
import argparse
import collections
import glob
import os
import re
import shutil
import struct
import subprocess
import sys
import time

AQL_SLOT = 64
# HSA packet types: 1 invalid (queue terminator), 2 kernel dispatch,
# 3 barrier-and, 5 barrier-or.
PACKET_INVALID = 1
PACKET_DISPATCH = 2
PACKET_BARRIER_AND = 3
PACKET_NAMES = {1: "invalid", 2: "dispatch", 3: "barrier-and", 5: "barrier-or"}


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def gufo_pid():
    """The serving process, identified by its command line."""
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as handle:
                cmdline = handle.read().replace(b"\0", b" ").decode()
        except OSError:
            continue
        if "gufo serve" in cmdline and "--model" in cmdline:
            return int(entry)
    return None


def gpu_busy_percent():
    """Busiest card, or None when the attribute is unavailable."""
    values = []
    for path in glob.glob("/sys/class/drm/card*/device/gpu_busy_percent"):
        try:
            with open(path) as handle:
                values.append(int(handle.read().strip()))
        except (OSError, ValueError):
            continue
    return max(values) if values else None


def device_used_mib():
    """PCIe-visible VRAM use in MiB; a GTT byte count on this APU."""
    for path in ("/sys/class/drm/card0/device/mem_info_gtt_used",
                 "/sys/class/drm/card1/device/mem_info_gtt_used"):
        try:
            with open(path) as handle:
                return int(handle.read().strip()) // (1024 * 1024)
        except (OSError, ValueError):
            continue
    return None


def tail_lines(path, count):
    try:
        with open(path, errors="replace") as handle:
            return collections.deque(handle, maxlen=count)
    except OSError:
        return []


def outstanding_requests(log_path):
    """Chat requests accepted but not completed, read from the server log.

    The log is append-only within one run, so counting is enough; request ids
    are not reused inside a process.
    """
    received = {}
    completed = set()
    try:
        with open(log_path, errors="replace") as handle:
            for line in handle:
                match = re.search(r"request=(\S+) event=(received|completed)", line)
                if not match:
                    continue
                request_id, event = match.groups()
                if event == "received":
                    received[request_id] = line.rstrip()
                else:
                    completed.add(request_id)
    except OSError:
        return {}
    return {k: v for k, v in received.items() if k not in completed}


def ring_candidates(pid):
    """Readable 1 MiB private mappings, which is what a ROCr AQL ring is.

    On this host every observed AQL ring is exactly 1 MiB and anonymous; the
    kernarg buffers beside them are 4 MiB and the code objects are not a round
    size, so the size filter separates them.
    """
    candidates = []
    with open(f"/proc/{pid}/maps") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 5:
                continue
            start, end = (int(x, 16) for x in fields[0].split("-"))
            perms = fields[1]
            if not perms.startswith("rw"):
                continue
            if end - start != 1024 * 1024:
                continue
            # ROCr backs a ring with a shared anonymous mapping, which the
            # kernel reports as /dev/zero; a coredump lists the same mappings
            # as "/dev/zero (deleted)". Bracketed entries are kernel objects.
            # Anything else at this size is a real file and not a ring.
            path = fields[5] if len(fields) > 5 else ""
            if path and not path.startswith("[") and not path.startswith("/dev/zero"):
                continue
            candidates.append((start, end))
    return candidates


def read_memory(pid, start, length):
    with open(f"/proc/{pid}/mem", "rb", buffering=0) as handle:
        handle.seek(start)
        return handle.read(length)


def decode_packet(data, offset):
    header, setup = struct.unpack_from("<HH", data, offset)
    workgroup = struct.unpack_from("<3H", data, offset + 4)
    grid = struct.unpack_from("<3I", data, offset + 12)
    private_bytes, group_bytes = struct.unpack_from("<II", data, offset + 24)
    kernel_object, kernarg, _, completion = struct.unpack_from("<4Q", data, offset + 32)
    return {
        "header": header,
        "type": header & 0xFF,
        "setup": setup,
        "workgroup": workgroup,
        "grid": grid,
        "private_bytes": private_bytes,
        "group_bytes": group_bytes,
        "kernel_object": kernel_object,
        "kernarg": kernarg,
        "completion_signal": completion,
    }


def describe_ring(pid, start, length):
    """Decode a ring into its pending window and its valid-packet run."""
    data = read_memory(pid, start, length)
    slots = len(data) // AQL_SLOT
    headers = [struct.unpack_from("<H", data, i * AQL_SLOT)[0] for i in range(slots)]
    histogram = collections.Counter(h & 0xFF for h in headers)
    valid = [i for i, h in enumerate(headers) if (h & 0xFF) in (2, 3, 5)]
    runs = []
    if valid:
        first = previous = valid[0]
        for index in valid[1:]:
            if index == previous + 1:
                previous = index
                continue
            runs.append((first, previous))
            first = previous = index
        runs.append((first, previous))
    lines = [f"ring 0x{start:x} slots={slots} header-types={dict(histogram)}"]
    lines.append(f"  valid-packet runs: {[(a, b) for a, b in runs]}")
    for begin, end in runs:
        lines.append(f"  run {begin}..{end} ({end - begin + 1} packets)")
        for slot in (begin, min(begin + 1, end), end):
            packet = decode_packet(data, slot * AQL_SLOT)
            lines.append(
                "    slot {slot} type={tname} header=0x{h:04x} setup={s} "
                "grid={g} wg={w} priv={p} grp={gr} ko=0x{ko:x} karg=0x{ka:x} cs=0x{cs:x}".format(
                    slot=slot,
                    tname=PACKET_NAMES.get(packet["type"], packet["type"]),
                    h=packet["header"],
                    s=packet["setup"],
                    g=packet["grid"],
                    w=packet["workgroup"],
                    p=packet["private_bytes"],
                    gr=packet["group_bytes"],
                    ko=packet["kernel_object"],
                    ka=packet["kernarg"],
                    cs=packet["completion_signal"],
                )
            )
        # The first pending packet is the one the command processor was facing.
        head = decode_packet(data, begin * AQL_SLOT)
        lines.append(
            "    HEAD type={} header=0x{:04x}".format(
                PACKET_NAMES.get(head["type"], head["type"]), head["header"]
            )
        )
    return "\n".join(lines), runs


def capture(pid, out_dir, log_path):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.txt"), "w") as summary:

        def emit(text):
            log(text)
            summary.write(text + "\n")
            summary.flush()

        emit(f"captured {time.strftime('%Y-%m-%d %H:%M:%S')} pid={pid}")
        emit(f"gpu_busy_percent={gpu_busy_percent()}")
        emit(f"gtt_used_mib={device_used_mib()}")
        pending = outstanding_requests(log_path)
        emit(f"outstanding chat requests: {sorted(pending)}")
        for line in pending.values():
            emit(f"  {line[:400]}")
        emit("--- last log lines ---")
        for line in tail_lines(log_path, 8):
            emit(f"  {line[:300]}")

        emit("--- thread states ---")
        try:
            states = subprocess.run(
                ["ps", "-L", "-o", "tid,stat,pcpu,wchan:24", "-p", str(pid)],
                capture_output=True, text=True, timeout=30,
            ).stdout
            emit(states.strip())
        except (subprocess.SubprocessError, OSError) as error:
            emit(f"ps failed: {error}")

        emit("--- bindings ---")
        try:
            emit(subprocess.run(["ps", "-L", "-o", "tid,wchan:30", "-p", str(pid)],
                                capture_output=True, text=True,
                                timeout=30).stdout.strip())
        except (subprocess.SubprocessError, OSError) as error:
            emit(f"ps failed: {error}")

        rings = ring_candidates(pid)
        emit(f"--- ring candidates ({len(rings)}) ---")
        snapshots = []
        for start, end in rings:
            try:
                text, runs = describe_ring(pid, start, end - start)
            except OSError as error:
                emit(f"ring 0x{start:x}: unreadable ({error})")
                continue
            emit(text)
            snapshots.append((start, runs))
        # A second sample proves the queue is frozen, not merely busy.
        time.sleep(5)
        emit("--- second sample 5s later ---")
        for start, previous_runs in snapshots:
            try:
                _, runs = describe_ring(pid, start, 1024 * 1024)
            except OSError as error:
                emit(f"ring 0x{start:x}: unreadable ({error})")
                continue
            state = "FROZEN" if runs == previous_runs else "PROGRESSED"
            emit(f"ring 0x{start:x}: {state} runs={runs}")

        emit("--- kernel log ---")
        try:
            emit(subprocess.run(["dmesg"], capture_output=True, text=True,
                                timeout=30).stdout.strip().splitlines()[-15:])
        except (subprocess.SubprocessError, OSError) as error:
            emit(f"dmesg failed: {error}")
        emit("--- gdb backtrace ---")
        gdb = shutil.which("gdb")
        if gdb is None:
            emit("gdb not installed")
        else:
            try:
                result = subprocess.run(
                    [gdb, "-p", str(pid), "-batch", "-ex", "thread apply all bt 8"],
                    capture_output=True, text=True, timeout=300,
                )
                with open(os.path.join(out_dir, "backtrace.txt"), "w") as handle:
                    handle.write(result.stdout)
                emit(f"backtrace written ({len(result.stdout)} bytes)")
            except (subprocess.SubprocessError, OSError) as error:
                emit(f"gdb failed: {error}")
        emit(f"done: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default=os.path.expanduser(
        "~/.local/state/llm-scripts/run.log"))
    parser.add_argument("--idle-seconds", type=float, default=20.0,
                        help="GPU-idle seconds with a request outstanding")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()

    log("waiting for a request that stops progressing")
    idle_since = None
    history = collections.deque(maxlen=120)
    last_sample = 0.0
    while True:
        pid = gufo_pid()
        if pid is None:
            idle_since = None
            history.clear()
            time.sleep(arguments.poll_seconds)
            continue
        pending = outstanding_requests(arguments.log)
        busy = gpu_busy_percent()
        # Sample the rings continuously so a stall can be shown as a
        # transition: the last packets the queue accepted, and the packet it
        # was facing when it stopped retiring.
        now = time.monotonic()
        if now - last_sample >= 5.0:
            last_sample = now
            sample = {}
            for start, end in ring_candidates(pid):
                try:
                    _, runs = describe_ring(pid, start, end - start)
                except OSError:
                    continue
                sample[start] = runs
            history.append((time.strftime("%H:%M:%S"), bool(pending), sample))
        if pending and busy == 0:
            idle_since = idle_since or time.monotonic()
            idle = time.monotonic() - idle_since
            if idle >= arguments.idle_seconds:
                # An idle GPU during a long host-side phase (a multi-GB
                # snapshot restore, for instance) looks the same as a stall
                # from the outside. Only treat it as one once the queue has
                # stopped retiring packets, because attaching gdb stops the
                # process for as long as it runs.
                before = {}
                for start, end in ring_candidates(pid):
                    try:
                        _, runs = describe_ring(pid, start, end - start)
                    except OSError:
                        continue
                    before[start] = runs
                time.sleep(5)
                frozen = True
                for start, previous in before.items():
                    try:
                        _, runs = describe_ring(pid, start, 1024 * 1024)
                    except OSError:
                        continue
                    if runs != previous:
                        frozen = False
                        break
                if not frozen or not before:
                    log("gpu idle but the queue is still retiring; not a stall")
                    idle_since = None
                    continue
                out_dir = arguments.out or time.strftime("/tmp/gufo-wedge-%Y%m%d-%H%M%S")
                log(
                    f"stall: {sorted(pending)} GPU idle {idle:.0f}s, "
                    f"queue frozen at {[hex(k) for k in before]}"
                )
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "history.txt"), "w") as handle:
                    for stamp, had_pending, sample in history:
                        handle.write(
                            f"{stamp} pending={int(had_pending)} "
                            + " ".join(f"0x{k:x}={v}" for k, v in sample.items())
                            + "\n"
                        )
                log(f"history written ({len(history)} samples)")
                try:
                    capture(pid, out_dir, arguments.log)
                except OSError as error:
                    log(f"capture failed: {error}")
                    sys.exit(1)
                return 0
        else:
            if idle_since is not None:
                log("progress resumed, resetting")
            idle_since = None
        time.sleep(arguments.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
