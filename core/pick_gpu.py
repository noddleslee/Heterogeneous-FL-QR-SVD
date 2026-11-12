#!/usr/bin/env python3
"""
Query SLURM sinfo and print the GPU type (A100 or A40) with the most
free GPUs on the alvis partition.

Prints the winner to stdout; free-GPU counts to stderr.
Falls back to A40 if sinfo fails or returns no data.

Usage:
    GPU=$(python3 pick_gpu.py)
"""
import re
import subprocess
import sys

PARTITION  = "alvis"
CANDIDATES = ["A100", "A40"]   # tie-broken in this order (A100 preferred)


def count_free(candidates=CANDIDATES):
    """Return {gpu_type: free_gpu_count} for each candidate."""
    try:
        raw = subprocess.check_output(
            [
                "sinfo", "-p", PARTITION, "-N",
                "--Format=nodelist:15,gres:55,gresused:55,statelong:10",
                "-h",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        print(f"[pick_gpu] sinfo failed: {exc}", file=sys.stderr)
        return {t: 0 for t in candidates}

    free = {t: 0 for t in candidates}

    for line in raw.splitlines():
        # Skip nodes that are not usable (only idle / mixed are schedulable)
        if not re.search(r"\b(idle|mixed)\b", line, re.IGNORECASE):
            continue

        for gpu_type in candidates:
            # Total GPUs on this node �� gres column has (S:...) notation
            m_total = re.search(
                rf"gpu:{gpu_type}:(\d+)\(S:", line, re.IGNORECASE
            )
            # Used GPUs �� gresused column has (IDX:...) notation
            m_used = re.search(
                rf"gpu:{gpu_type}:(\d+)\(IDX:", line, re.IGNORECASE
            )
            if m_total:
                total = int(m_total.group(1))
                used  = int(m_used.group(1)) if m_used else 0
                free[gpu_type] += max(0, total - used)

    return free


def pick(free, candidates=CANDIDATES):
    """Return the candidate with the most free GPUs; prefer first on tie."""
    best = candidates[0]
    for t in candidates[1:]:
        if free[t] > free[best]:
            best = t
    return best


if __name__ == "__main__":
    free = count_free()
    print(f"[pick_gpu] Free GPUs on {PARTITION}: " +
          "  ".join(f"{t}={free[t]}" for t in CANDIDATES),
          file=sys.stderr)

    if not any(free.values()):
        print("[pick_gpu] No free GPUs detected �� falling back to A40",
              file=sys.stderr)
        print("A40")
        sys.exit(0)

    winner = pick(free)
    print(f"[pick_gpu] Selected: {winner}", file=sys.stderr)
    print(winner)
