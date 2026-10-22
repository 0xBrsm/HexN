# SPDX-License-Identifier: GPL-3.0-only
"""Resolve training hardware choices before a new manifest is frozen."""
from hexset.bench.throughput import default_workers


def resolve(args) -> None:
    """Freeze auto device/workers; preserve every explicit resource choice."""
    if getattr(args, "device", None) == "auto":
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(args, "collect_workers") and args.collect_workers is None:
        # Reserve a process for the learner and one for batch assembly.
        count = max(1, default_workers() - 2)
        count = min(count, args.lanes)
        if getattr(args, "collect_mode", "cohort") == "cohort":
            count = min(count, args.games_per_iteration)
        count = max(1, count)
        if getattr(args, "corpus", None):
            count = 1  # A fixed corpus performs no collection.
        if getattr(args, "pair_boards", False):
            # Paired leagues need equal worker quotas to keep board mates
            # within the same iteration.
            while args.games_per_iteration % count:
                count -= 1
        args.collect_workers = count
