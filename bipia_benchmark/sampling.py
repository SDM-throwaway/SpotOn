"""
Sampling utilities for the BIPIA defense benchmark.

Extracted from benchmark_defense.py so sampling logic can be tested
independently of the full benchmark class (which requires BIPIA, aiohttp, etc).
"""

import json
import os
import random
from datetime import datetime


def _now() -> datetime:
    """Return current datetime. Exists as a module-level function so tests can monkeypatch it."""
    return datetime.now()


def select_samples(samples: list[dict], seed: int, limit: int) -> list[dict]:
    """
    Shuffle samples deterministically by seed, take the first `limit` entries,
    and assign a unique 'id' to each selected sample.

    Args:
        samples: Full list of sample dicts from the BIPIA dataset.
        seed:    Random seed that controls which subset is selected.
                 The same seed always produces the same subset.
        limit:   Maximum number of samples to return.
                 0 means no cap (return all samples after shuffling).

    Returns:
        A new list of sample dicts (copies of the originals) each with an
        'id' key set to its position index within the selected subset.
        The originals are not mutated.
    """
    pool = [dict(s) for s in samples]  # shallow copy - don't mutate caller's list
    random.seed(seed)
    random.shuffle(pool)
    selected = pool[:limit] if limit > 0 else pool
    for idx, s in enumerate(selected):
        s["id"] = idx
    return selected


def save_samples(
    samples: list[dict],
    dataset_name: str,
    base_dir: str = "experiment_output",
) -> tuple[list[dict], str]:
    """
    Save samples to a timestamped directory and return (samples, output_dir).

    Args:
        samples:      List of sample dicts to persist.
        dataset_name: Used as prefix for the output directory name.
        base_dir:     Parent directory under which the timestamped folder is created.
                      Created automatically if it does not exist.

    Returns:
        A tuple of (samples, output_dir) where output_dir is the absolute path
        to the newly created directory.
    """
    timestamp = _now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, f"{dataset_name}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "samples.json"), "w") as fh:
        json.dump(samples, fh, indent=2)
    return samples, out_dir
