"""Named deterministic random-number-generator streams."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RNGStreams:
    """Independent generators derived from one master seed."""

    spot: np.random.Generator
    positions: np.random.Generator
    shot_noise: np.random.Generator
    read_noise: np.random.Generator
    fixed_pattern: np.random.Generator
    trajectory: np.random.Generator


def make_rng_streams(master_seed: int) -> RNGStreams:
    """Create stable named RNG streams from ``master_seed``.

    ``SeedSequence.spawn`` prevents adding draws to one experiment from
    silently changing the random values used by unrelated experiments.
    """

    seed_sequence = np.random.SeedSequence(master_seed)
    children = seed_sequence.spawn(6)
    generators = [np.random.default_rng(child) for child in children]
    return RNGStreams(
        spot=generators[0],
        positions=generators[1],
        shot_noise=generators[2],
        read_noise=generators[3],
        fixed_pattern=generators[4],
        trajectory=generators[5],
    )
