"""
Transformer Simulator
Version 1

Creates transformer loading from
Demand and Generation.
"""

import random

from config import SEED

random.seed(SEED)


def next_transformer(demand, generation):
    """
    Parameters
    ----------
    demand : float
    generation : float

    Returns
    -------
    dict
    """

    reserve = generation - demand

    # Base loading
    load = 70

    # Demand effect
    load += (demand - 170) * 0.30

    # Reserve reduces loading
    load -= reserve * 0.20

    # Random fluctuation
    load += random.uniform(-3, 3)

    # Clamp
    load = max(55, min(110, load))

    return {
        "transformer_load": round(load, 2),
        "capacity_utilization": round(load, 2)
    }