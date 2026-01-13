import numpy as np


def get_transition(states, n_states):
    """
    Count only true state changes.
    Ignore NaNs.
    Do NOT count self-transitions.
    """
    T = np.zeros((n_states, n_states), dtype=int)
    states = np.asarray(states)

    for s0, s1 in zip(states[:-1], states[1:]):
        if np.isnan(s0) or np.isnan(s1):
            continue
        if s0 == s1:
            continue
        T[int(s0), int(s1)] += 1

    return T