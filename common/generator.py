import numba
import numpy as np


@numba.jit(nopython=True)
def sample_distribution(choices, probabilities, size=1):
    cumulative_probabilities = np.cumsum(probabilities)
    selected_indices = np.searchsorted(cumulative_probabilities, np.random.random(size), side='right')
    selected_indices = np.clip(selected_indices, 0, len(probabilities) - 1)

    return choices[selected_indices]


@numba.jit(nopython=True)
def sample_distribution_once(probabilities):
    cumulative_probabilities = np.cumsum(probabilities)
    selected_index = np.searchsorted(cumulative_probabilities, np.random.random())

    return selected_index
