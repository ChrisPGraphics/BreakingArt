import numba
import numpy as np


@numba.jit(nopython=True)
def dilate(array: np.ndarray, iterations: int = 1, mask: np.ndarray = None) -> np.ndarray:
    result = np.zeros_like(array)
    height, width = array.shape

    for _ in range(iterations):
        for y, x in zip(*np.where(array)):
            result[y, x] = 1

            if y - 1 >= 0:
                result[y - 1, x] = 1
            if x - 1 >= 0:
                result[y, x - 1] = 1
            if y + 1 < height:
                result[y + 1, x] = 1
            if x + 1 < width:
                result[y, x + 1] = 1

    if mask is None:
        return result

    return np.logical_and(result, mask)


@numba.jit(nopython=True)
def count_true(binary_mask: np.ndarray) -> int:
    count = 0

    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i, j]:
                count += 1

    return count
