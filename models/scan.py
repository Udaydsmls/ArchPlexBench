import math

import torch


def parallel_scan(A, X):
    """Hillis-Steele parallel prefix scan for the linear recurrence
    h_t = A[:, t] * h_{t-1} + X[:, t], h_{-1} = 0.

    A and X share shape (B, T, ...) with the time dimension at index 1.
    Runs in O(log T) sequential steps with full parallelism within each step,
    replacing a T-step Python loop that would otherwise dispatch T CUDA kernels.
    """
    T = A.shape[1]
    if T <= 1:
        return X

    log_T = math.ceil(math.log2(T))
    for d in range(log_T):
        step = 2 ** d
        if step >= T:
            break
        pad_shape = list(A.shape)
        pad_shape[1] = step
        A_prev = torch.cat([A.new_ones(pad_shape), A[:, :-step]], dim=1)
        X_prev = torch.cat([X.new_zeros(pad_shape), X[:, :-step]], dim=1)
        X = A * X_prev + X
        A = A * A_prev
    return X
