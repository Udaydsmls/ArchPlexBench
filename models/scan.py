import math

import torch
import torch.nn.functional as F


@torch.compile
def parallel_scan(A, X):
    """Hillis-Steele parallel prefix scan for the linear recurrence
    h_t = A[:, t] * h_{t-1} + X[:, t], h_{-1} = 0.

    A and X share shape (B, T, ...) with the time dimension at index 1.
    Runs in O(log T) sequential steps with full parallelism within each step.
    Wrapped in torch.compile so TorchInductor can fuse the per-iteration
    pad+mul+mul+add into a single kernel, cutting kernel-launch overhead and
    memory bandwidth — without this the elementwise chain dominates wall time.
    """
    T = A.shape[1]
    if T <= 1:
        return X

    log_T = math.ceil(math.log2(T))
    pad_zero = (0, 0) * (A.ndim - 2)
    for d in range(log_T):
        step = 2 ** d
        if step >= T:
            break
        pad_spec = pad_zero + (step, 0)
        A_prev = F.pad(A[:, :-step], pad_spec, value=1.0)
        X_prev = F.pad(X[:, :-step], pad_spec, value=0.0)
        X = A * X_prev + X
        A = A * A_prev
    return X
