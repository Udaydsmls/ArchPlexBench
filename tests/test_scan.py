"""Property tests for the Hillis-Steele parallel prefix scan.

Run with:
    pytest tests/test_scan.py -v
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from models.scan import parallel_scan


def _sequential_scan(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Naive O(T) loop: h_t = A_t * h_{t-1} + X_t, with h_{-1} = 0."""
    B, T = A.shape[:2]
    rest = A.shape[2:]
    h = torch.zeros(B, *rest, dtype=A.dtype)
    out = torch.zeros_like(X)
    for t in range(T):
        h = A[:, t] * h + X[:, t]
        out[:, t] = h
    return out


@given(
    batch=st.integers(1, 8),
    seq_len=st.integers(1, 256),
    d_model=st.integers(1, 64),
    d_state=st.integers(1, 16),
)
@settings(max_examples=200, deadline=None)
def test_matches_sequential(batch, seq_len, d_model, d_state):
    torch.manual_seed(0)
    # keep A in (0, 1) so the recurrence stays numerically stable
    A = torch.rand(batch, seq_len, d_model, d_state) * 0.99 + 1e-4
    X = torch.randn(batch, seq_len, d_model, d_state)

    expected = _sequential_scan(A, X)
    actual = parallel_scan(A, X)

    assert actual.shape == expected.shape
    max_diff = (actual - expected).abs().max().item()
    assert max_diff < 1e-4, (
        f"max diff {max_diff:.2e} at "
        f"batch={batch} seq_len={seq_len} d_model={d_model} d_state={d_state}"
    )


@given(seq_len=st.integers(1, 256))
@settings(max_examples=50, deadline=None)
def test_zero_input_gives_zero_output(seq_len):
    A = torch.rand(1, seq_len, 4, 4) * 0.9 + 0.05
    X = torch.zeros(1, seq_len, 4, 4)
    assert parallel_scan(A, X).abs().max().item() == 0.0


@given(seq_len=st.integers(1, 256))
@settings(max_examples=50, deadline=None)
def test_unit_A_equals_cumsum(seq_len):
    # when A=1 the recurrence is just a cumulative sum
    A = torch.ones(1, seq_len, 4, 4)
    X = torch.randn(1, seq_len, 4, 4)
    expected = X.cumsum(dim=1)
    max_diff = (parallel_scan(A, X) - expected).abs().max().item()
    assert max_diff < 1e-4, f"cumsum diff {max_diff:.2e} at seq_len={seq_len}"
