import torch
import numpy as np

from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
from helper import kl_divergence

"""
    runnable in /Thesis-ML/src/ with:
        pytest tests/model/sample.py
"""
shape = st.shared(
    st.tuples(
        st.integers(min_value=1, max_value=512),
        st.integers(min_value=1, max_value=512)
    )
)

@given(arrays(dtype=np.float64, shape=shape, elements=st.integers(max_value=7e2, min_value=-7e2)), 
        arrays(dtype=np.float64, shape=shape, elements=st.integers(max_value=1e3, min_value=-1e3)))
def test_kld_properties(logvar, mu):    
    logvar, mu = map(torch.from_numpy, (logvar, mu))
    result     = kl_divergence(logvar, mu)

    assert logvar.shape == mu.shape, "logvar and mu must have the same shape"
    assert result >= 0, "KLD should always be positive"
    assert isinstance(result, torch.Tensor), "Output should be a torch.Tensor"
    assert result.shape == (), "Output should be a scalar (0-dim tensor)"
    assert torch.isfinite(result), "Output should be a finite number"