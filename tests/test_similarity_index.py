import numpy as np
from neuro_py.ensemble import similarity_index

def test_similarityindex():
    patterns = [np.random.normal(0, .1, 100) for _ in range(10)]

    si, combos, pvalues = similarity_index(patterns)
    assert all(i <= 1 for i in si)
    assert combos[0,0] == 0
    assert all(i <= 1 for i in pvalues)
    assert combos.shape[0] == pvalues.shape[0] == si.shape[0]