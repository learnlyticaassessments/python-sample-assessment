import numpy as np
from assignment.src.stats_ops import mean, std_dev

def test_mean_large_dataset():
    data = np.arange(1, 1001)
    assert np.isclose(mean(data), np.mean(data))

def test_std_dev_precision():
    data = [1.000001, 1.000002, 1.000003]
    assert np.isclose(std_dev(data), np.std(data), rtol=1e-7)
