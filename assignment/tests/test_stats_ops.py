import numpy as np
from src.stats_ops import mean, std_dev

def test_mean_simple():
    data = [1, 2, 3, 4, 5]
    assert mean(data) == 3.0

def test_mean_negative():
    data = [-5, -10, -15]
    assert mean(data) == -10.0

def test_mean_floats():
    data = [1.5, 2.5, 3.0]
    assert mean(data) == 2.3333333333333335

def test_std_dev_simple():
    data = [1, 2, 3, 4, 5]
    assert np.isclose(std_dev(data), np.std(data))

def test_std_dev_zero():
    data = [5, 5, 5]
    assert std_dev(data) == 0.0

def test_std_dev_negative():
    data = [-1, -2, -3]
    assert np.isclose(std_dev(data), np.std(data))