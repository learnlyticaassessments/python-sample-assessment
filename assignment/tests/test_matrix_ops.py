import numpy as np
import pytest
from src.matrix_ops import matrix_multiply, matrix_inverse

def test_matrix_multiply_square():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    expected = np.array([[19, 22], [43, 50]])
    assert np.allclose(matrix_multiply(A, B), expected)

def test_matrix_multiply_rectangular():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[7, 8], [9, 10], [11, 12]])
    expected = np.array([[58, 64], [139, 154]])
    assert np.allclose(matrix_multiply(A, B), expected)

def test_matrix_multiply_identity():
    A = np.array([[1, 2], [3, 4]])
    I = np.eye(2)
    assert np.allclose(matrix_multiply(A, I), A)

def test_matrix_inverse_square():
    A = np.array([[4, 7], [2, 6]])
    expected = np.linalg.inv(A)
    assert np.allclose(matrix_inverse(A), expected)

def test_matrix_inverse_identity():
    I = np.eye(3)
    assert np.allclose(matrix_inverse(I), I)

def test_matrix_inverse_inverse():
    A = np.array([[1, 2], [3, 5]])
    inv = matrix_inverse(A)
    assert np.allclose(matrix_multiply(A, inv), np.eye(2))