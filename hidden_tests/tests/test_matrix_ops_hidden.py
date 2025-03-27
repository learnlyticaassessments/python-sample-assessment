import numpy as np
from assignment.src.matrix_ops import matrix_multiply, matrix_inverse

def test_matrix_multiply_large():
    A = np.ones((50, 50))
    B = np.eye(50)
    assert np.allclose(matrix_multiply(A, B), A)

def test_matrix_inverse_edge():
    A = np.array([[0.1, 0.2], [0.3, 0.4]])
    expected = np.linalg.inv(A)
    assert np.allclose(matrix_inverse(A), expected)
