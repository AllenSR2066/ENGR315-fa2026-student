"""
ENGR 315 - Numerical Methods: Linear Systems

Complete the four functions below using the methods from lecture.
Your functions will be tested with different linear systems, not just the
5-by-5 example at the bottom. Do not hard-code the sample answers.

Submit this .py file to Gradescope. Keep function names and return values unchanged.
You may assume all test matrices are square, nonsingular, and suitable for
Jacobi and Conjugate Gradient (symmetric positive definite, nonzero diagonal).
"""

import numpy as np
from scipy.sparse.linalg import cg


def direct_solve(_A, _b):
    """
    Solve A @ x = b using NumPy's direct linear-system solver.
    :param _A: n-by-n array-like coefficient matrix
    :param _b: length-n array-like right-hand side
    :return: length-n numpy array of solution values
    """
    # Hint: use np.linalg.solve() as in lecture.
    currents = None

    # YOUR CODE HERE #

    return currents


def verify_solution(_A, _b, _x):
    """
    Verify an estimated solution of A @ x = b.
    :return: (calculated_b, matches), where calculated_b is a numpy array
             equal to A @ x and matches is a Python bool from np.allclose().
    """
    # Hint: use np.dot() and np.allclose() as in lecture.
    calculated_b = None
    matches = None

    # YOUR CODE HERE #

    return calculated_b, matches


def jacobi_solve(_A, _b, _tolerance=0.001, _max_iterations=100):
    """
    Solve A @ x = b using Jacobi iteration.
    Start from an all-zero estimate. Use only previous-iteration values
    to calculate every new estimate.
    Stop when max(abs(x_new - x_old)) < _tolerance, or when
    _max_iterations updates have been completed.

    :return: (x, iterations, converged), where x is a numpy array,
             iterations is the number of updates performed (int),
             and converged is a Python bool.
    """
    # Hint: use np.diag(), np.dot(), a for loop, and an if statement.
    x = None
    iterations = None
    converged = None

    # YOUR CODE HERE #

    return x, iterations, converged


def scipy_iterative_solve(_A, _b, _rtol=1e-5):
    """
    Solve A @ x = b using scipy.sparse.linalg.cg().
    :return: (x, info), where x is a numpy array and info is the
             integer status from cg() (0 means successful convergence).
    """
    # Hint: cg(A, b, rtol=_rtol) returns two values.
    x = None
    info = None

    # YOUR CODE HERE #

    return x, info


if __name__ == "__main__":
    # Assignment example: a 5-by-5 linear system.
    # The first four functions above are graded using other valid systems too.
    A = np.array([[10, -2,  0,  0,  0],
                  [-2, 12, -3,  0,  0],
                  [ 0, -3, 14, -2,  0],
                  [ 0,  0, -2, 11, -2],
                  [ 0,  0,  0, -2,  9]], dtype=float)
    b = np.array([6, 13, 28, 28, 37], dtype=float)

    print("Direct. Expected: [1. 2. 3. 4. 5.]. Result:", direct_solve(A, b))
    print("Verify exact. Expected: (array [6. 13. 28. 28. 37.], True). Result:",
          verify_solution(A, b, np.array([1., 2., 3., 4., 5.])))
    print("Verify wrong. Expected match: False. Result:",
          verify_solution(A, b, np.zeros(5)))
    print("Jacobi. Expected: approximately [1. 2. 3. 4. 5.], 9, True. Result:",
          jacobi_solve(A, b))
    print("SciPy CG. Expected: approximately [1. 2. 3. 4. 5.], info=0. Result:",
          scipy_iterative_solve(A, b))
