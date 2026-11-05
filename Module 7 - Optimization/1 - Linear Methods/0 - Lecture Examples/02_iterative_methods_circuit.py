"""
Numerical Methods: Jacobi and SciPy CG for Circuit 2.
"""
import numpy as np
from scipy.sparse.linalg import cg

# Mesh equations from Slide 14:
#  6*I1 - I2            = 9
# -I1 + 7*I2 - 2*I3     = 17
#       -2*I2 + 8*I3    = 2
A = np.array([[6, -1, 0],
              [-1, 7, -2],
              [0, -2, 8]], dtype=float)
b = np.array([9, 17, 2], dtype=float)

# EXAMPLE 1: Jacobi method — manual implementation using NumPy and a for loop
I = np.zeros(3)  # Initial guess I1 = I2 = I3 = 0
tolerance = 0.001  # Stop when max change between iterations is < 0.001 A
max_iterations = 100
D = np.diag(A)
R = A - np.diag(D)

print('JACOBI ITERATION')
print('Iteration |       I1       I2       I3 | Max change (A)')
print(f'{0:9d} | {I[0]:8.4f} {I[1]:8.4f} {I[2]:8.4f} |       ---')
converged = False
for k in range(1, max_iterations + 1):
    I_new = (b - np.dot(R, I)) / D
    change = np.max(np.abs(I_new - I))
    print(f'{k:9d} | {I_new[0]:8.4f} {I_new[1]:8.4f} {I_new[2]:8.4f} | {change:10.6f}')
    I = I_new
    if change < tolerance:
        converged = True
        break

print(f'Jacobi converged: {converged}')
print(f'Iterations completed: {k}')
print('Jacobi currents (A):', np.round(I, 4))
print('Jacobi residual (A @ I - b):', A @ I - b)

# PART 2: SciPy built-in iterative solver, Conjugate Gradient.
# This A is symmetric positive definite, as required for cg().
print('\nSCIPY CONJUGATE GRADIENT')
cg_currents, info = cg(A, b, rtol=1e-5)
print('CG currents (A):', np.round(cg_currents, 4))
print('CG solver status (0 = converged):', info)

# Reference: Direct solver for the SAME Circuit 2
reference_currents = np.linalg.solve(A, b)
print('\nREFERENCE DIRECT SOLUTION (CIRCUIT 2)')
print('Direct currents (A):', np.round(reference_currents, 4))
print('Is the Jacobi solution close to the direct solution at (atol=0.001)?:',
      np.allclose(I, reference_currents, atol=0.001))
print('CG close to direct:', np.allclose(cg_currents, reference_currents))
