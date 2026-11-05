"""
Numerical Methods: Direct solution of Circuit 1.
"""
import numpy as np

# Mesh equations from Slides 7–8:
#  5*I1 - I2            = 9
# -I1 + 6*I2 - 2*I3     = 2
#       -2*I2 + 7*I3    = 5
A = np.array([[5, -1, 0],
              [-1, 6, -2],
              [0, -2, 7]], dtype=float)
b = np.array([9, 2, 5], dtype=float)

# Direct method: NumPy's linear system solver (LU-based internally)
currents = np.linalg.solve(A, b)

print('CIRCUIT 1 — DIRECT METHOD')
for i, current in enumerate(currents, start=1):
    print(f'I{i} = {current:.4f} A')

# Verify A @ x = b (np.dot gives the same matrix-vector product)
verified_b = np.dot(A, currents)
print('\nCalculated A @ currents:', verified_b)
print('Original b:           ', b)
print('Match (np.allclose):  ', np.allclose(verified_b, b))

