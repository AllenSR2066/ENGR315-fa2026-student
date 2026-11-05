ENGR 315 | Numerical Methods: Nonlinear Systems and Approximations
================================================================

Run the files in this order:

1. ENGR315_Nonlinear_01_Newton_Raphson_NumPy.py
   - Performs Newton-Raphson with NumPy, a for loop, an explicit Jacobian,
     and np.linalg.solve().
   - Prints each iteration, including the initial guess and residual.
   - Verifies the tank dimensions, perimeter, and diagonal.

2. ENGR315_Nonlinear_02_SciPy_Root.py
   - Solves the same equations using scipy.optimize.root(method='hybr').
   - Demonstrates root(method='broyden1').
   - Shows a different initial guess and compares the solutions.

Required packages:
   python -m pip install numpy scipy

Run in VS Code's terminal:
   python ENGR315_Nonlinear_01_Newton_Raphson_NumPy.py
   python ENGR315_Nonlinear_02_SciPy_Root.py

MATHEMATICAL MODEL
   length**2 + width**2 - 25 = 0
   length + width - 7 = 0

INITIAL GUESS: [2.0, 4.0]
RESIDUAL TOLERANCE for NumPy implementation: 0.001

IMPORTANT LECTURE CORRECTION
The earlier intermediate iteration values shown in some draft slides were
incorrect. Verified Newton-Raphson iterations are:

   0: [2.000000, 4.000000], maximum residual 5.00000000
   1: [2.750000, 4.250000], maximum residual 0.62500000
   2: [2.958333, 4.041667], maximum residual 0.08680556
   3: [2.998397, 4.001603], maximum residual 0.00321026
   4: [2.999997, 4.000003], maximum residual 0.00000512

Because the stopping rule is maximum absolute residual < 0.001,
convergence is achieved at iteration 4 (not iteration 3).

The solved dimensions are approximately 3 m by 4 m; swapping the two
corresponds to the other valid solution of the same design equations.
