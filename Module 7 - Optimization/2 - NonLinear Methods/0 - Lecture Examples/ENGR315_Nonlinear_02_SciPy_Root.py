"""ENGR 315 | Nonlinear Systems | Example 2: SciPy's built-in solvers.

Same water-tank problem as Example 1. Compare SciPy's Powell hybrid
method with Broyden's first method and explore different initial guesses.

Run: python ENGR315_Nonlinear_02_SciPy_Root.py
Requires: numpy, scipy
"""

import numpy as np
from scipy.optimize import root


def tank_equations(dimensions):
    """Residuals for the water tank's diagonal and perimeter constraints."""
    length, width = dimensions
    return np.array([length**2 + width**2 - 25, length + width - 7], dtype=float)


def show_solution(label, result):
    """Report solver status, numerical residuals, and design constraints."""
    length, width = result.x
    perimeter = 2 * (length + width)
    diagonal = np.sqrt(length**2 + width**2)
    print(f"\n{label}")
    print(f"  Dimensions (m):  {np.round(result.x, 6)}")
    print(f"  Solver success:  {result.success}")
    print(f"  Residuals:       {np.round(result.fun, 9)}")
    print(f"  Max |residual|:  {np.max(np.abs(result.fun)):.9g}")
    print(f"  Perimeter:       {perimeter:.6f} m (target: 14)")
    print(f"  Diagonal:        {diagonal:.6f} m (target: 5)")
    print(f"  Message:         {result.message}")


if __name__ == "__main__":
    initial_guess = [2.0, 4.0]

    # The default root() method is "hybr" (modified Powell hybrid).
    hybrid = root(tank_equations, initial_guess, method="hybr")
    show_solution("1. SciPy: Powell hybrid", hybrid)

    # Compare an alternative algorithm on the SAME equations.
    broyden = root(tank_equations, initial_guess, method="broyden1")
    show_solution("2. SciPy: Broyden's first method", broyden)

    # Nonlinear equations may have multiple valid solutions.
    other_guess = [4.0, 2.0]
    other_solution = root(tank_equations, other_guess, method="hybr")
    show_solution("3. Powell hybrid, different initial guess", other_solution)

    # Compare against a verified NumPy Newton-Raphson result obtained
    # with the SAME initial guess and tolerance of 0.001.
    numpy_result = np.array([2.99999744, 4.00000256])
    print("\nCOMPARISON")
    print("  NumPy Newton-Raphson:", np.round(numpy_result, 6))
    print("  SciPy Powell hybrid: ", np.round(hybrid.x, 6))
    print("  Solutions close:     ", bool(np.allclose(
        numpy_result, hybrid.x, atol=0.001, rtol=0
    )))
