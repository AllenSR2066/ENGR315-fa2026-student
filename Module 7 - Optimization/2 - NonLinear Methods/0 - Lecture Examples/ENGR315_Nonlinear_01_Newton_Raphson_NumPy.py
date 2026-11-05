"""ENGR 315 | Nonlinear Systems | Example 1: Newton-Raphson with NumPy.

Engineering problem: A rectangular water-tank base has a perimeter of
14 meters and a diagonal of 5 meters. Find its length and width.

Run: python ENGR315_Nonlinear_01_Newton_Raphson_NumPy.py
Requires: numpy
"""

import numpy as np


def tank_equations(dimensions):
    """Return residuals [length**2 + width**2 - 25, length + width - 7]."""
    length, width = dimensions
    return np.array([length**2 + width**2 - 25, length + width - 7], dtype=float)


def tank_jacobian(dimensions):
    """Return the 2 x 2 Jacobian of the tank equations."""
    length, width = dimensions
    return np.array([[2 * length, 2 * width], [1.0, 1.0]], dtype=float)


def newton_raphson(initial_guess, tolerance=0.001, max_iterations=20):
    """Solve the two nonlinear equations using Newton-Raphson iterations.

    Stop when the maximum absolute residual falls below tolerance.
    The method may fail if the Jacobian is singular or an initial guess
    is unsuitable.
    """
    dimensions = np.array(initial_guess, dtype=float)

    print("Iteration |   Length (m) |    Width (m) | Max |residual|")
    print("-" * 59)

    for iteration in range(max_iterations + 1):
        residuals = tank_equations(dimensions)
        error = np.max(np.abs(residuals))
        print(f"{iteration:9d} | {dimensions[0]:12.6f} | "
              f"{dimensions[1]:12.6f} | {error:14.8f}")

        if error < tolerance:
            return dimensions, True

        if iteration == max_iterations:
            break

        # J * correction = -F : reuse np.linalg.solve from linear systems!
        jacobian = tank_jacobian(dimensions)
        try:
            correction = np.linalg.solve(jacobian, -residuals)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Newton-Raphson cannot proceed: the Jacobian is singular. "
                "Try a different initial guess."
            ) from exc

        dimensions = dimensions + correction

    return dimensions, False


def verify_dimensions(dimensions):
    """Check the physical dimensions and numerical residuals."""
    length, width = dimensions
    perimeter = 2 * (length + width)
    diagonal = np.sqrt(length**2 + width**2)
    residuals = tank_equations(dimensions)
    print("\nVERIFICATION")
    print(f"Length:             {length:.6f} m")
    print(f"Width:              {width:.6f} m")
    print(f"Perimeter:          {perimeter:.6f} m (target: 14 m)")
    print(f"Diagonal:           {diagonal:.6f} m (target: 5 m)")
    print(f"Residuals:          {residuals}")
    print("Requirements met:  ", bool(np.allclose([perimeter, diagonal],
                                                  [14, 5], rtol=0, atol=0.001)))


if __name__ == "__main__":
    initial_guess = [2.0, 4.0]
    tolerance = 0.001
    max_iterations = 20

    answer, converged = newton_raphson(initial_guess, tolerance, max_iterations)
    print("\nConverged:", converged)
    verify_dimensions(answer)
