**7.2.4.1 - Nonlinear Systems**

An engineer is designing a rectangular solar panel array. The length `x` and width `y` must have a combined length of
9 m, and the diagonal must be √41 m (about 6.40 m). Find the dimensions using the numerical methods from lecture
(see `0 - Lecture Examples`).

The equations and Jacobian are provided, so you do not need to derive any partial derivatives:

```
f1(x, y) = x² + y² − 41 = 0
f2(x, y) = x + y − 9 = 0
J(x, y)  = [[2x, 2y], [1, 1]]

Newton-Raphson: solve J(xₖ) Δx = −F(xₖ), then xₖ₊₁ = xₖ + Δx
```

Complete the four `TODO` sections in `nonlinear_systems.py`. Do not rename the functions or change their parameters.
Use NumPy and SciPy as taught in class. Do not hard-code the final dimensions (4 and 5).

1. `residuals(dimensions)` - Return a NumPy array of shape `(2,)` with `[f1, f2]`. (20 points)
2. `jacobian(dimensions)` - Return the `(2, 2)` Jacobian matrix. (20 points)
3. `newton_raphson(initial_guess, tolerance, max_iterations)` - Use a `for` loop and `np.linalg.solve(J, -F)`. Stop when the
   maximum absolute residual is strictly smaller than `tolerance`. Raise `RuntimeError` if it does not converge within
   `max_iterations`. Do not modify the `initial_guess` array you are given. (40 points)
4. `solve_scipy(initial_guess)` - Use [scipy.optimize.root()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html)
   and check `solution.success`; raise `RuntimeError` if the solver fails. (20 points)

Use the initial guess `[3.0, 6.0]` for your local demonstration. The grader also uses other reasonable guesses,
including `[6.0, 3.0]`. Either `(4, 5)` or `(5, 4)` is a valid answer. Run the file locally before submitting.
Submit only the completed `nonlinear_systems.py` to Gradescope. You have 3 graded submissions.
