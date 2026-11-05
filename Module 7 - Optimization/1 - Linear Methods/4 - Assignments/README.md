**7.1.4.1 - Linear Systems**

An electrical circuit analysis produces a system of linear equations `A @ x = b`, where `x` holds the unknown
currents. In this assignment you will solve such systems using the direct and iterative methods from lecture
(see `0 - Lecture Examples`).

Complete the four functions in `linear_systems.py`. Do not rename the functions or change their parameters or return values.
Your functions are tested with several different square, nonsingular systems (symmetric positive definite with a
nonzero diagonal), not just the 5-by-5 example at the bottom of the file, so do not hard-code the sample answers.

1. `direct_solve(_A, _b)` - Solve the system with [np.linalg.solve()](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html). (20 points)
2. `verify_solution(_A, _b, _x)` - Return `(calculated_b, matches)`, where `calculated_b` is `A @ x` and `matches` is the
   `bool` result of [np.allclose()](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html) comparing it to `b`. (20 points)
3. `jacobi_solve(_A, _b, _tolerance, _max_iterations)` - Jacobi iteration starting from an all-zero estimate, using only
   previous-iteration values for every new estimate. Stop when `max(abs(x_new - x_old)) < _tolerance` or after `_max_iterations`
   updates. Return `(x, iterations, converged)`. (40 points)
4. `scipy_iterative_solve(_A, _b, _rtol)` - Solve with [scipy.sparse.linalg.cg()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html)
   (Conjugate Gradient). Return `(x, info)`, where `info == 0` means the solver converged. (20 points)

Run the file locally to compare your output with the expected values printed next to each result. Submit only the
completed `linear_systems.py` to Gradescope. You have 3 graded submissions.
