"""
Module 7.3 - Multivariate Linear Regression
Example 3 (optional): split the full model's R^2 fairly among groups of inputs.

Size, rooms and location overlap (bigger apartments have more rooms), so the R^2 a group
adds depends on which groups are already in the model. This script adds the groups in
every possible order and averages each group's gain (a "Shapley" split). The shares add
up exactly to the full model's R^2.

Run 02_full_regression.py first to see how the inputs are built; this file repeats that
setup by importing it (with example 2's printing switched off).
"""

import contextlib
import io
import itertools
from importlib import import_module

import numpy as np

# Re-use the cleaned data, the fit() function and the input groups from example 2.
# redirect_stdout sends example 2's printed output to a throwaway buffer, so only this
# script's results appear.
with contextlib.redirect_stdout(io.StringIO()):
    example = import_module("02_full_regression")
fit = example.fit
groups = example.groups

group_names = list(groups)
gains = {name: [] for name in group_names}

for order in itertools.permutations(group_names):
    columns_so_far = []
    r_squared_so_far = 0.0
    for name in order:
        columns_so_far = columns_so_far + groups[name]
        new_r_squared = fit(columns_so_far)[1]
        gains[name].append(new_r_squared - r_squared_so_far)
        r_squared_so_far = new_r_squared

total = sum(np.mean(g) for g in gains.values())
print(f"\nShare of the full model's R^2 = {total:.2f}:")
for name in group_names:
    share = np.mean(gains[name])
    print(f"  {name:20s} {share:.3f}  ({100 * share / total:.0f}% of what the model explains)")
