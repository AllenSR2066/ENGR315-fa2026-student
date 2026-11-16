"""
Module 7.3 - Multivariate Linear Regression
Example 1: fit rent from size, bedrooms and bathrooms for 14 Shenandoah Valley apartments.

Model:  rent = b0 + b1*square_feet + b2*bedrooms + b3*bathrooms
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# 1. Load the 14 apartments
# ---------------------------------------------------------------
apartments = pd.read_csv(
    "../../../data/apartments/apartments_shenandoah_sample.csv")
print(apartments[["cityname", "price", "square_feet", "bedrooms", "bathrooms"]])

# ---------------------------------------------------------------
# 2. One input: rent vs. square feet (same idea as np.polyfit)
# ---------------------------------------------------------------
slope, intercept = np.polyfit(apartments["square_feet"], apartments["price"], 1)
print(f"\nOne input:  rent = {intercept:.0f} + {slope:.2f} * square_feet")

# ---------------------------------------------------------------
# 3. Several inputs: build X (one row per apartment) and y
# ---------------------------------------------------------------
print("\nSeveral inputs:")

# Inputs and output
input_names = ["square_feet", "bedrooms", "bathrooms"]
inputs = apartments[input_names].to_numpy()
y = apartments["price"].to_numpy()

# Column of 1s for the intercept b0
ones = np.ones(len(apartments))
X = np.column_stack([ones, inputs])

# Least-squares solution of X b = y (sv: not needed)
b, residual_sum, rank, sv = np.linalg.lstsq(X, y, rcond=None)
for name, weight in zip(["intercept"] + input_names, b):
    print(f"{name:12s} {weight:8.2f}")

# ---------------------------------------------------------------
# 4. How good is the fit?  R^2 = 1 - SS_residual / SS_total
# ---------------------------------------------------------------
predicted = X @ b
ss_residual = np.sum((y - predicted) ** 2)
ss_total = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - ss_residual / ss_total    # 0.63
print(f"\nR^2 = {r_squared:.2f}")

# Predict the rent of a new apartment: 900 sq ft, 2 bedrooms, 1 bathroom
new_rent = np.array([1, 900, 2, 1]) @ b   # about $808
print(f"Predicted rent for 900 sq ft, 2 bed, 1 bath: ${new_rent:.0f}")


# ---------------------------------------------------------------
# 5. Which inputs matter?  Standardize (z-score) so the weights are comparable
#    z = (value - mean) / standard deviation
# ---------------------------------------------------------------
def fit_r_squared(columns):
    """Fit rent from the given columns and return R^2."""
    X_cols = np.column_stack([ones, apartments[columns].to_numpy()])
    b_cols = np.linalg.lstsq(X_cols, y, rcond=None)[0]
    residuals = y - X_cols @ b_cols
    return 1 - np.sum(residuals ** 2) / ss_total


# means: average of each column.  spreads: standard deviation of each column.
means = inputs.mean(axis=0)
spreads = inputs.std(axis=0)
z_inputs = (inputs - means) / spreads

print("\nColumn means:              ", np.round(means, 2))
print("Column standard deviations:", np.round(spreads, 2))
print("First apartment, raw:      ", inputs[0])
print("First apartment, z-scores: ", np.round(z_inputs[0], 2))

X_z = np.column_stack([ones, z_inputs])
b_z = np.linalg.lstsq(X_z, y, rcond=None)[0]

print("\nStandardized weights ($ per 1 standard deviation):")
for name, weight in zip(input_names, b_z[1:]):
    print(f"  {name:14s} {weight:8.0f}")

# Divide by the standard deviation of rent too: the weights lose their units ("beta"
# weights) and usually fall between -1 and +1.
betas = b_z[1:] / np.std(y)
print("\nBeta weights (unitless):")
for name, beta in zip(input_names, betas):
    print(f"  {name:14s} {beta:8.2f}")

# ---------------------------------------------------------------
# 6. Drop each input in turn and see how much R^2 falls
# ---------------------------------------------------------------
print("\nR^2 lost when an input is dropped:")
for name in input_names:
    remaining = [n for n in input_names if n != name]
    print(f"  without {name:14s} R^2 = {fit_r_squared(remaining):.2f}"
          f"  (drop of {r_squared - fit_r_squared(remaining):.2f})")
