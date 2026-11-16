"""
Module 7.3 - Multivariate Linear Regression
Example 2: predict monthly rent for 8,000+ Virginia apartments from
quantitative inputs (size, bedrooms, bathrooms) and qualitative inputs
(region, amenities, pet policy).
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# 1. Load and clean
# ---------------------------------------------------------------
apartments = pd.read_csv("../../../data/apartments/apartments_for_rent_VA.csv")
apartments = apartments.drop_duplicates()
apartments = apartments.dropna(subset=["bedrooms", "bathrooms"])
print(f"{len(apartments)} listings")

# ---------------------------------------------------------------
# 2. Qualitative -> quantitative
# ---------------------------------------------------------------
# 2a. City name -> region (a category with 4 levels).
#     Drawing these boundaries is a judgment call.
northern_virginia = ["Arlington", "Alexandria", "Fairfax", "Woodbridge", "Reston",
                     "Falls Church", "Vienna", "Herndon", "Ashburn", "Mc Lean", "Manassas",
                     "Stafford", "Leesburg", "Springfield", "Sterling", "Annandale", "Lorton",
                     "Centreville", "Manassas Park", "South Riding", "Gainesville",
                     "Fort Belvoir", "Chantilly", "Oakton", "Haymarket", "Nokesville",
                     "Round Hill", "Burke", "Bristow", "Dumfries", "Great Falls",
                     "Lovettsville", "Dunn Loring"]
hampton_roads = ["Norfolk", "Virginia Beach", "Newport News", "Hampton", "Yorktown",
                 "Chesapeake", "Williamsburg", "Portsmouth", "Fort Eustis", "Suffolk",
                 "Poquoson", "Smithfield", "Gloucester", "Hayes"]
richmond_area = ["Richmond", "Henrico", "Petersburg", "Midlothian", "Glen Allen", "Chester",
                 "Mechanicsville", "Ashland", "Colonial Heights", "Hopewell", "Prince George",
                 "Chesterfield", "Highland Springs", "Powhatan", "Oilville"]


def city_to_region(city):
    if city in northern_virginia:
        return "Northern Virginia"
    if city in hampton_roads:
        return "Hampton Roads"
    if city in richmond_area:
        return "Richmond"
    return "Rest of Virginia"


# Turn each city name into one of 4 regions (function defined in the script)
apartments["region"] = apartments["cityname"].apply(city_to_region)

# Blank amenity lists become "" so the text search works on every row
amenities = apartments["amenities"].fillna("")

# For each amenity: 1 if it appears in the listing's list, otherwise 0
for amenity in ["Gym", "Parking", "Pool", "Washer Dryer"]:
    column_name = "has_" + amenity.lower().replace(" ", "_")
    apartments[column_name] = amenities.str.contains(amenity).astype(int)

# Pet policy: 1 if the listing states one, 0 if it is blank ("not stated")
apartments["pets_allowed_listed"] = apartments["pets_allowed"].notna().astype(int)

# One 0/1 column per region; drop_first=True leaves out Hampton Roads (the baseline)
region_columns = pd.get_dummies(apartments["region"], prefix="region", drop_first=True, dtype=int)

# Add the new region columns to the table
apartments = pd.concat([apartments, region_columns], axis=1)

print(apartments["region"].value_counts())

# ---------------------------------------------------------------
# 3. Fit:  X b = y  by least squares
# ---------------------------------------------------------------
y = apartments["price"].to_numpy()


def fit(columns):
    """Fit rent from the given columns. Returns (weights, R^2)."""
    # Build X: a column of 1s, then one column per input
    X = np.column_stack([np.ones(len(apartments)),
                         apartments[columns].to_numpy()])

    # Least-squares weights, exactly as for the 14 apartments
    b = np.linalg.lstsq(X, y, rcond=None)[0]

    # R^2: how much of the variation in rent the model explains
    residuals = y - X @ b
    r_squared = 1 - np.sum(residuals ** 2) / np.sum((y - np.mean(y)) ** 2)
    return b, r_squared


size = ["square_feet"]
rooms = ["bedrooms", "bathrooms"]
location = list(region_columns.columns)
features = ["has_gym", "has_parking", "has_pool", "has_washer_dryer", "pets_allowed_listed"]

print("\nR^2 as inputs are added:")
print(f"  size only                    {fit(size)[1]:.2f}")
print(f"  + bedrooms, bathrooms        {fit(size + rooms)[1]:.2f}")
print(f"  + region                     {fit(size + rooms + location)[1]:.2f}")
print(f"  + amenities, pets            {fit(size + rooms + location + features)[1]:.2f}")

all_inputs = size + rooms + location + features
b, r_squared = fit(all_inputs)
print("\nFull model weights:")
print(f"  {'intercept':22s} {b[0]:8.2f}")
for name, weight in zip(all_inputs, b[1:]):
    print(f"  {name:22s} {weight:8.2f}")

# ---------------------------------------------------------------
# 4. Which group of inputs matters most?  Drop each group, refit, compare R^2
# ---------------------------------------------------------------
groups = {"size": size, "rooms": rooms, "location": location, "amenities and pets": features}
print("\nR^2 lost when a group is dropped:")
for group_name, group_columns in groups.items():
    remaining = [c for c in all_inputs if c not in group_columns]
    print(f"  {group_name:20s} {r_squared - fit(remaining)[1]:.3f}")

# ---------------------------------------------------------------
# 5. Standardized weights: z-score every input so the weights can be compared
#    (weight = change in rent for a one-standard-deviation change in that input)
# ---------------------------------------------------------------
inputs = apartments[all_inputs].to_numpy(dtype=float)
z = (inputs - inputs.mean(axis=0)) / inputs.std(axis=0)
X_z = np.column_stack([np.ones(len(apartments)), z])
b_z = np.linalg.lstsq(X_z, y, rcond=None)[0]

# Beta weights: also divide by the standard deviation of rent, so they have no units
betas = b_z[1:] / np.std(y)

print("\nStandardized weights, largest first ($ per 1 std dev, and unitless beta):")
order = np.argsort(-np.abs(b_z[1:]))
for i in order:
    print(f"  {all_inputs[i]:26s} {b_z[1 + i]:8.0f}   beta {betas[i]:+.2f}")
