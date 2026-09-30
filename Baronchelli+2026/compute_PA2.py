import numpy as np


# Data for Table 2
# Note: uncertainties are treated as Type B errors
# (rectangular probability distribution), following international conventions.
# With this definition, the quoted error is divided by sqrt(3) to obtain
# the corresponding Gaussian uncertainty used in the inverse-variance weights.


# Format: (ID, PA, Error, Region, Stacked, MeetingCriteria)

data = [
    # North - stacked
    (19874, 110, 10.0 / np.sqrt(3.0), 'North', True, True),
    (35274, 105, 10.0 / np.sqrt(3.0), 'North', True, True),
    (35311, 120, 15.0 / np.sqrt(3.0), 'North', True, True),
    (35350, 115, 10.0 / np.sqrt(3.0), 'North', True, True),
    (35360, 105, 15.0 / np.sqrt(3.0), 'North', True, True),
    (35364, 100, 15.0 / np.sqrt(3.0), 'North', True, True),

    # North - not stacked
    (33527, 35, 10.0 / np.sqrt(3.0), 'North', False, False),   # < 100: filtered out
    (34205, 140, 10.0 / np.sqrt(3.0), 'North', False, False),
    (34597, 135, 10.0 / np.sqrt(3.0), 'North', False, False),
    (35265, 130, 10.0 / np.sqrt(3.0), 'North', False, False),
    (35380, 60, 15.0 / np.sqrt(3.0), 'North', False, False),   # < 100: filtered out
    (34840, 110, 10.0 / np.sqrt(3.0), 'North', False, False),
    (35142, 105, 10.0 / np.sqrt(3.0), 'North', False, False),

    # South - stacked
    (53568, 135, 20.0 / np.sqrt(3.0), 'South', True, True),
    (53600, 125, 10.0 / np.sqrt(3.0), 'South', True, True),
    (53617, 130, 10.0 / np.sqrt(3.0), 'South', True, True),
    (53619, 125, 10.0 / np.sqrt(3.0), 'South', True, True),

    # South - not stacked
    (53683, 120, 15.0 / np.sqrt(3.0), 'South', False, False),
    (34667, 135, 10.0 / np.sqrt(3.0), 'South', False, False),
    (34672, 25, 15.0 / np.sqrt(3.0), 'South', False, False),   # < 100: filtered out
    (34686, 135, 20.0 / np.sqrt(3.0), 'South', False, False),
    (34690, 125, 15.0 / np.sqrt(3.0), 'South', False, False),
    (53677, 130, 15.0 / np.sqrt(3.0), 'South', False, False),
    (53723, 130, 20.0 / np.sqrt(3.0), 'South', False, False),
    (53742, 25, 10.0 / np.sqrt(3.0), 'South', False, True),    # < 100: filtered out despite "yes"
    (53763, 60, 10.0 / np.sqrt(3.0), 'South', False, False),   # < 100: filtered out
]


def weighted_linear_mean(pa_values, errors):
    """Return the inverse-variance weighted arithmetic mean and its uncertainty."""
    weights = 1.0 / (errors ** 2)
    mean = np.sum(pa_values * weights) / np.sum(weights)
    uncertainty = np.sqrt(1.0 / np.sum(weights))
    return mean, uncertainty


def weighted_circular_mean_180(pa_values, errors):
    """
    Return the inverse-variance weighted circular mean for 180-degree-periodic PA data.

    Position angles are axial quantities, so theta and theta + 180 degrees
    represent the same orientation. The angles are therefore doubled before
    applying the circular mean, and the result is divided by two afterward.

    The returned uncertainty is the same inverse-variance uncertainty used for
    the linear weighted mean (not a full circular-statistics confidence interval).
    """
    weights = 1.0 / (errors ** 2)

    angles_rad = np.deg2rad(pa_values)
    doubled_angles = 2.0 * angles_rad

    weighted_cosine = np.sum(weights * np.cos(doubled_angles))
    weighted_sine = np.sum(weights * np.sin(doubled_angles))

    mean_rad = 0.5 * np.arctan2(weighted_sine, weighted_cosine)
    mean_deg = np.rad2deg(mean_rad) % 180.0

    uncertainty = np.sqrt(1.0 / np.sum(weights))
    return mean_deg, uncertainty


def weighted_mean_analysis(subset, name):
    # Keep only PA values between 100 and 170 degrees, inclusive.
    filtered = [d for d in subset if 100 <= d[1] <= 170]

    print(f"\n--- {name} ---")

    if not filtered:
        print("No valid data remain after filtering.")
        return

    pas = np.array([d[1] for d in filtered], dtype=float)
    errors = np.array([d[2] for d in filtered], dtype=float)

    # Calculate both weighted means.
    linear_mean, linear_uncertainty = weighted_linear_mean(pas, errors)
    circular_mean, circular_uncertainty = weighted_circular_mean_180(pas, errors)

    excluded = len(subset) - len(filtered)
    print(f"Sources used (n): {len(filtered)} (excluded after filtering: {excluded})")
    print(f"Weighted linear mean:  {linear_mean:.2f} deg +/- {linear_uncertainty:.2f} deg")
    print(f"Weighted circular mean (180-deg periodic): {circular_mean:.2f} deg +/- {circular_uncertainty:.2f} deg")


# --- SCENARIO DEFINITIONS ---

# Scenario A: Core North (Stacked Sample)
scenario_a = [d for d in data if d[3] == 'North' and d[4] is True]

# Scenario B: Extended North (all North sources with a numerical PA)
scenario_b = [d for d in data if d[3] == 'North']

# Scenario C: Core South (Stacked + ID 53742)
scenario_c = [d for d in data if (d[3] == 'South' and d[4] is True) or d[0] == 53742]

# Scenario D: Extended South (all South sources with a numerical PA)
scenario_d = [d for d in data if d[3] == 'South']

# Scenario E: Global (all sources in the dataset)
scenario_e = data


# --- RUN ANALYSES ---
weighted_mean_analysis(scenario_a, "Scenario A (Core North)")
weighted_mean_analysis(scenario_b, "Scenario B (Extended North)")
weighted_mean_analysis(scenario_c, "Scenario C (Core South)")
weighted_mean_analysis(scenario_d, "Scenario D (Extended South)")
weighted_mean_analysis(scenario_e, "Scenario E (Global Combined)")
