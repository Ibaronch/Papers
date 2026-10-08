import numpy as np
from scipy.stats import binom, norm
import math
import matplotlib.pyplot as plt
from pdb import set_trace as stop 

def calculate_alignment_significance(k, n, delta_theta):
    """
    Calculates the jet alignment probabilities.
    
    Parameters:
    k (int): Number of aligned sources.
    n (int): Total number of sources in the sample.
    delta_theta (float): Angular window width in degrees.
    """
    
    # 1. Individual probability p (fraction of the 180° semicircle)
    p = delta_theta / 180.0
    
    # 2. P(fixed theta): Cumulative binomial probability P(X >= k)
    # binom.sf(k-1, n, p) calculates the "Survival Function", i.e., P(X > k-1)
    p_fixed = binom.sf(k - 1, n, p)
    
    # 3. P(any theta): Scan Statistic approximation for the Look-Elsewhere Effect
    # Formula: n * binom_coeff(n-1, k-1) * p^(k-1) * (1-p)^(n-k)
    if k > 0:
        comb_coeff = math.comb(n - 1, k - 1)
        p_any = n * comb_coeff * (p**(k - 1)) * ((1 - p)**(n - k))
        p_any = min(p_any, 1.0) # Cap at 1.0 if the approximation exceeds it
    else:
        p_any = 1.0

    # 4. Conversion to Sigma (Z-score)
    # We use norm.isf (Inverse Survival Function) to convert the probability to sigma
    sigma_fixed = norm.isf(p_fixed)
    sigma_any = norm.isf(p_any)

    print(f"--- Results for n={n}, k={k}, Δθ={delta_theta}° ---")
    print(f"Individual probability (p): {p:.4f}")
    print(f"P(fixed theta): {p_fixed:.2e} ({sigma_fixed:.2f} σ)")
    print(f"P(any theta):   {p_any:.2e} ({sigma_any:.2f} σ)\n")

    return p_fixed, p_any

# --- EXAMPLES FROM YOUR PAPER ---

# Scenario A: Core North
print("Scenario A")
p_fixed_A, p_any_A=calculate_alignment_significance(k=6, n=6, delta_theta=20)

# Scenario B: Extended North
print("Scenario B")
p_fixed_B, p_any_B=calculate_alignment_significance(k=11, n=13, delta_theta=40)

# Scenario C: Core South
# Only jets
print("Scenario C - ANY THETA (DO NOT CONSIDER THE FIXED THETA ONE)")
_, p_any_C=calculate_alignment_significance(k=4, n=5, delta_theta=10)
# widened angular window to account for the inclusion of the filament at 115°
print("Scenario C - FIXED THETA (DO NOT CONSIDER THE ANY THETA ONE)")
p_fixed_C, _=calculate_alignment_significance(k=4, n=5, delta_theta=20)

# Scenario D: Extended South
# Only jets
print("Scenario D - ANY THETA (DO NOT CONSIDER THE FIXED THETA ONE)")
_, p_any_D=calculate_alignment_significance(k=10, n=13, delta_theta=15)
# widened angular window to account for the inclusion of the filament at 115°
print("Scenario D - FIXED THETA (DO NOT CONSIDER THE ANY THETA ONE)")
p_fixed_D, _=calculate_alignment_significance(k=10, n=13, delta_theta=20)

# Scenario E: Combined (Global Conservative)
p_fixed_E, p_any_E=calculate_alignment_significance(k=21, n=26, delta_theta=40)

print('combined probabilities (BxD)')
print(p_fixed_B*p_fixed_D, p_any_B*p_any_D)

p_fixed_BxD=p_fixed_B*p_fixed_D
p_any_BxD=p_any_B*p_any_D

# Compute corresponding significance:
# Conversion to Sigma (Z-score)
# We use norm.isf (Inverse Survival Function) to convert the probability to sigma
sigma_fixed_BxD = norm.isf(p_fixed_BxD)
sigma_any_BxD = norm.isf(p_any_BxD)

print(f"--- Results for combined probabilities BxD ---")
print(f"P(fixed theta): {p_fixed_BxD:.2e} ({sigma_fixed_BxD:.2f} σ)")
print(f"P(any theta):   {p_any_BxD:.2e} ({sigma_any_BxD:.2f} σ)\n")

print('---------------------------------\n')



##############################################
# PA
combined_PAs = np.array([110, 105, 120, 115, 105, 100, 35, 140, 135, 130, 60, 110, 105, 135, 125, 130, 125, 120, 135, 25, 135, 125, 130, 130, 25, 60])

# PA uncertainties
combined_PAs_E=(1.0/np.sqrt(3.0))*np.array([10, 10, 15, 10, 15, 15, 10, 10, 10, 10, 15, 10, 10, 20, 10, 10, 10, 15, 10, 15, 20, 15, 15, 20, 10, 10])

# ------------------------------------------------------------
# Central direction physically adopted in the paper # 5.4
theta0 = 115.0
# theta0 = 120
# theta0 = 121.7
# ------------------------------------------------------------
theta0_unc=(1.0/np.sqrt(3.0))*15.0

##############################################



##############################################
# PART 2 - plot significance Vs angular window.
##############################################


# ============================================================
# SIGNIFICANCE ANALYSIS AS A FUNCTION OF DELTA_THETA
# ONLY FOR THE COMBINED SAMPLE
# ============================================================

# Delta_theta value adopted in the paper
delta_theta_adopted = 40.0

# Range of window widths to explore.
# Here Delta_theta is the TOTAL WIDTH of the window,
# consistent with p = Delta_theta / 180 in the code above.
delta_theta_values = np.arange(10, 91.0, 10)

n_combined = len(combined_PAs)


p_fixed_values = []
p_any_values = []
sigma_fixed_values = []
sigma_any_values = []
k_values = []

for delta_theta in delta_theta_values:

    # --------------------------------------------------------
    # Count how many sources fall in the window centered
    # on the filament direction theta0.
    #
    # Since PAs are axial (modulo 180°), the angular
    # distance must be calculated modulo 180°.
    # --------------------------------------------------------

    angular_distance = np.abs(
        ((combined_PAs - theta0 + 90.0) % 180.0) - 90.0
    )

    # Delta_theta is the TOTAL width of the window,
    # so the half-width is Delta_theta / 2.
    k = np.sum(angular_distance <= delta_theta / 2.0)

    k_values.append(k)

    # --------------------------------------------------------
    # Calculate p-value and significance using the same function
    # already defined above.
    
    p_fixed, p_any=calculate_alignment_significance(k=k, n=n_combined, delta_theta=delta_theta)
    
    sigma_fixed = norm.isf(p_fixed)
    sigma_any = norm.isf(p_any)

    p_fixed_values.append(p_fixed)
    p_any_values.append(p_any)
    sigma_fixed_values.append(sigma_fixed)
    sigma_any_values.append(sigma_any)


# Convert everything to numpy arrays
p_fixed_values = np.array(p_fixed_values)
p_any_values = np.array(p_any_values)
sigma_fixed_values = np.array(sigma_fixed_values)
sigma_any_values = np.array(sigma_any_values)
k_values = np.array(k_values)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n============================================================")
print("SIGNIFICANCE AS A FUNCTION OF DELTA_THETA")
print("COMBINED SAMPLE ONLY")
print("============================================================")

print(f"{'Delta_theta':>12} {'k':>5} "
      f"{'P_fixed':>15} {'sigma_fixed':>15} "
      f"{'P_any':>15} {'sigma_any':>15}")

print("-" * 82)

for delta, k, pf, sf, pa, sa in zip(
        delta_theta_values,
        k_values,
        p_fixed_values,
        sigma_fixed_values,
        p_any_values,
        sigma_any_values):

    print(f"{delta:12.1f} {k:5d} "
          f"{pf:15.3e} {sf:15.3f} "
          f"{pa:15.3e} {sa:15.3f}")


# ============================================================
# PLOT
# ============================================================

#fig, ax_sigma = plt.subplots(figsize=(9.0, 6.0))
fig, ax_sigma = plt.subplots(figsize=(10.0, 5.0))

# ------------------------------------------------------------
# Left axis: significance
# ------------------------------------------------------------

line1, = ax_sigma.plot(
    delta_theta_values,
    sigma_fixed_values,
    marker='o',
    linewidth=2,
    label='Fixed direction'
)

line2, = ax_sigma.plot(
    delta_theta_values,
    sigma_any_values,
    marker='s',
    linewidth=2,
    linestyle='--',
    label='Any direction (LEE)'
)

ax_sigma.set_xlabel(
    r'Angular window width $\Delta\theta$ (deg)',
    fontsize=21,
    fontweight='bold'
)

ax_sigma.set_ylabel(
    r'Significance ($\sigma$)',
    fontsize=21,
    fontweight='bold'
)

ax_sigma.grid(
    True,
    linestyle=':',
    alpha=0.5
)

# ------------------------------------------------------------
# Vertical line on the adopted Delta_theta
# ------------------------------------------------------------

ax_sigma.axvline(
    delta_theta_adopted,
    linestyle=':',
    linewidth=2,
    color='black',
    label=rf'Adopted $\Delta\theta={delta_theta_adopted:.0f}^\circ$'
)

# ------------------------------------------------------------
# Right axis: p-value
# ------------------------------------------------------------

ax_p = ax_sigma.twinx()

line3, = ax_p.plot(
    delta_theta_values,
    p_fixed_values,
    marker='o',
    linewidth=1.5,
    alpha=0.35
)

line4, = ax_p.plot(
    delta_theta_values,
    p_any_values,
    marker='s',
    linewidth=1.5,
    linestyle='--',
    alpha=0.35
)

ax_p.set_ylabel(
    r'$p$-value',
    fontsize=21,
    fontweight='bold'
)

ax_p.set_yscale('log')

# ------------------------------------------------------------
# Limits and legend
# ------------------------------------------------------------

ax_sigma.set_xlim(
    delta_theta_values.min() - 2.5,
    delta_theta_values.max() + 2.5
)

# Prevent the legend from overlapping the plot
lines = [line1, line2]
labels = [line1.get_label(), line2.get_label()]

ax_sigma.legend(
    lines,
    labels,
    loc='best',
    fontsize=19
)

plt.title(
    'Combined sample: alignment significance vs. angular window',
    fontsize=17,
    fontweight='bold'
)

# Axes ticks configuration for better readability
# X-axis and Left Y-axis (ax_sigma)
ax_sigma.tick_params(axis='both', which='major', labelsize=18, width=2, length=6)
for label in ax_sigma.get_xticklabels() + ax_sigma.get_yticklabels():
    label.set_fontweight('bold')
# Right Y-axis (ax_p)
ax_p.tick_params(axis='y', which='major', labelsize=18, width=2, length=6)
for label in ax_p.get_yticklabels():
    label.set_fontweight('bold')
fig.tight_layout()

# ------------------------------------------------------------
# PDF Saving
# ------------------------------------------------------------

output_filename = 'combined_alignment_significance_vs_delta_theta.pdf'

fig.savefig(
    output_filename,
    format='pdf',
    bbox_inches='tight'
)

plt.show()

print(f"\nPlot saved to: {output_filename}")

##############################################
# PART 3 - MONTE CARLO SIMULATIONS & POLAR PLOT
##############################################
import matplotlib.cm as cm
import matplotlib.colors as mcolors

print("\n============================================================")
print("STARTING MONTE CARLO SIMULATIONS")
print("============================================================")

N_sim = 1000000 #original


# 1. Separation between the "aligned" sources and the 5 "non-aligned"
# The 5 sources left out are those between 25 and 60 degrees.
# They are always considered "failures" for the window calculation.
mask_aligned = combined_PAs > 60
k_mc = np.sum(mask_aligned) # 21
n_mc = len(combined_PAs)    # 26

# 2. Random matrix generation (vectorized for max speed)
# We perturb ALL 26 sources so they appear in the final polar plot
PA_pert_matrix_all = np.random.normal(loc=combined_PAs, scale=combined_PAs_E, size=(N_sim, n_mc))
theta0_pert_array = np.random.normal(loc=theta0, scale=theta0_unc, size=N_sim)

# We extract ONLY the 21 aligned sources to compute the window bounds
PA_pert_matrix_aligned = PA_pert_matrix_all[:, mask_aligned]

# 3. Delta_theta calculation for all simulations simultaneously
# We compute min and max along the rows (axis=1) of the ALIGNED data only
max_PA_pert = np.max(PA_pert_matrix_aligned, axis=1)
min_PA_pert = np.min(PA_pert_matrix_aligned, axis=1)

# ANY case: window based strictly on perturbed data bounds
delta_theta_any_MC = max_PA_pert - min_PA_pert

# FIXED case: widened window (if needed) to include perturbed theta_0
max_fixed_MC = np.maximum(max_PA_pert, theta0_pert_array)
min_fixed_MC = np.minimum(min_PA_pert, theta0_pert_array)
delta_theta_fixed_MC = max_fixed_MC - min_fixed_MC

# 4. Vectorized probability calculation
# P_ANY
p_any_array = delta_theta_any_MC / 180.0
p_any_array = np.clip(p_any_array, 0.0, 1.0) # Prevent p > 1 for extreme perturbations
comb_coeff = math.comb(n_mc - 1, k_mc - 1)
p_val_any = n_mc * comb_coeff * (p_any_array**(k_mc - 1)) * ((1 - p_any_array)**(n_mc - k_mc))
p_val_any = np.clip(p_val_any, 0.0, 1.0)
sigma_any_MC = norm.isf(p_val_any)

# P_FIXED
p_fixed_array = delta_theta_fixed_MC / 180.0
p_fixed_array = np.clip(p_fixed_array, 0.0, 1.0)
p_val_fixed = binom.sf(k_mc - 1, n_mc, p_fixed_array)

sigma_fixed_MC = norm.isf(p_val_fixed)

# Avoid negative sigmas for p-values > 0.5
#sigma_any_MC[sigma_any_MC < 0] = 0
#sigma_fixed_MC[sigma_fixed_MC < 0] = 0

#print(f"MC Results over {N_sim} runs:")
#print(f"Mean p-value ANY:   {np.mean(p_val_any):.2e} +/- {np.std(p_val_any):.2e}")
#print(f"Mean Sigma ANY:     {np.mean(sigma_any_MC):.2f} +/- {np.std(sigma_any_MC):.2f}\n")
#print(f"Mean p-value FIXED: {np.mean(p_val_fixed):.2e} +/- {np.std(p_val_fixed):.2e}")
#print(f"Mean Sigma FIXED:   {np.mean(sigma_fixed_MC):.2f} +/- {np.std(sigma_fixed_MC):.2f}")
print("================================================")
# 1 sigma (68%)
perc_up=84
perc_down=16
# 2 sigma (95%)
#perc_up=97.5
#perc_down=2.5
# 3 sigma (99.7%)
#perc_up=99.85
#perc_down=0.15


#perc_down=6
#perc_down=5


print(f"Median p-value ANY:   {np.median(p_val_any):.2e} perc(16,84) --> {np.percentile(p_val_any,[perc_down])[0]:.2e} , {np.percentile(p_val_any,[perc_up])[0]:.2e}")
print(f"Median Sigma ANY:     {norm.isf(np.median(p_val_any)):.2f}  perc(16,84) -->  {norm.isf(np.percentile(p_val_any,[perc_down])[0]):.2f} , {norm.isf(np.percentile(p_val_any,[perc_up])[0]):.2f}\n")
print(f"Median p-value FIXED: {np.median(p_val_fixed):.2e} perc(16,84) --> {np.percentile(p_val_fixed,[perc_down])[0]:.2e} , {np.percentile(p_val_fixed,[perc_up])[0]:.2e}")                                  
print(f"Median Sigma FIXED:   {norm.isf(np.median(p_val_fixed)):.2f}  perc(16,84) -->  {norm.isf(np.percentile(p_val_fixed,[perc_down])[0]):.2f} , {norm.isf(np.percentile(p_val_fixed,[perc_up])[0]):.2f}\n")


# ============================================================
# POLAR DIAGRAM (ROSE PLOT) OF PERTURBED PAs
# ============================================================
print("\nGenerating polar diagram...")

# Flatten the array of ALL 260,000 perturbed PAs (10000 runs * 26 jets)
all_PA_pert = PA_pert_matrix_all.flatten() % 180.0
all_theta0_pert = theta0_pert_array % 180.0

# Since PAs are axial data (lines, not 0-360 directional vectors), 
# we duplicate the values offset by 180° so the polar plot 
# is symmetrical across the center as per convention.
plot_angles_rad = np.concatenate([np.radians(all_PA_pert), np.radians(all_PA_pert + 180.0)])
plot_theta0_rad = np.concatenate([np.radians(all_theta0_pert), np.radians(all_theta0_pert + 180.0)])

# Create bins (e.g., 5 degrees width (72) or 1° (360))
N_bins = 360#72
bins = np.linspace(0.0, 2 * np.pi, N_bins + 1)
counts, _ = np.histogram(plot_angles_rad, bins=bins)
counts_t0, _ = np.histogram(plot_theta0_rad, bins=bins)

fig, ax_polar = plt.subplots(figsize=(10.0, 8.0), subplot_kw={'projection': 'polar'})

# Create the bar chart (Rose Plot) for the data
width = (2 * np.pi) / N_bins
bars = ax_polar.bar(bins[:-1], counts, width=width, bottom=0.0, align='edge', zorder=2)

# Color bars based on intensity (frequency)
norm_color = mcolors.Normalize(vmin=counts.min(), vmax=counts.max())
cmap = cm.YlOrRd # From yellow (low counts) to dark red (high counts)

# With borders between adjacent bins.
#for count, bar in zip(counts, bars):
#    bar.set_facecolor(cmap(norm_color(count)))
#    bar.set_edgecolor('black')
#    bar.set_linewidth(0.3)
#    bar.set_alpha(0.85)
    
# Modified (no borders between adjacent bins)
for count, bar in zip(counts, bars):
    bar.set_facecolor(cmap(norm_color(count)))
    bar.set_edgecolor('none')  # <--- Removes black borders between bins
    bar.set_linewidth(0)       # Assigns zero thickness to borders
#    bar.set_alpha(0.85)
    bar.set_alpha(0.99)
    
# Overlay theta_0 as an outline histogram ("step")
theta_plot = np.append(bins[:-1], bins[0])
radii_t0_plot = np.append(counts_t0, counts_t0[0])

# Scale theta_0 counts so they are visible (they only have N_sim elements,
# not N_sim * 26 like the PAs). We map them to the max of the PA histogram.
scale_factor = np.max(counts) / np.max(counts_t0) if np.max(counts_t0) > 0 else 1
radii_t0_plot = radii_t0_plot * scale_factor

ax_polar.plot(theta_plot, radii_t0_plot, color='blue', linewidth=2.5, linestyle='-', zorder=3, 
              label=r'Perturbed filament PA (scaled)')

# Astronomical setup (North = 0 at top, clockwise rotation)
ax_polar.set_theta_zero_location("N")
ax_polar.set_theta_direction(-1)

#Astronomical setup (North = 0 at top, counter-clockwise rotation)
ax_polar.set_theta_zero_location("N")
# +1 --> clockwise rotation
# -1 --> counter-clockwise rotation
ax_polar.set_theta_direction(1)
# Show angles on the circumference every 10 degrees
ax_polar.set_xticks(np.radians(np.arange(0, 360, 10)), ha='center', va='center')
# Restored fontsize to 14
ax_polar.set_xticklabels([f"{angle}°" for angle in np.arange(0, 360, 10)], fontsize=14, fontweight='bold')

# Push angle labels heavily away from the outer circle rim
ax_polar.tick_params(axis='x', pad=12)

# Remove radial labels to keep it clean, leaving only the grid
ax_polar.set_yticklabels([])
ax_polar.grid(True, linestyle=':', alpha=0.9)

# Dedicated colorbar (increased pad to 0.15 to prevent touching right-hand labels)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_color)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax_polar, pad=0.08, aspect=30)
cbar.set_label('MC occurrences of perturbed PAs', fontsize=14, fontweight='bold')
cbar.ax.tick_params(labelsize=12)

# Increased title pad to 45 so top 14pt labels don't collide with the title
plt.title('Perturbed PAs distribution ($10^6$ MC Simulations)', fontsize=18, fontweight='bold', pad=45)

# Centered legend below the diagram to clear bottom labels
legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), fontsize=14)

for text in legend.get_texts():
    text.set_fontweight('bold')
    
# Save output
output_polar = 'combined_polar_distribution_MC.pdf'
fig.savefig(output_polar, format='pdf', bbox_inches='tight')

# Restored screen display and terminal prints
plt.show()
print(f"MC polar plot saved to: {output_polar}")
print("Analysis completed!")
