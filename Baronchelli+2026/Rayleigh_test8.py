import numpy as np
from scipy.stats import norm
from astropy.stats import rayleightest


# =====================================================================
# FUNCTIONS FOR TESTS ON THE ORIGINAL DATA
# =====================================================================

def esegui_test_rayleigh(nome_campione, pa_angles):
    """
    Computes the Rayleigh test for a set of Position Angles
    (axial data in the 0-180 deg range).

    One-tailed significance.
    """
    if len(pa_angles) == 0:
        return

    angoli_rad = np.radians(2 * np.array(pa_angles))
    p_value = rayleightest(angoli_rad)

    if p_value < 1e-15:
        sigma = norm.isf(1e-15)
        sigma_str = f"> {sigma:.2f}"
    else:
        sigma = norm.isf(p_value)
        sigma_str = f"{sigma:.2f}"

    print(f"--- {nome_campione} ---")
    print(f"Number of sources (n) : {len(pa_angles)}")
    print(f"p-value Rayleigh    : {p_value:.2e}")
    print(f"Significance         : {sigma_str} σ\n")


def v_test_alignment(nome_campione, pa_angles, theta_ref=115.0):
    """
    Performs the V-Test to assess the alignment of the PAs
    around the physical direction theta_ref.
    """
    n = len(pa_angles)

    if n == 0:
        return

    pa_array = np.array(pa_angles)

    alpha = np.radians(2 * (pa_array - theta_ref))
    V = np.sum(np.cos(alpha))

    u = V * np.sqrt(2.0 / n)
    p_value = norm.sf(u)

    print(f"--- Alignment Test vs Filament (θ0 = {theta_ref}°) ---")
    print(f"--- {nome_campione} ---")
    print(f"Number of sources (N) : {n}")
    print(f"V statistic          : {V:.3f}")
    print(f"p-value             : {p_value:.2e}")
    print(f"u statistic (σ)      : {u:.2f} σ\n")


# =====================================================================
# DATASETS
# =====================================================================

pa_core_north = np.array(
    [110, 105, 120, 115, 105, 100],
    dtype=float
)

pa_core_north_unc = (1.0 / np.sqrt(3.0)) * np.array(
    [10, 10, 15, 10, 15, 15],
    dtype=float
)


pa_extended_north = np.array(
    [110, 105, 120, 115, 105, 100, 35, 140, 135, 130, 60, 110, 105],
    dtype=float
)

pa_extended_north_unc = (1.0 / np.sqrt(3.0)) * np.array(
    [10, 10, 15, 10, 15, 15, 10, 10, 10, 10, 15, 10, 10],
    dtype=float
)


pa_core_south = np.array(
    [135, 125, 130, 125],
    dtype=float
)

pa_core_south_unc = (1.0 / np.sqrt(3.0)) * np.array(
    [20, 10, 10, 10],
    dtype=float
)


pa_extended_south = np.array(
    [135, 125, 130, 125, 120, 135, 25, 135, 125, 130, 130, 25, 60],
    dtype=float
)

pa_extended_south_unc = (1.0 / np.sqrt(3.0)) * np.array(
    [20, 10, 10, 10, 15, 10, 15, 20, 15, 15, 20, 10, 10],
    dtype=float
)


pa_global = np.concatenate(
    [pa_extended_north, pa_extended_south]
)

pa_global_unc = np.concatenate(
    [pa_extended_north_unc, pa_extended_south_unc]
)


# =====================================================================
# ANALYSIS OF THE ORIGINAL DATA
# =====================================================================

print(
    "RAYLEIGH TEST RESULTS "
    "(No angular windows)\n" + "=" * 55
)

esegui_test_rayleigh(
    "Scenario A (Core North)",
    pa_core_north
)

esegui_test_rayleigh(
    "Scenario B (Extended North)",
    pa_extended_north
)

esegui_test_rayleigh(
    "Scenario C (Core South)",
    pa_core_south
)

esegui_test_rayleigh(
    "Scenario D (Extended South)",
    pa_extended_south
)

esegui_test_rayleigh(
    "Scenario E (Combined Global)",
    pa_global
)


print(
    "V-TEST / Moore-Rayleigh TEST RESULTS "
    "(No angular windows)\n" + "=" * 55
)

v_test_alignment(
    "Scenario A (Core North)",
    pa_core_north,
    theta_ref=115.0
)

v_test_alignment(
    "Scenario B (Extended North)",
    pa_extended_north,
    theta_ref=115.0
)

v_test_alignment(
    "Scenario C (Core South)",
    pa_core_south,
    theta_ref=115.0
)

v_test_alignment(
    "Scenario D (Extended South)",
    pa_extended_south,
    theta_ref=115.0
)

v_test_alignment(
    "Scenario E (Combined Global)",
    pa_global,
    theta_ref=115.0
)


# =====================================================================
# GLOBAL PARAMETERS
# =====================================================================

pa_global_array = np.asarray(pa_global, dtype=float)
pa_global_unc_array = np.asarray(pa_global_unc, dtype=float)

theta_ref = 115.0

# Nuova incertezza richiesta per theta_0:
theta_ref_unc = (1.0 / np.sqrt(3.0)) * 15.0

n_tot = len(pa_global_array)


# =====================================================================
# IDENTIFICATION OF THE 5 MISALIGNED SOURCES
# =====================================================================
#
# The condition is exactly the same as in the original simulation.
#
# < 100 deg  or  > 140 deg
#
# Therefore:
#   35, 60, 25, 25, 60 deg
#
# while 140 deg is NOT included.
# =====================================================================

mask_out_of_range = (
    (pa_global_array < 100.0) |
    (pa_global_array > 140.0)
)

num_outliers = np.sum(mask_out_of_range)

print("\n" + "=" * 70)
print(" MISALIGNED SOURCES USED IN THE RANDOMIZATION")
print("=" * 70)

print(
    "Misaligned PAs :",
    pa_global_array[mask_out_of_range]
)

print(
    f"Number of misaligned sources : {num_outliers}"
)

print()


# =====================================================================
# FUNCTION TO SUMMARIZE A MONTE CARLO DISTRIBUTION
# =====================================================================

def mc_summary(x):
    median = np.median(x)
    p16 = np.percentile(x, 16)
    p84 = np.percentile(x, 84)

    return median, p16, p84


def se_median(x):
    """
    Numerical standard error of the median estimated from the width
    16-84 percentile.

    ATTENZIONE:
    this does NOT represent the propagated physical uncertainty.
    """
    return (
        (np.percentile(x, 84) - np.percentile(x, 16))
        / (2.0 * np.sqrt(len(x)))
    )


# =====================================================================
# ==============================================================
# MC 1
# RANDOMIZATION OF THE 5 MISALIGNED SOURCES
# ==============================================================
#
# THIS IS THE ORIGINAL SIMULATION.
#
# The following are NOT perturbed:
#   - the aligned PAs
#   - their uncertainties
#   - theta_0
#
# The 5 misaligned sources are instead distributed as:
#
#    90/120 -> U(0, 90)
#    30/120 -> U(150, 180)
#
# This is kept separate specifically to quantify the effect of the
# randomization of the misaligned sources alone.
# =====================================================================

print("\n" + "=" * 70)
print(" MC 1: RANDOMIZATION OF THE 5 MISALIGNED SOURCES")
print("=" * 70 + "\n")


n_iter_random = 100_000

rng = np.random.default_rng(12345)

ray_p_vals_random = np.empty(n_iter_random)
v_stats_random = np.empty(n_iter_random)
v_p_vals_random = np.empty(n_iter_random)


for i in range(n_iter_random):

    sim_pa = pa_global_array.copy().astype(float)

    random_angles = []

    for _ in range(num_outliers):

        if rng.random() < (90.0 / 120.0):
            random_angles.append(
                rng.uniform(0.0, 90.0)
            )
        else:
            random_angles.append(
                rng.uniform(150.0, 180.0)
            )

    sim_pa[mask_out_of_range] = random_angles


    # ---------------------------------------------------------------
    # Rayleigh
    # ---------------------------------------------------------------

    angoli_rad = np.radians(2.0 * sim_pa)

    rp = rayleightest(angoli_rad)

    ray_p_vals_random[i] = rp


    # ---------------------------------------------------------------
    # V-Test
    # ---------------------------------------------------------------

    alpha = np.radians(
        2.0 * (sim_pa - theta_ref)
    )

    V = np.sum(np.cos(alpha))

    u = V * np.sqrt(2.0 / n_tot)

    vp = norm.sf(u)

    v_stats_random[i] = V
    v_p_vals_random[i] = vp


# =====================================================================
# MC 1 RESULTS
# =====================================================================

ray_sigma_random = norm.isf(
    np.maximum(ray_p_vals_random, 1e-15)
)

v_sigma_random = norm.isf(
    np.maximum(v_p_vals_random, 1e-15)
)


median_ray_p_random, ray_p16_random, ray_p84_random = mc_summary(
    ray_p_vals_random
)

median_ray_sigma_random, ray_s16_random, ray_s84_random = mc_summary(
    ray_sigma_random
)

median_v_stat_random, v_stat16_random, v_stat84_random = mc_summary(
    v_stats_random
)

median_v_p_random, v_p16_random, v_p84_random = mc_summary(
    v_p_vals_random
)

median_v_sigma_random, v_s16_random, v_s84_random = mc_summary(
    v_sigma_random
)


print(
    f"Number of MC iterations       : {n_iter_random:,}"
)

print("\nRAYLEIGH TEST")
print(
    f"Median p-value          : {median_ray_p_random:.3e}"
)
print(
    f"16-84% interval         : "
    f"[{ray_p16_random:.3e}, {ray_p84_random:.3e}]"
)
print(
    f"Median SE               : "
    f"{se_median(ray_p_vals_random):.3e}"
)
print(
    f"Median significance     : "
    f"{median_ray_sigma_random:.3f} σ"
)
print(
    f"16-84% interval         : "
    f"[{ray_s16_random:.3f}, {ray_s84_random:.3f}] σ"
)

print("\nV-TEST")
print(
    f"Median V statistic     : "
    f"{median_v_stat_random:.4f}"
)
print(
    f"16-84% interval         : "
    f"[{v_stat16_random:.4f}, {v_stat84_random:.4f}]"
)
print(
    f"Median SE               : "
    f"{se_median(v_stats_random):.4f}"
)

print(
    f"Median p-value          : "
    f"{median_v_p_random:.3e}"
)
print(
    f"16-84% interval         : "
    f"[{v_p16_random:.3e}, {v_p84_random:.3e}]"
)

print(
    f"Median significance     : "
    f"{median_v_sigma_random:.3f} σ"
)
print(
    f"16-84% interval         : "
    f"[{v_s16_random:.3f}, {v_s84_random:.3f}] σ"
)


# =====================================================================
# ==============================================================
# FUNZIONE RAYLEIGH VETTORIALIZZATA
# ==============================================================
#
# Reproduces the formula used by astropy.stats.rayleightest.
#
# For the current case n = 26 < 50, so the
# finite-sample correction is included.
# =====================================================================

def rayleigh_pvalue_vectorized(angles_rad):
    """
    Vectorized calculation of the Rayleigh p-value.

    angles_rad:
        array of shape (N_MC, N_sources)

    returns:
        p-value array of shape (N_MC,)
    """

    n = angles_rad.shape[1]

    C = np.sum(
        np.cos(angles_rad),
        axis=1
    )

    S = np.sum(
        np.sin(angles_rad),
        axis=1
    )

    R2 = C**2 + S**2

    # r_bar^2 = R^2 / n^2
    # z = n * r_bar^2 = R^2 / n
    z = R2 / n

    # n = 26, quindi n < 50
    correction = (
        1.0
        + (2.0 * z - z**2) / (4.0 * n)
        - (
            24.0 * z
            - 132.0 * z**2
            + 76.0 * z**3
            - 9.0 * z**4
        ) / (288.0 * n**2)
    )

    p = np.exp(-z) * correction

    return np.clip(
        p,
        0.0,
        1.0
    )


# =====================================================================
# ==============================================================
# MC 2
# FULL UNCERTAINTY PROPAGATION
# ==============================================================
#
# 10^6 iterations.
#
# For EACH iteration:
#
# 1) each PA has its own perturbation:
#
#       PA_i,MC = PA_i + N(0, sigma_i)
#
# 2) the 5 misaligned sources are randomized as in the
#    original MC:
#
#       90/120 -> U(0,90)
#       30/120 -> U(150,180)
#
#    and after this randomization, their
#    individual uncertainty is also applied.
#
# 3) theta_0 is perturbed:
#
#       theta_0,MC ~ N(115, 15/sqrt(3))
#
# 4) the same theta_0,MC is applied to all sources
#    in that iteration.
# =====================================================================

print("\n" + "=" * 70)
print(" MC 2: PROPAGAZIONE COMPLETA - 10^6 ITERAZIONI")
print("=" * 70 + "\n")


n_iter_unc = 1_000_000

batch_size = 50_000

ray_p_vals = np.empty(n_iter_unc)
ray_sigmas = np.empty(n_iter_unc)

v_stats = np.empty(n_iter_unc)
v_p_vals = np.empty(n_iter_unc)
v_sigmas = np.empty(n_iter_unc)

theta_ref_mc_all = np.empty(n_iter_unc)


print(
    f"Total number of sources      : {n_tot}"
)

print(
    f"Number of randomized sources : {num_outliers}"
)

print(
    f"Nominal theta_0             : {theta_ref:.3f}°"
)

print(
    f"theta_0 uncertainty          : {theta_ref_unc:.3f}°"
)

print(
    f"Number of iterations         : {n_iter_unc:,}"
)

print()


counter = 0


while counter < n_iter_unc:

    current_batch = min(
        batch_size,
        n_iter_unc - counter
    )


    # ===============================================================
    # 1. PA PERTURBATION
    # ===============================================================

    sim_pa = (
        pa_global_array[None, :]
        + rng.normal(
            loc=0.0,
            scale=pa_global_unc_array[None, :],
            size=(current_batch, n_tot)
        )
    )


    # ===============================================================
    # 2. RANDOMIZATION OF THE 5 MISALIGNED SOURCES
    # ===============================================================

    if num_outliers > 0:

        choose_low = (
            rng.random(
                size=(current_batch, num_outliers)
            )
            < (90.0 / 120.0)
        )


        random_outliers = np.where(
            choose_low,

            rng.uniform(
                0.0,
                90.0,
                size=(current_batch, num_outliers)
            ),

            rng.uniform(
                150.0,
                180.0,
                size=(current_batch, num_outliers)
            )
        )


        # Replace the outlier with its random PA
        sim_pa[:, mask_out_of_range] = random_outliers


        # After randomization, apply the corresponding
        # individual PA uncertainty
        sim_pa[:, mask_out_of_range] += rng.normal(
            loc=0.0,
            scale=pa_global_unc_array[
                mask_out_of_range
            ][None, :],
            size=(current_batch, num_outliers)
        )


    # ===============================================================
    # 3. THETA_0 PERTURBATION
    # ===============================================================

    theta_ref_mc = rng.normal(
        loc=theta_ref,
        scale=theta_ref_unc,
        size=current_batch
    )

    theta_ref_mc_all[
        counter:counter + current_batch
    ] = theta_ref_mc


    # ===============================================================
    # 4. RAYLEIGH TEST
    # ===============================================================

    angles_rad = np.radians(
        2.0 * sim_pa
    )

    rp = rayleigh_pvalue_vectorized(
        angles_rad
    )

    ray_p_vals[
        counter:counter + current_batch
    ] = rp

    ray_sigmas[
        counter:counter + current_batch
    ] = norm.isf(
        np.maximum(
            rp,
            1e-15
        )
    )


    # ===============================================================
    # 5. V-TEST
    # ===============================================================

    alpha = np.radians(
        2.0 * (
            sim_pa
            - theta_ref_mc[:, None]
        )
    )

    V = np.sum(
        np.cos(alpha),
        axis=1
    )

    u = (
        V
        * np.sqrt(2.0 / n_tot)
    )

    vp = norm.sf(u)


    v_stats[
        counter:counter + current_batch
    ] = V

    v_p_vals[
        counter:counter + current_batch
    ] = vp

    v_sigmas[
        counter:counter + current_batch
    ] = norm.isf(
        np.maximum(
            vp,
            1e-15
        )
    )


    # ===============================================================
    # PROGRESS
    # ===============================================================

    counter += current_batch

    if (
        counter % (10 * batch_size) == 0
        or counter == n_iter_unc
    ):
        print(
            f"Completed {counter:,} / "
            f"{n_iter_unc:,} "
            f"({100.0 * counter / n_iter_unc:.1f}%)"
        )


# =====================================================================
# MC 2 RESULTS
# =====================================================================

# ---------------------------------------------------------------------
# theta_0
# ---------------------------------------------------------------------

median_theta, theta16, theta84 = mc_summary(
    theta_ref_mc_all
)


# ---------------------------------------------------------------------
# Rayleigh
# ---------------------------------------------------------------------

median_ray_p, ray_p16, ray_p84 = mc_summary(
    ray_p_vals
)

median_ray_sigma, ray_s16, ray_s84 = mc_summary(
    ray_sigmas
)


# ---------------------------------------------------------------------
# V-Test
# ---------------------------------------------------------------------

median_v_stat, v_stat16, v_stat84 = mc_summary(
    v_stats
)

median_v_p, v_p16, v_p84 = mc_summary(
    v_p_vals
)

median_v_sigma, v_s16, v_s84 = mc_summary(
    v_sigmas
)


# =====================================================================
# MC 2 OUTPUT
# =====================================================================

print("\n" + "=" * 70)
print(" MC 2 RESULTS - PROPAGAZIONE COMPLETA")
print("=" * 70)

print("\nPERTURBED THETA_0")

print(
    f"Nominal value          : "
    f"{theta_ref:.3f}°"
)

print(
    f"Input uncertainty      : "
    f"{theta_ref_unc:.3f}°"
)

print(
    f"MC median              : "
    f"{median_theta:.3f}°"
)

print(
    f"Interval 16-84%     : "
    f"[{theta16:.3f}, {theta84:.3f}]°"
)


print("\n" + "-" * 70)

print("RAYLEIGH TEST")

print(
    f"Median p-value          : "
    f"{median_ray_p:.3e}"
)

print(
    f"16-84% interval         : "
    f"[{ray_p16:.3e}, {ray_p84:.3e}]"
)

print(
    f"Median SE               : "
    f"{se_median(ray_p_vals):.3e}"
)

print(
    f"Median significance     : "
    f"{median_ray_sigma:.3f} σ"
)

print(
    f"16-84% interval         : "
    f"[{ray_s16:.3f}, {ray_s84:.3f}] σ"
)

print(
    f"Median SE               : "
    f"{se_median(ray_sigmas):.4f} σ"
)


print("\n" + "-" * 70)

print(
    f"V-TEST "
    f"(theta_0 nominale = {theta_ref:.1f}°)"
)

print(
    f"Median V statistic     : "
    f"{median_v_stat:.4f}"
)

print(
    f"16-84% interval         : "
    f"[{v_stat16:.4f}, {v_stat84:.4f}]"
)

print(
    f"Median SE               : "
    f"{se_median(v_stats):.4f}"
)

print(
    f"Median p-value          : "
    f"{median_v_p:.3e}"
)

print(
    f"16-84% interval         : "
    f"[{v_p16:.3e}, {v_p84:.3e}]"
)

print(
    f"Median SE               : "
    f"{se_median(v_p_vals):.3e}"
)

print(
    f"Median significance     : "
    f"{median_v_sigma:.3f} σ"
)

print(
    f"16-84% interval         : "
    f"[{v_s16:.3f}, {v_s84:.3f}] σ"
)

print(
    f"Median SE               : "
    f"{se_median(v_sigmas):.4f} σ"
)


print("\n" + "=" * 70)
print(" END OF SIMULATIONS")
print("=" * 70)
