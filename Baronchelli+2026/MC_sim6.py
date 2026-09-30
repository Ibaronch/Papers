import numpy as np
from scipy.special import comb
from scipy.stats import binom, norm
import time
from pdb import set_trace as stop 

def pvalue_to_sigma(p, p_err=0):
    """
    Converts a p-value to sigma and propagates the uncertainty.
    Uncertainty on sigma (delta_Z) = delta_p / pdf(Z)
    """
    if p <= 0: 
        return float('inf'), 0.0
    if p >= 1: 
        return 0.0, 0.0
    
    z_score = norm.isf(p)
    
    if p_err > 0:
        # Propagation of error: sigma_err = p_err / normal_pdf(z_score)
        z_err = p_err / norm.pdf(z_score)
    else:
        z_err = 0.0
        
    return z_score, z_err

def compute_fixed_theoretical(n, k, p):
    """Computes cumulative binomial probability P(>=k) for a fixed direction."""
    return 1 - binom.cdf(k - 1, n, p)

def compute_any_theoretical(n, k, p):
    """Rare-event Bonferroni approximation to the scan statistic(Naus/Glaz)."""
    val = n * comb(n - 1, k - 1) * (p**(k - 1)) * ((1 - p)**(n - k))
    return min(val, 1.0)

def run_optimized_simulation(n, k, delta_theta, iterations=1_000_000):
    """
    Monte Carlo simulation with block-vectorization.
    Returns frequencies and Poisson-based uncertainties.
    """
    block_size = 1_000_000 
    successes_fixed = 0
    successes_any = 0

    # Cicle over data blocks for RAM optimization
    for _ in range(iterations // block_size):
        
        data = np.random.uniform(0, 180, (block_size, n))
        
        # Fixed window test
        n_in_fixed = np.sum(data <= delta_theta, axis=1)
        successes_fixed += np.sum(n_in_fixed >= k)
        
        # Any window test (Scan Statistic)
        data.sort(axis=1)
        extended_data = np.hstack([data, data + 180])
        window_widths = extended_data[:, k-1 : n+k-1] - extended_data[:, 0 : n]
        found_any = np.any(window_widths <= delta_theta, axis=1)
        successes_any += np.sum(found_any)

    freq_f = successes_fixed / iterations
    err_f = np.sqrt(successes_fixed) / iterations if successes_fixed > 0 else 0
    
    freq_a = successes_any / iterations
    err_a = np.sqrt(successes_any) / iterations if successes_any > 0 else 0
    
    return (freq_f, err_f), (freq_a, err_a)

def multiply_pvalues(p1, err1, p2, err2):
    """
    Multiplies two independent p-values and propagates the error.
    (err_tot / p_tot)^2 = (err1 / p1)^2 + (err2 / p2)^2
    """
    if p1 <= 0 or p2 <= 0:
        return 0.0, 0.0
    
    p_tot = p1 * p2
    err_tot = p_tot * np.sqrt((err1 / p1)**2 + (err2 / p2)**2)
    return p_tot, err_tot

# --- CONFIGURATION ---
N_TOTAL_ITER = 5_000_000_000 

scenarios = {
    "A (Core North)":                  {"n": 6,  "k": 6,  "dt": 20},
    "B (Extended North)":              {"n": 13, "k": 11, "dt": 40},
    "C1 (Core South - ok 4 any)":      {"n": 5,  "k": 4,  "dt": 10},
    "C2 (Core South - ok 4 fixed)":    {"n": 5,  "k": 4,  "dt": 20},
    "D1 (Extended South- ok 4 any))":  {"n": 13, "k": 10, "dt": 15},
    "D2 (Extended South - ok 4 fixed)":{"n": 13, "k": 10, "dt": 20},
    "E (Combined N/S)":                {"n": 26, "k": 21, "dt": 40},
}

# Column size
col_name = 34
col_type = 6
col_pval = 9
col_sig  = 9
col_mc_p = 21
col_mc_s = 23

header = (f"{'SCENARIO':<{col_name}} | {'TYPE':<{col_type}} | "
          f"{'TH P-VAL':<{col_pval}} | {'TH SIG':<{col_sig}} | "
          f"{'MC P-VAL (± ERR)':<{col_mc_p}} | {'MC SIG (± ERR)':<{col_mc_s}}")

print(f"Starting simulation with {N_TOTAL_ITER:.0e} iterations...")
print(header)
print("-" * len(header))

saved_results = {}

for name, params in scenarios.items():
    start_time = time.time()
    p_val = params['dt'] / 180.0
    
    # Theoretical results
    th_p_f = compute_fixed_theoretical(params['n'], params['k'], p_val)
    th_p_a = compute_any_theoretical(params['n'], params['k'], p_val)
    
    th_s_f, _ = pvalue_to_sigma(th_p_f)
    th_s_a, _ = pvalue_to_sigma(th_p_a)
    
    # Monte Carlo results
    (mc_p_f, mc_e_f), (mc_p_a, mc_e_a) = run_optimized_simulation(params['n'], params['k'], params['dt'], N_TOTAL_ITER)
    
    # Sigma with propagated error
    mc_s_f, mc_se_f = pvalue_to_sigma(mc_p_f, mc_e_f)
    mc_s_a, mc_se_a = pvalue_to_sigma(mc_p_a, mc_e_a)
    
    # Memorize results for Global-joint scenario computation
    if name.startswith("B (") or name.startswith("D1 (") or name.startswith("D2 ("):
        saved_results[name] = {
            'th_p_f': th_p_f, 'th_p_a': th_p_a,
            'mc_p_f': mc_p_f, 'mc_e_f': mc_e_f,
            'mc_p_a': mc_p_a, 'mc_e_a': mc_e_a
        }

    # strings for columns alignment
    th_p_f_str = f"{th_p_f:.2e}"
    th_s_f_str = f"{th_s_f:.2f}σ"
    mc_p_f_str = f"{mc_p_f:.2e} ± {mc_e_f:.2e}"
    sig_f_str = f"{mc_s_f:.4f} ± {mc_se_f:.2e}" if mc_p_f > 0 else "N/A"

    th_p_a_str = f"{th_p_a:.2e}"
    th_s_a_str = f"{th_s_a:.2f}σ"
    mc_p_a_str = f"{mc_p_a:.2e} ± {mc_e_a:.2e}"
    sig_a_str = f"{mc_s_a:.4f} ± {mc_se_a:.2e}" if mc_p_a > 0 else "N/A"

    # Print line "Fixed". Apply to every corresponding column
    print(f"{name:<{col_name}} | {'Fixed':<{col_type}} | "
          f"{th_p_f_str:<{col_pval}} | {th_s_f_str:<{col_sig}} | "
          f"{mc_p_f_str:<{col_mc_p}} | {sig_f_str:<{col_mc_s}}")
          
    # print line "Any"
    print(f"{'':<{col_name}} | {'Any':<{col_type}} | "
          f"{th_p_a_str:<{col_pval}} | {th_s_a_str:<{col_sig}} | "
          f"{mc_p_a_str:<{col_mc_p}} | {sig_a_str:<{col_mc_s}}")
          
    print(f"[Execution time: {time.time() - start_time:.1f}s]")
    print("-" * len(header))

# --- Compute combinations B x D1 and B x D2) ---
print("\n" + "=" * len(header))
print(f"{'COMBINED SCENARIOS (B x D)':^{len(header)}}")
print("=" * len(header))

key_B = "B (Extended North)"
keys_D = ["D1 (Extended South- ok 4 any))", "D2 (Extended South - ok 4 fixed)"]

if key_B in saved_results:
    res_B = saved_results[key_B]
    
    for key_D in keys_D:
        if key_D in saved_results:
            res_D = saved_results[key_D]
            
            # --- multiplication "FIXED" ---
            th_p_f_comb = res_B['th_p_f'] * res_D['th_p_f']
            th_s_f_comb, _ = pvalue_to_sigma(th_p_f_comb)
            
            mc_p_f_comb, mc_e_f_comb = multiply_pvalues(
                res_B['mc_p_f'], res_B['mc_e_f'], 
                res_D['mc_p_f'], res_D['mc_e_f']
            )
            mc_s_f_comb, mc_se_f_comb = pvalue_to_sigma(mc_p_f_comb, mc_e_f_comb)
            
            # --- Moltiplicazione ANY ---
            th_p_a_comb = res_B['th_p_a'] * res_D['th_p_a']
            th_s_a_comb, _ = pvalue_to_sigma(th_p_a_comb)
            
            mc_p_a_comb, mc_e_a_comb = multiply_pvalues(
                res_B['mc_p_a'], res_B['mc_e_a'], 
                res_D['mc_p_a'], res_D['mc_e_a']
            )
            mc_s_a_comb, mc_se_a_comb = pvalue_to_sigma(mc_p_a_comb, mc_e_a_comb)
            
            # Strings format
            th_p_f_comb_str = f"{th_p_f_comb:.2e}"
            th_s_f_comb_str = f"{th_s_f_comb:.2f}σ"
            mc_p_f_comb_str = f"{mc_p_f_comb:.2e} ± {mc_e_f_comb:.2e}"
            sig_f_str_comb = f"{mc_s_f_comb:.4f} ± {mc_se_f_comb:.2e}" if mc_p_f_comb > 0 else "N/A"
            
            th_p_a_comb_str = f"{th_p_a_comb:.2e}"
            th_s_a_comb_str = f"{th_s_a_comb:.2f}σ"
            mc_p_a_comb_str = f"{mc_p_a_comb:.2e} ± {mc_e_a_comb:.2e}"
            sig_a_str_comb = f"{mc_s_a_comb:.4f} ± {mc_se_a_comb:.2e}" if mc_p_a_comb > 0 else "N/A"
            
            d_label = "D1" if "D1" in key_D else "D2"
            combo_name = f"B x {d_label}"
            
            if d_label == "D1":
            
                print(f"{combo_name:<{col_name}} | {'Any':<{col_type}} | "
                      f"{th_p_a_comb_str:<{col_pval}} | {th_s_a_comb_str:<{col_sig}} | "
                      f"{mc_p_a_comb_str:<{col_mc_p}} | {sig_a_str_comb:<{col_mc_s}}")
                
            if d_label == "D2":

                print(f"{combo_name:<{col_name}} | {'Fixed':<{col_type}} | "
                      f"{th_p_f_comb_str:<{col_pval}} | {th_s_f_comb_str:<{col_sig}} | "
                      f"{mc_p_f_comb_str:<{col_mc_p}} | {sig_f_str_comb:<{col_mc_s}}")

            print("-" * len(header))
