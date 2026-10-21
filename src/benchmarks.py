"""
benchmarks.py
=============
Генерация 4 фигур для статьи (Figure 1–4).

Запускать через scripts/run_all.py (там настроен sys.path).
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend: только savefig, без show()
import matplotlib.pyplot as plt
from pathlib import Path

from bs_engine import bs_call_price, bs_call_delta, CrankNicolsonBS
from bs_monte_carlo import monte_carlo_call_price, DeltaHedgingSimulator


# --- Общий стиль ---
plt.style.use("seaborn-v0_8-whitegrid"
              if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.titlesize": 14,
    "savefig.dpi": 300,
    "figure.autolayout": True,
})

# --- Параметры Base Case ---
S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2


def _save(fig, outdir, name):
    path = Path(outdir) / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved: {path}")


# =====================================================================
# FIGURE 1: Price & Delta Profiles
# =====================================================================
def figure1(outdir):
    print("Generating Figure 1: Price and Delta Profiles...")

    fdm = CrankNicolsonBS(K=K, T=T, r=r, sigma=sigma, M=200, N_t=200)
    V_fdm, Delta_fdm = fdm.solve()
    S_grid = fdm.S_grid

    V_exact = bs_call_price(S_grid, K, T, r, sigma)
    Delta_exact = bs_call_delta(S_grid, K, T, r, sigma)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(S_grid, V_exact, "k-", linewidth=2, label="Exact Black-Scholes")
    ax1.plot(S_grid[::5], V_fdm[::5], "ro", markersize=4,
             label="FDM Crank-Nicolson")
    ax1.axvline(K, color="gray", linestyle="--", alpha=0.7,
                label=f"Strike K={K}")
    ax1.set_xlabel("Spot Price S")
    ax1.set_ylabel("Option Price V(S, 0)")
    ax1.set_title("Option Price Profile V(S)")
    ax1.set_xlim(50, 150)
    ax1.legend()

    ax2.plot(S_grid, Delta_exact, "k-", linewidth=2, label="Exact Delta")
    ax2.plot(S_grid[::5], Delta_fdm[::5], "bs", markersize=3,
             label="FDM Delta")
    ax2.axvline(K, color="gray", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Spot Price S")
    ax2.set_ylabel("Delta dV/dS")
    ax2.set_title(r"Delta Profile $\Delta(S)$")
    ax2.set_xlim(50, 150)
    ax2.legend()

    fig.suptitle("Figure 1: Comparison of Crank-Nicolson FDM "
                 "and Analytical Solution", fontweight="bold")
    _save(fig, outdir, "figure1_profiles.png")


# =====================================================================
# FIGURE 2: Convergence Rates
# =====================================================================
def figure2(outdir):
    print("Generating Figure 2: Convergence Rates...")

    exact_price = bs_call_price(S0, K, T, r, sigma)

    grid_steps = np.array([20, 40, 80, 160, 320, 640])
    fdm_errors = []
    for M in grid_steps:
        fdm_i = CrankNicolsonBS(K=K, T=T, r=r, sigma=sigma, M=M, N_t=M)
        V_i, _ = fdm_i.solve()
        price_i = np.interp(S0, fdm_i.S_grid, V_i)
        fdm_errors.append(abs(price_i - exact_price))
    fdm_errors = np.array(fdm_errors)

    mc_samples = np.array([1000, 5000, 25000, 100000, 500000])
    mc_std_errors, mc_anti_errors = [], []
    for N in mc_samples:
        _, err_s, _ = monte_carlo_call_price(
            S0, K, T, r, sigma, N_sim=N, use_antithetic=False)
        _, err_a, _ = monte_carlo_call_price(
            S0, K, T, r, sigma, N_sim=N, use_antithetic=True)
        mc_std_errors.append(err_s)
        mc_anti_errors.append(err_a)
    mc_std_errors = np.array(mc_std_errors)
    mc_anti_errors = np.array(mc_anti_errors)

    slope_fdm  = np.polyfit(np.log(grid_steps), np.log(fdm_errors), 1)[0]
    slope_mc   = np.polyfit(np.log(mc_samples), np.log(mc_std_errors), 1)[0]
    slope_anti = np.polyfit(np.log(mc_samples), np.log(mc_anti_errors), 1)[0]
    var_ratio  = np.mean((mc_std_errors / mc_anti_errors) ** 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.loglog(grid_steps, fdm_errors, "o-", color="crimson", linewidth=2,
               label=rf"FDM Absolute Error (slope $\approx$ {slope_fdm:.2f})")
    ref_fdm = fdm_errors[0] * (grid_steps[0] / grid_steps) ** 2
    ax1.loglog(grid_steps, ref_fdm, "k--", alpha=0.6,
               label=r"Theoretical $O(M^{-2})$")
    ax1.set_xlabel(r"Grid Steps $M$ (where $N_t = M$)")
    ax1.set_ylabel("Absolute Error")
    ax1.set_title("FDM Crank-Nicolson Convergence")
    ax1.legend()

    ax2.loglog(mc_samples, mc_std_errors, "s-", color="navy", linewidth=2,
               label=rf"Standard MC (slope $\approx$ {slope_mc:.2f})")
    ax2.loglog(mc_samples, mc_anti_errors, "d-", color="teal", linewidth=2,
               label=rf"Antithetic MC (slope $\approx$ {slope_anti:.2f})")
    ref_mc = mc_std_errors[0] * np.sqrt(mc_samples[0] / mc_samples)
    ax2.loglog(mc_samples, ref_mc, "k--", alpha=0.6,
               label=r"Theoretical $O(N^{-1/2})$")
    ax2.set_xlabel(r"Number of Simulations $N$ "
                   r"(antithetic: $N/2$ paired draws)")
    ax2.set_ylabel("Standard Error")
    ax2.set_title(rf"Monte Carlo Convergence — Variance reduction: "
                  rf"{var_ratio:.2f}$\times$")
    ax2.legend()

    fig.suptitle("Figure 2: Empirical Convergence Rates (Log-Log Scale)",
                 fontweight="bold")
    _save(fig, outdir, "figure2_convergence.png")

    print(f"  slopes: FDM={slope_fdm:.3f}, MC={slope_mc:.3f}, "
          f"Anti={slope_anti:.3f} | Variance reduction: {var_ratio:.2f}x")


# =====================================================================
# FIGURE 3: Pareto Frontier
# =====================================================================
def figure3(outdir):
    print("Generating Figure 3: Pareto Frontier...")

    exact_price = bs_call_price(S0, K, T, r, sigma)

    fdm_times, fdm_errs = [], []
    for M in [20, 50, 100, 200, 400, 800]:
        t0 = time.perf_counter()
        fdm_i = CrankNicolsonBS(K=K, T=T, r=r, sigma=sigma, M=M, N_t=M)
        V_i, _ = fdm_i.solve()
        t_el = (time.perf_counter() - t0) * 1000
        err = abs(np.interp(S0, fdm_i.S_grid, V_i) - exact_price)
        fdm_times.append(t_el)
        fdm_errs.append(err)

    mc_times, mc_errs, anti_times, anti_errs = [], [], [], []
    for N in [5000, 20000, 100000, 500000, 2000000]:
        _, err_s, t_s = monte_carlo_call_price(
            S0, K, T, r, sigma, N_sim=N, use_antithetic=False)
        _, err_a, t_a = monte_carlo_call_price(
            S0, K, T, r, sigma, N_sim=N, use_antithetic=True)
        mc_times.append(t_s * 1000);    mc_errs.append(err_s)
        anti_times.append(t_a * 1000);  anti_errs.append(err_a)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(fdm_times, fdm_errs, "o-", color="crimson",
              linewidth=2, markersize=7, label="FDM Crank-Nicolson")
    ax.loglog(mc_times, mc_errs, "s-", color="navy",
              linewidth=2, markersize=7, label="Standard Monte Carlo")
    ax.loglog(anti_times, anti_errs, "d-", color="teal",
              linewidth=2, markersize=7, label="Antithetic Monte Carlo")

    ax.set_xlabel("CPU Execution Time (ms)")
    ax.set_ylabel("Error (Abs Error / StdErr)")
    ax.set_title("Figure 3: Computational Efficiency Pareto Frontier",
                 fontweight="bold")

    ax.annotate("", xy=(0.05, 0.05), xytext=(0.20, 0.20),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="black",
                                lw=1.5, alpha=0.6))
    ax.text(0.07, 0.07, "better", transform=ax.transAxes,
            fontsize=10, alpha=0.6, style="italic")

    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    _save(fig, outdir, "figure3_pareto.png")


# =====================================================================
# FIGURE 4: Delta Hedging P&L
# =====================================================================
"""
def figure4(outdir, N_paths=10000):
    print("Generating Figure 4: Delta Hedging P&L Distributions...")

    sim = DeltaHedgingSimulator(S0, K, T, r, sigma)

    pnl_monthly, _ = sim.run_simulation(N_steps=12,  N_paths=N_paths, seed=42)
    pnl_weekly,  _ = sim.run_simulation(N_steps=52,  N_paths=N_paths, seed=42)
    pnl_daily,   _ = sim.run_simulation(N_steps=252, N_paths=N_paths, seed=42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.hist(pnl_monthly, bins=50, density=True, alpha=0.5, color="red",
             label=f"Monthly (N=12), Std={np.std(pnl_monthly):.2f}")
    ax1.hist(pnl_weekly, bins=50, density=True, alpha=0.5, color="orange",
             label=f"Weekly (N=52), Std={np.std(pnl_weekly):.2f}")
    ax1.hist(pnl_daily, bins=50, density=True, alpha=0.6, color="green",
             label=f"Daily (N=252), Std={np.std(pnl_daily):.2f}")
    ax1.axvline(0, color="black", linestyle="--", linewidth=1.5,
                label="Zero P&L (Ideal)")
    ax1.set_xlabel("Terminal P&L ($)")
    ax1.set_ylabel("Probability Density")
    ax1.set_title("P&L Distribution Across Rehedging Frequencies")
    ax1.set_xlim(-6, 6)
    ax1.legend()

    step_range = np.array([4, 12, 26, 52, 100, 252, 500])
    std_pnl_list = []
    for n_st in step_range:
        pnl_i, _ = sim.run_simulation(N_steps=n_st, N_paths=3000, seed=42)
        std_pnl_list.append(np.std(pnl_i))
    std_pnl_list = np.array(std_pnl_list)

    ax2.plot(step_range, std_pnl_list, "bo-", linewidth=2,
             label="Empirical Std(P&L)")
    ref_hedging = std_pnl_list[0] * np.sqrt(step_range[0] / step_range)
    ax2.plot(step_range, ref_hedging, "k--", alpha=0.7,
             label=r"Theoretical $O(1/\sqrt{N_{steps}})$")

    slope_hedge = np.polyfit(np.log(step_range), np.log(std_pnl_list), 1)[0]
    ax2.set_xlabel(r"Number of Rehedging Steps $N_{steps}$")
    ax2.set_ylabel(r"Std Dev of Hedging Error $\sigma$(P&L)")
    ax2.set_title(rf"Hedging Risk Decay $\sigma$(P&L) $\to$ 0 "
                  rf"(slope $\approx$ {slope_hedge:.2f})")
    ax2.legend()

    fig.suptitle("Figure 4: Delta Hedging Performance "
                 "and Hedging Error Breakdown", fontweight="bold")
    _save(fig, outdir, "figure4_hedging_pnl.png")

    print(f"  hedging decay slope = {slope_hedge:.3f} (theoretical -0.5)")
"""

def figure4b_fdm_vs_analytic(outdir, N_paths=5000):
    """
    Сравнение P&L с аналитической и FDM-дельтой.
    """
    print("Generating Figure 4b: FDM vs Analytic Hedging...")
    from fdm_delta import make_fdm_delta_func

    sim = DeltaHedgingSimulator(S0, K, T, r, sigma)

    # Три частоты рехеджа
    frequencies = [12, 52, 252]
    results = {"analytic": [], "fdm": []}

    delta_fdm = make_fdm_delta_func(K, T, r, sigma, M=200, N_t=200)

    for N_steps in frequencies:
        pnl_a, _ = sim.run_simulation(N_steps=N_steps, N_paths=N_paths, seed=42)
        pnl_f, _ = sim.run_simulation(N_steps=N_steps, N_paths=N_paths,
                                       seed=42, delta_func=delta_fdm)
        results["analytic"].append(np.std(pnl_a))
        results["fdm"].append(np.std(pnl_f))

    # График
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(frequencies, results["analytic"], "o-", color="navy",
              label="Analytic delta")
    ax.loglog(frequencies, results["fdm"], "s--", color="crimson",
              label="FDM delta (M=200)")
    ax.set_xlabel(r"Number of Rehedging Steps $N_{steps}$")
    ax.set_ylabel(r"Std Dev of P&L $\sigma$(P&L)")
    ax.set_title("Hedging Risk: Analytic vs FDM Delta",
                 fontweight="bold")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    _save(fig, outdir, "figure4b_fdm_vs_analytic.png")

    print(f"  analytic: {results['analytic']}")
    print(f"  fdm:      {results['fdm']}")

def figure4c_mc_vs_analytic(outdir, N_paths=5000):
    """
    Сравнение P&L с аналитической, FDM и MC дельтой.
    """
    print("Generating Figure 4c: MC vs Analytic vs FDM Hedging...")
    from fdm_delta import make_fdm_delta_func
    from mc_delta import make_mc_delta_func

    sim = DeltaHedgingSimulator(S0, K, T, r, sigma)
    frequencies = [12, 52, 252]
    
    results = {"analytic": [], "fdm": [], "mc": []}

    delta_fdm = make_fdm_delta_func(K, T, r, sigma, M=200, N_t=200)
    # N_inner=5000 — компромисс между шумом и скоростью
    delta_mc = make_mc_delta_func(N_inner=5000, seed=42)

    for N_steps in frequencies:
        pnl_a, _ = sim.run_simulation(N_steps=N_steps, N_paths=N_paths, seed=42)
        pnl_f, _ = sim.run_simulation(N_steps=N_steps, N_paths=N_paths,
                                       seed=42, delta_func=delta_fdm)
        pnl_m, _ = sim.run_simulation(N_steps=N_steps, N_paths=N_paths,
                                       seed=42, delta_func=delta_mc)
        results["analytic"].append(np.std(pnl_a))
        results["fdm"].append(np.std(pnl_f))
        results["mc"].append(np.std(pnl_m))

    # График
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(frequencies, results["analytic"], "o-", color="navy",
              linewidth=2, label="Analytic delta")
    ax.loglog(frequencies, results["fdm"], "s--", color="crimson",
              linewidth=2, label="FDM delta (M=200)")
    ax.loglog(frequencies, results["mc"], "d-.", color="darkgreen",
              linewidth=2, label="MC delta (N=5000)")
    
    ax.set_xlabel(r"Number of Rehedging Steps $N_{steps}$")
    ax.set_ylabel(r"Std Dev of P&L $\sigma$(P&L)")
    ax.set_title("Hedging Risk: Analytic vs FDM vs MC Delta",
                 fontweight="bold")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    _save(fig, outdir, "figure4c_mc_vs_analytic.png")

    print(f"  analytic: {results['analytic']}")
    print(f"  fdm:      {results['fdm']}")
    print(f"  mc:       {results['mc']}")
# =====================================================================
# Публичный API
# =====================================================================

def generate_all_figures(outdir="figures"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    figure1(outdir)
    figure2(outdir)
    figure3(outdir)
    # figure4(outdir)
    figure4b_fdm_vs_analytic(outdir) 
    figure4c_mc_vs_analytic(outdir)
    print("\nALL FIGURES GENERATED SUCCESSFULLY.")

if __name__ == "__main__":
    generate_all_figures(outdir="figures")
