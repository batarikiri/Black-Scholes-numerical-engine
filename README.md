# Black-Scholes Numerical Engine

Numerical study of the Black-Scholes PDE via FDM (Crank-Nicolson), Monte Carlo, and Delta Hedging.

## Project Structure
- `src/bs_engine.py` — Analytical solution + FDM solver
- `src/bs_monte_carlo.py` — Monte Carlo pricing + Delta Hedging simulator
- `src/fdm_delta.py` / `src/mc_delta.py` — Adapters to feed numerical deltas into the hedging simulator
- `src/benchmarks.py` — Generates Figures 1–4
- `scripts/run_all.py` — Main entry point

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python scripts/run_all.py
```
::: frame
:::

::: frame
Problem Statement: The Black-Scholes PDE We solve the parabolic PDE for
the option price $V(S,t)$: $$\begin{equation*}
        \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0
\end{equation*}$$ **Boundary conditions** (European Call):

- Terminal payoff: $V(S,T) = \max(S - K, 0)$

- Left boundary: $V(0,t) = 0$

- Right boundary: $V(S_{\max},t) \approx S_{\max} - K e^{-r(T-t)}$

**Notation:**

  ----- ----------------------- ---------- -------------------------------------------------
  $S$   Asset price (spot)      $r$        Risk-free interest rate
  $t$   Current time            $\sigma$   Volatility (diffusion coefficient)
  $T$   Maturity (expiration)   $V(S,t)$   Option price (unknown function)
  $K$   Strike price            $\Delta$   Delta (hedge ratio) = $\partial V / \partial S$
  ----- ----------------------- ---------- -------------------------------------------------
:::

::: frame
Analytical Benchmark (Ground Truth) The closed-form solution for a
European Call: $$\begin{equation*}
        C(S,t) = S\Phi(d_1) - K e^{-r(T-t)}\Phi(d_2)
\end{equation*}$$ where $\Phi(\cdot)$ is the standard normal CDF, and
$$\begin{equation*}
        d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)(T-t)}{\sigma\sqrt{T-t}}, \quad d_2 = d_1 - \sigma\sqrt{T-t}
\end{equation*}$$

![](./1.png){width="70%" height="35%"}
:::

::: frame
Method 1: Crank-Nicolson FDM **Discretization:** Grid $S_i = i\Delta S$,
$t_j = j\Delta t$. Let $V_i^j = V(S_i, t_j)$.

**Crank-Nicolson Scheme** (average of implicit and explicit):
$$\begin{equation*}
        \frac{V_i^{j+1} - V_i^j}{\Delta t} + \frac{1}{2}\left[ \mathcal{L}V_i^{j+1} + \mathcal{L}V_i^j \right] = 0
\end{equation*}$$ Matrix form:
$(I + \frac{\Delta t}{2}L)V^{j+1} = (I - \frac{\Delta t}{2}L)V^j$.

**Properties:**

- Order: $O(\Delta t^2 + \Delta S^2)$

- Unconditionally stable (A-stable)

- Tridiagonal system $\rightarrow$ solved in $O(M)$ via Thomas algorithm
:::

::: frame
Crank-Nicolson: Matrix coefficients **Coefficients** (from central
differences, $S_i = i\Delta S$): $$\begin{equation*}
        a_i = \frac{\Delta t}{4}(\sigma^2 i^2 - ri), \quad
        b_i = -\frac{\Delta t}{2}(\sigma^2 i^2 + r), \quad
        c_i = \frac{\Delta t}{4}(\sigma^2 i^2 + ri)
\end{equation*}$$ **System:** $$\begin{equation*}
        -a_i V_{i-1}^{j+1} + (1 - b_i) V_i^{j+1} - c_i V_{i+1}^{j+1} = a_i V_{i-1}^j + (1 + b_i) V_i^j + c_i V_{i+1}^j
\end{equation*}$$
:::

::: frame
FDM Results: Convergence

![FDM error decay: empirical slope $\approx -2.02$ (theoretical
$O(M^{-2})$).](./Black-Scholes-numerical-engine-main/fdm_convergence.png){width="80%"
height="70%"}

The empirical slope perfectly matches the theoretical second-order
convergence.
:::

::: frame
Method 2: Monte Carlo **GBM Simulation** (risk-neutral measure):
$$\begin{equation*}
        S_T^{(k)} = S_0 \exp\left[\left(r - \frac{\sigma^2}{2}\right)T + \sigma\sqrt{T}Z^{(k)}\right], \quad Z^{(k)} \sim \mathcal{N}(0,1)
\end{equation*}$$

**Price Estimator**: $$\begin{equation*}
        \widehat{C} = e^{-rT} \frac{1}{N} \sum_{k=1}^N \max(S_T^{(k)} - K, 0)
\end{equation*}$$

**Properties:**

- Convergence: $O(N^{-1/2})$ from the Central Limit Theorem.

- Antithetic Variates: Pair $(Z, -Z)$ to reduce variance.

- Empirical variance reduction: $\approx 2.03\times$.
:::

::: frame
Monte Carlo Results: Convergence

![MC error decay: empirical slope $\approx -0.50$ (theoretical
$O(N^{-1/2})$).](./Black-Scholes-numerical-engine-main/mc_convergence.png){width="80%"
height="70%"}

Both standard and antithetic MC follow the $1/\sqrt{N}$ law. Antithetic
reduces variance by a factor of 2.
:::

::: frame
Delta Hedging Mechanics **Goal:** Eliminate the directional risk of a
sold option.\
**Idea:** Hold $\Delta$ shares of the underlying asset. If $S$ changes
by $dS$, the option changes by $\Delta dS$, and the shares change by
$\Delta dS$ (net change $\approx 0$).

**Algorithm (Discrete Rebalancing):**

1.  **At $t=0$:** Sell 1 Call for $C_0$. Compute
    $\Delta_0 = \partial V/\partial S$. Buy $\Delta_0$ shares. Bank
    cash: $B_0 = C_0 - \Delta_0 S_0$.

2.  **At each step $t_j$:** Recompute $\Delta_j$. Buy/sell
    $(\Delta_j - \Delta_{j-1})$ shares. Pay interest on the bank account
    $B_j = B_{j-1} e^{r\Delta t}$.

3.  **At maturity $T$:** Liquidate shares. Pay the option payoff.
    Compute the terminal P&L.

**Terminal P&L:** $$\begin{equation*}
        P\&L = B_T + \Delta_T S_T - \max(S_T - K, 0)
\end{equation*}$$ In continuous time, $P\&L \equiv 0$. In discrete time,
residual error appears.
:::

::: frame
Delta Hedging: Error Decay

![](./5.png){width="95%" height="65%"}

**Error decay:** $\sigma(P\&L) \propto 1/\sqrt{N_{\text{steps}}}$
(empirical slope $\approx -0.48$).
:::

::: frame
Delta Hedging: Error Decay **Theoretical error bound:**
$$\begin{equation*}
        P\&L \approx -\frac{1}{2} \sum_j \Gamma_j S_j^2 \sigma^2 \left[ (\Delta W_j)^2 - \Delta t \right]
\end{equation*}$$ where $\Gamma = \partial^2 V / \partial S^2$.\
This gives $\text{Var}(P\&L) \propto \frac{1}{N_{\text{steps}}}$, hence
$\sigma(P\&L) \propto \frac{1}{\sqrt{N_{\text{steps}}}}$.
:::

::: frame
Hedging Trade-off: FDM vs MC Delta

![FDM delta matches analytics perfectly.](./4.png){width="\\linewidth"
height="60%"}

FDM is deterministic and smooth; MC is universal but introduces
stochastic noise.
:::

:::: frame
Validation Summary (Base Case) $S_0 = 100$, $K = 100$, $T = 1$,
$r = 0.05$, $\sigma = 0.2$. Exact price: $10.4506$.

  Method                         Price       Absolute Error           Order
  ---------------------------- --------- ----------------------- ---------------
  Analytical                    10.4506            ---                 ---
  FDM ($M=200$)                 10.4544   $3.85 \times 10^{-3}$    $O(M^{-2})$
  MC ($N=10^5$)                 10.4739   $2.3 \times 10^{-2}$    $O(N^{-1/2})$
  MC + Antithetic ($N=10^5$)    10.4529   $2.3 \times 10^{-3}$    $O(N^{-1/2})$

**Hedging error decay** (empirical slope $\approx -0.48$):

::: center
     Monthly ($N=12$)         Weekly ($N=52$)         Daily ($N=252$)
  ----------------------- ----------------------- -----------------------
   $\sigma(P\&L) = 1.91$   $\sigma(P\&L) = 0.95$   $\sigma(P\&L) = 0.44$
:::
::::

:::::: frame
Code Architecture

::::: columns
::: column
0.5 **Structure:**

- `bs_engine.py` --- Analytics + FDM

- `bs_monte_carlo.py` --- MC + Hedging

- `fdm_delta.py` / `mc_delta.py` --- Adapters

- `benchmarks.py` --- Figures generation

**Code**\
<https://github.com/batarikiri/black-scholes-numerical-engine>
:::

::: column
0.5 **Done:**

- FDM and MC validated against analytics.

- FDM delta replaces analytical delta without loss.

- MC delta introduces noise (trade-off).

**Other tasks:**

- American options (LCP)

- Stochastic Heston Volatility (2D PDE)
:::
:::::
::::::

