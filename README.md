# Black-Scholes Numerical Engine

Numerical study of the Black-Scholes PDE via FDM (Crank-Nicolson), Monte Carlo, and Delta Hedging.

## Presentation
- [Compiled PDF](presentation/black_scholes_pres.pdf)

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
