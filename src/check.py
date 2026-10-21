from bs_engine import CrankNicolsonBS, bs_call_price, bs_call_delta
import numpy as np

fdm = CrankNicolsonBS(K=100, T=1, r=0.05, sigma=0.2, M=200, N_t=200)
V, _ = fdm.solve()

# В точке (S=100, t=0):
S_test, t_test = 100.0, 0.0
print(f"FDM price: {fdm.get_price(S_test, t_test):.4f}")     # ≈ 10.45
print(f"FDM delta: {fdm.get_delta(S_test, t_test):.4f}")     # ≈ 0.637

# В точке (S=110, t=0.5):
print(f"FDM price (110, 0.5): {fdm.get_price(110, 0.5):.4f}")
print(f"FDM delta (110, 0.5): {fdm.get_delta(110, 0.5):.4f}")
print(f"Analytic price: {bs_call_price(110, 100, 0.5, 0.05, 0.2):.4f}")
print(f"Analytic delta: {bs_call_delta(110, 100, 0.5, 0.05, 0.2):.4f}")