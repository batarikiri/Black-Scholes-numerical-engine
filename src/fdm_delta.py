"""
fdm_delta.py
============
Оборачивает FDM-решётку в функцию delta(S, K, tau, r, sigma),
совместимую с DeltaHedgingSimulator.
"""

import numpy as np
from bs_engine import CrankNicolsonBS


def make_fdm_delta_func(K, T, r, sigma, M=200, N_t=200, S_max_mult=3.0):
    """
    Возвращает функцию delta_fdm(S, K, tau, r, sigma) → np.ndarray.
    
    Внутри создаёт FDM-решётку и интерполирует дельту в точке (S, t = T - tau).
    """
    fdm = CrankNicolsonBS(K=K, T=T, r=r, sigma=sigma,
                          S_max_mult=S_max_mult, M=M, N_t=N_t)
    fdm.solve()  # сохраняет self.Delta_full внутри

    def delta_fdm(S, K_ignored, tau, r_ignored, sigma_ignored):
        S_arr = np.asarray(S, dtype=float)
        tau_arr = np.asarray(tau, dtype=float)
        t_arr = T - tau_arr
        res = fdm.get_delta(S_arr, t_arr)
        # Если на входе был скаляр, вернуть скаляр
        if np.ndim(S) == 0 and np.ndim(tau) == 0:
            return float(res[0]) if isinstance(res, np.ndarray) else float(res)
        return res

    return delta_fdm