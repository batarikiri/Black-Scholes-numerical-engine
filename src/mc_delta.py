"""
mc_delta.py
===========
Оборачивает Monte Carlo оценку дельты в функцию delta(S, K, tau, r, sigma),
совместимую с DeltaHedgingSimulator.
"""

import numpy as np


def make_mc_delta_func(N_inner=10000, seed=None):
    """
    Возвращает функцию delta_mc(S, K, tau, r, sigma) → np.ndarray.
    
    Для каждой точки (S, tau) запускает внутренний Monte Carlo
    и оценивает дельту через pathwise derivative.
    
    N_inner: число внутренних путей для оценки дельты в каждой точке.
    """
    rng = np.random.default_rng(seed)

    def delta_mc(S, K, tau, r, sigma):
        S_arr = np.atleast_1d(np.asarray(S, dtype=float))
        tau_arr = np.atleast_1d(np.asarray(tau, dtype=float))
        
        # Broadcast to common shape
        S_arr, tau_arr = np.broadcast_arrays(S_arr, tau_arr)
        
        deltas = np.zeros_like(S_arr)
        
        for i in np.ndindex(S_arr.shape):
            s = S_arr[i]
            t = tau_arr[i]
            
            if t <= 1e-10 or s <= 1e-10:
                # При экспирации дельта — ступенька
                deltas[i] = 1.0 if s > K else 0.0
                continue
            
            # Внутренний MC
            Z = rng.standard_normal(N_inner)
            ST = s * np.exp((r - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * Z)
            
            # Pathwise derivative
            pathwise = np.exp(-r * t) * (ST / s) * (ST > K)
            deltas[i] = np.mean(pathwise)
        
        # Если на входе был скаляр — вернуть скаляр
        if np.ndim(S) == 0 and np.ndim(tau) == 0:
            return float(deltas.flat[0])
        return deltas

    return delta_mc