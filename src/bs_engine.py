"""
bs_engine.py
============
Модуль 1: Аналитический движок Блэка–Шоулза (closed-form).
Модуль 2: FDM Crank–Nicolson solver.

Содержит:
  - bs_call_price, bs_call_delta, bs_call_gamma  (аналитика)
  - CrankNicolsonBS                              (PDE решатель)
"""

import numpy as np
from scipy.stats import norm
from scipy.linalg import solve_banded

# =====================================================================
# МОДУЛЬ 1: Аналитический движок (Black-Scholes Closed-Form Benchmark)
# =====================================================================

def bs_call_price(S, K, T, r, sigma):
    """
    Точная цена европейского Call-опциона Блэка-Шоулза.
    Поддерживает как скаляры, так и NumPy массивы для S.
    """
    S = np.asarray(S, dtype=float)
    # Защита от деления на 0 при T -> 0 или S -> 0
    if T <= 0:
        return np.maximum(S - K, 0.0)

    # Для S = 0 цена опциона Call всегда 0
    S_safe = np.maximum(S, 1e-12)

    d1 = (np.log(S_safe / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = S_safe * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    # Если на вход пришел 0, возвращаем 0
    if np.ndim(S) == 0:
        return 0.0 if S == 0 else float(price)
    price[S == 0] = 0.0
    return price


def bs_call_delta(S, K, T, r, sigma):
    """Точная Дельта (Delta = dV/dS = Phi(d1)) для Call-опциона."""
    S = np.asarray(S, dtype=float)
    if T <= 0:
        return np.where(S > K, 1.0, 0.0)

    S_safe = np.maximum(S, 1e-12)
    d1 = (np.log(S_safe / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1)

    if np.ndim(S) == 0:
        return 0.0 if S == 0 else float(delta)
    delta[S == 0] = 0.0
    return delta


def bs_call_gamma(S, K, T, r, sigma):
    """Точная Гамма (Gamma = d^2V/dS^2) для Call-опциона."""
    S = np.asarray(S, dtype=float)
    if T <= 0:
        return np.zeros_like(S)

    S_safe = np.maximum(S, 1e-12)
    d1 = (np.log(S_safe / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S_safe * sigma * np.sqrt(T))

    if np.ndim(S) == 0:
        return 0.0 if S == 0 else float(gamma)
    gamma[S == 0] = 0.0
    return gamma


# =====================================================================
# МОДУЛЬ 2: Сеточный метод конечных разностей (Crank-Nicolson FDM)
# =====================================================================

class CrankNicolsonBS:
    """
    Решатель PDE Блэка-Шоулза полунеявной схемой Кранка - Николсон.
    Использует прогонку O(M) через scipy.linalg.solve_banded.
    """
    def __init__(self, K, T, r, sigma, S_max_mult=3.0, M=200, N_t=200):
        """
        K, T, r, sigma: Финансовые параметры
        S_max_mult: Верхняя граница сетки S_max = S_max_mult * K
        M: Число шагов по пространству S (узлов M+1)
        N_t: Число шагов по времени t
        """
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.M = M
        self.N_t = N_t

        self.S_max = S_max_mult * K
        self.dS = self.S_max / M
        self.dt = T / N_t

        # Сетка по пространству S_i = i * dS, i = 0...M
        self.S_grid = np.linspace(0, self.S_max, M + 1)

    def solve(self):
        """
        Основной цикл решения PDE назад по времени (от t=T до t=0).
        Возвращает:
          - V: вектор цен на всей сетке S в момент t=0
          - Delta: вектор численной Дельты в момент t=0
        """
        M = self.M
        dt = self.dt
        dS = self.dS
        r = self.r
        sigma = self.sigma

        # Индексы внутренних узлов i = 1 ... M-1
        i = np.arange(1, M)

        # Безразмерные коэффициенты дискретизации
        a_i = 0.25 * dt * (sigma**2 * i**2 - r * i)
        b_i = -0.5 * dt * (sigma**2 * i**2 + r)
        c_i = 0.25 * dt * (sigma**2 * i**2 + r * i)

        # Диагонали матрицы системы
        ab = np.zeros((3, M - 1))
        ab[0, 1:] = -c_i[:-1]   # Upper diagonal
        ab[1, :]  = 1.0 - b_i   # Main diagonal
        ab[2, :-1] = -a_i[1:]   # Lower diagonal

        # Начальное краевое условие при tau = 0 (t = T): Payoff = max(S - K, 0)
        V = np.maximum(self.S_grid - self.K, 0.0)
        
        # --- СОЗДАЕМ МАТРИЦУ ДЛЯ ИСТОРИИ ---
        V_history = np.zeros((M + 1, self.N_t + 1))
        V_history[:, -1] = V  # t = T (последний столбец)

        # Цикл назад по времени tau от 1 до N_t (соответствует от t = T-dt до t = 0)
        for j in range(1, self.N_t + 1):
            tau_next = j * dt  # Время до экспирации на следующем шаге

            # Левая и правая граничные условия на следующем шаге
            V_0_next = 0.0  # Call-specific, для Put надо -1
            V_M_next = self.S_max - self.K * np.exp(-r * tau_next)

            # Формирование вектора правой части (RHS) для внутренних узлов i = 1...M-1
            RHS = a_i * V[:-2] + (1.0 + b_i) * V[1:-1] + c_i * V[2:]

            # Учет граничных условий в RHS
            RHS[0] += a_i[0] * V_0_next
            RHS[-1] += c_i[-1] * V_M_next

            # Решение СЛАУ для внутренних точек
            V_inner = solve_banded((1, 1), ab, RHS)

            # Обновление вектора V
            V[0] = V_0_next
            V[1:-1] = V_inner
            V[-1] = V_M_next
            
            # --- СОХРАНЯЕМ ШАГ В ИСТОРИЮ ---
            # j=1 (tau=dt) -> t = T-dt -> индекс N_t - 1
            # j=N_t (tau=T) -> t = 0 -> индекс 0
            V_history[:, self.N_t - j] = V

        # --- РАСЧЕТ ДЕЛЬТЫ ДЛЯ ВСЕЙ ИСТОРИИ ---
        self.V_full = V_history
        self.Delta_full = np.zeros_like(V_history)
        
        for j in range(self.N_t + 1):
            v_j = V_history[:, j]
            d_j = np.zeros_like(v_j)
            d_j[1:-1] = (v_j[2:] - v_j[:-2]) / (2 * dS)
            d_j[0] = 0.0
            d_j[-1] = 1.0
            self.Delta_full[:, j] = d_j

        # Возвращаем значения в момент t=0 (первый столбец истории)
        V_0 = V_history[:, 0]
        Delta_0 = self.Delta_full[:, 0]

        return V_0, Delta_0
    # --------------------------------------------------------------
    # интерполяция в произвольной точке (S, t)
    # --------------------------------------------------------------
    def _interp_matrix(self, M_full, S, t):
        """
        Билинейная интерполяция матрицы M_full по сетке (S, t).
        M_full имеет форму (M+1, N_t+1): M_full[i, j] = value в (S_i, t_j).
        """
        S_arr = np.atleast_1d(np.asarray(S, dtype=float))
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))

        # Клиппинг к границам сетки
        S_clip = np.clip(S_arr, 0.0, self.S_max)
        t_clip = np.clip(t_arr, 0.0, self.T)

        # Индексы
        i = S_clip / self.dS
        j = t_clip / self.dt

        i0 = np.clip(np.floor(i).astype(int), 0, self.M)
        j0 = np.clip(np.floor(j).astype(int), 0, self.N_t)
        i1 = np.clip(i0 + 1, 0, self.M)
        j1 = np.clip(j0 + 1, 0, self.N_t)

        wi = i - i0
        wj = j - j0

        # Билинейная формула
        val = (
            (1 - wi) * (1 - wj) * M_full[i0, j0]
            + wi * (1 - wj) * M_full[i1, j0]
            + (1 - wi) * wj * M_full[i0, j1]
            + wi * wj * M_full[i1, j1]
        )

        #return val if np.ndim(S) == 0 and np.ndim(t) == 0 else val
                # Если на входе были скаляры — вернуть скаляр
        if np.ndim(S) == 0 and np.ndim(t) == 0:
            return float(val[0]) if isinstance(val, np.ndarray) else float(val)
        return val

    def get_price(self, S, t):
        """Цена опциона в произвольной точке (S, t) через интерполяцию."""
        return self._interp_matrix(self.V_full, S, t)

    def get_delta(self, S, t):
        """Дельта опциона в произвольной точке (S, t) через интерполяцию."""
        return self._interp_matrix(self.Delta_full, S, t)
     


if __name__ == "__main__":
    # Финансовые параметры
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    # 1. Аналитика
    price_exact = bs_call_price(S0, K, T, r, sigma)
    delta_exact = bs_call_delta(S0, K, T, r, sigma)

    # 2. FDM Кранка-Николсон
    fdm = CrankNicolsonBS(K=K, T=T, r=r, sigma=sigma, M=200, N_t=200)
    V_fdm, Delta_fdm = fdm.solve()

    price_fdm = fdm.get_price(S0, 0.0)
    delta_fdm = fdm.get_delta(S0, 0.0)

    print(f"=== RECONCILIATION TEST (S0={S0}, K={K}) ===")
    print(f"Exact Price : {price_exact:.6f} | FDM Price : {price_fdm:.6f} | Abs Error: {abs(price_exact - price_fdm):.2e}")
    print(f"Exact Delta : {delta_exact:.6f} | FDM Delta : {delta_fdm:.6f} | Abs Error: {abs(delta_exact - delta_fdm):.2e}")