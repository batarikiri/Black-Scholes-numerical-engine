import time
import numpy as np

# Импортируем аналитические функции из Модуля 1
# (если всё в одном файле/ноутбуке, функции уже доступны)
from bs_engine import bs_call_price, bs_call_delta

# =====================================================================
# МОДУЛЬ 3: Monte Carlo Engine (Standard vs Antithetic Variates)
# =====================================================================

def monte_carlo_call_price(S0, K, T, r, sigma, N_sim=100000, use_antithetic=True, seed=42):
    if seed is not None:
        np.random.seed(seed)

    start_time = time.perf_counter()

    if use_antithetic:
        half_sim = N_sim // 2
        Z = np.random.standard_normal(half_sim)

        # Считаем траектории для пар Z и -Z параллельно
        ST1 = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        ST2 = S0 * np.exp((r - 0.5 * sigma**2) * T - sigma * np.sqrt(T) * Z)

        payoffs1 = np.maximum(ST1 - K, 0.0)
        payoffs2 = np.maximum(ST2 - K, 0.0)

        # Парная оценка Y_i = (Payoff(Z) + Payoff(-Z)) / 2
        paired_discounted_payoffs = 0.5 * np.exp(-r * T) * (payoffs1 + payoffs2)

        price_est = np.mean(paired_discounted_payoffs)
        std_err = np.std(paired_discounted_payoffs, ddof=1) / np.sqrt(half_sim)
    else:
        Z = np.random.standard_normal(N_sim)
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        discounted_payoffs = np.exp(-r * T) * np.maximum(ST - K, 0.0)

        price_est = np.mean(discounted_payoffs)
        std_err = np.std(discounted_payoffs, ddof=1) / np.sqrt(N_sim)

    elapsed_time = time.perf_counter() - start_time
    return price_est, std_err, elapsed_time

def monte_carlo_call_delta(S0, K, T, r, sigma, N_sim=100000, seed=42):
    """
    Оценка дельты Call-опциона методом Монте-Карло (Pathwise Derivative).
    Возвращает: (delta_est, std_err, elapsed_time)
    """
    if seed is not None:
        np.random.seed(seed)
        
    start_time = time.perf_counter()
    
    Z = np.random.standard_normal(N_sim)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Pathwise derivative: dPayoff/dS0 = (ST/S0) * 1_{ST > K}
    # Умножаем на дисконт exp(-rT)
    pathwise_deltas = np.exp(-r * T) * (ST / S0) * (ST > K)
    
    delta_est = np.mean(pathwise_deltas)
    std_err = np.std(pathwise_deltas, ddof=1) / np.sqrt(N_sim)
    
    elapsed_time = time.perf_counter() - start_time
    return delta_est, std_err, elapsed_time
# =====================================================================
# МОДУЛЬ 4: Векторизованный симулятор Delta Hedging (Core Engine)
# =====================================================================

class DeltaHedgingSimulator:
    """
    Симулятор портфеля динамического рехеджирования Call-опциона.
    Полностью векторизован по траекториям в NumPy.
    """
    def __init__(self, S0, K, T, r, sigma):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def generate_paths(self, N_steps, N_paths, seed=42):
        """
        Генерация траекторий GBM: матрица размера (N_paths, N_steps + 1).
        """
        if seed is not None:
            np.random.seed(seed)

        dt = self.T / N_steps
        dt_sqrt = np.sqrt(dt)

        Z = np.random.standard_normal((N_paths, N_steps))

        drift = (self.r - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * dt_sqrt * Z
        log_returns = drift + diffusion

        # Накопленная сумма лог-доходностей
        log_S = np.zeros((N_paths, N_steps + 1))
        log_S[:, 0] = np.log(self.S0)
        log_S[:, 1:] = np.log(self.S0) + np.cumsum(log_returns, axis=1)

        return np.exp(log_S)

    def run_simulation(self, N_steps, N_paths=1000, seed=42, delta_func=None):
        """
        Запуск рехеджирования портфеля.

        Параметры:
          - N_steps: число шагов рехеджа
          - N_paths: число траекторий
          - seed: seed для генератора
          - delta_func: функция delta_func(S, K, tau, r, sigma) → delta.
            Если None — используется аналитика (bs_call_delta).

        Возвращает:
          - pnl: Массив P&L в момент T для каждой траектории (N_paths,)
          - S_paths: Массив траекторий цен акций (N_paths, N_steps + 1)
        """
        if delta_func is None:
            delta_func = bs_call_delta

        dt = self.T / N_steps
        S_paths = self.generate_paths(N_steps, N_paths, seed=seed)

        # 1. Начальное состояние в t = 0
        C0 = bs_call_price(self.S0, self.K, self.T, self.r, self.sigma)
        delta_0 = delta_func(self.S0, self.K, self.T, self.r, self.sigma)

        # Кэш на банковском счете: получили C0 за продажу опциона, купили delta_0 акций
        cash = C0 - delta_0 * self.S0
        delta_prev = delta_0

        # 2. Динамическое рехеджирование на каждом шаге от t_1 до t_{N_steps}
        for step in range(1, N_steps + 1):
            t_current = step * dt
            time_to_maturity = self.T - t_current
            S_t = S_paths[:, step]

            # Начисление процентов на банковский счет за интервал dt
            cash *= np.exp(self.r * dt)

            # Расчет новой аналитической Дельты Блэка-Шоулза
            #delta_current = bs_call_delta(S_t, self.K, time_to_maturity, self.r, self.sigma)
            # Расчет новой Дельты (аналитика или пользовательская функция)
            delta_current = delta_func(S_t, self.K, time_to_maturity, self.r, self.sigma)

            
            # Ребалансировка: докупаем/продаем (delta_current - delta_prev) акций
            hedge_trade = delta_current - delta_prev
            cash -= hedge_trade * S_t

            delta_prev = delta_current

        # 3. Исполнение контрактов в момент экспирации T
        ST = S_paths[:, -1]
        payoff = np.maximum(ST - self.K, 0.0)

        # Ликвидация: закрываем позицию по акциям по текущей цене ST и выплачиваем payoff по опциону
        terminal_pnl = cash + delta_prev * ST - payoff

        return terminal_pnl, S_paths


if __name__ == "__main__":
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    exact_price = bs_call_price(S0, K, T, r, sigma)

    print("=== TEST 1: MONTE CARLO PRICE & VARIANCE REDUCTION ===")
    p_std, err_std, t_std = monte_carlo_call_price(S0, K, T, r, sigma, N_sim=100000, use_antithetic=False)
    p_anti, err_anti, t_anti = monte_carlo_call_price(S0, K, T, r, sigma, N_sim=100000, use_antithetic=True)

    print(f"Exact Price       : {exact_price:.6f}")
    print(f"Standard MC       : {p_std:.6f} | StdErr: {err_std:.6f} | Time: {t_std*1000:.1f} ms")
    print(f"Antithetic MC     : {p_anti:.6f} | StdErr: {err_anti:.6f} | Time: {t_anti*1000:.1f} ms")
    print(f"Variance Reduction: {(err_std**2) / (err_anti**2):.2f}x lower variance!\n")

    print("=== TEST 2: DELTA HEDGING P&L DEPENDENCE ON REHEDGING FREQUENCY ===")
    simulator = DeltaHedgingSimulator(S0, K, T, r, sigma)
    N_paths = 5000

    # Ежемесячно (12 шагов), Еженедельно (52 шага), Ежедневно (252 шага)
    frequencies = {"Monthly (N=12)": 12, "Weekly (N=52)": 52, "Daily (N=252)": 252}

    print(f"Simulating {N_paths} paths across hedging frequencies...")
    for label, N_steps in frequencies.items():
        pnl, _ = simulator.run_simulation(N_steps=N_steps, N_paths=N_paths, seed=42)
        mean_pnl = np.mean(pnl)
        std_pnl = np.std(pnl)
        print(f"[{label:15s}] Mean P&L: {mean_pnl:8.4f} | Std P&L: {std_pnl:6.4f} | Max Loss: {np.min(pnl):7.4f}")