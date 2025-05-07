import numpy as np
import pandas as pd
from datetime import datetime

# ---------------------- CAGR ----------------------
def compute_cagr(portfolio_values: pd.Series) -> float:
    n_years = (portfolio_values.index[-1] - portfolio_values.index[0]).days / 365.25
    return (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (1 / n_years) - 1

# ---------------------- XIRR ----------------------
def compute_xirr(cash_flows: list[tuple[datetime, float]]) -> float:
    def xnpv(rate):
        return sum(cf / (1 + rate) ** ((date - cash_flows[0][0]).days / 365.0) for date, cf in cash_flows)

    def xirr():
        rate = 0.1
        for _ in range(100):
            f = xnpv(rate)
            df = sum(-((date - cash_flows[0][0]).days / 365.0) * cf / (1 + rate) ** (((date - cash_flows[0][0]).days / 365.0) + 1)
                     for date, cf in cash_flows)
            if df == 0:
                break
            rate -= f / df
            if abs(f) < 1e-6:
                return rate
        return rate

    return xirr()

# ---------------------- Sharpe Ratio ----------------------
def compute_sharpe_ratio(returns: pd.Series, risk_free_rate=0.01) -> float:
    excess_returns = returns - risk_free_rate / 252
    return np.mean(excess_returns) / np.std(excess_returns)

# ---------------------- Sortino Ratio ----------------------
def compute_sortino_ratio(returns: pd.Series, risk_free_rate=0.01) -> float:
    downside_returns = returns[returns < 0]
    excess_returns = returns - risk_free_rate / 252
    return np.mean(excess_returns) / np.std(downside_returns)

# ---------------------- Max Drawdown ----------------------
def compute_max_drawdown(portfolio_values: pd.Series) -> float:
    cummax = portfolio_values.cummax()
    drawdown = (portfolio_values - cummax) / cummax
    return drawdown.min()

# ---------------------- Alpha & Beta ----------------------
def compute_alpha_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> tuple[float, float]:
    cov_matrix = np.cov(portfolio_returns, benchmark_returns)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    alpha = np.mean(portfolio_returns) - beta * np.mean(benchmark_returns)
    return alpha, beta

# ---------------------- Value at Risk ----------------------
def compute_var(returns: pd.Series, confidence_level=0.95) -> float:
    return np.percentile(returns, (1 - confidence_level) * 100)
