import numpy as np
import pandas as pd

# ===========================================
# Performance Metrics Module
# ===========================================

def calculate_cagr(returns: pd.Series, periods_per_year=252):
    cumulative_return = (1 + returns).prod()
    n_years = len(returns) / periods_per_year
    return cumulative_return ** (1 / n_years) - 1

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate=0.01, periods_per_year=252):
    excess_returns = returns - (risk_free_rate / periods_per_year)
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate=0.01, periods_per_year=252):
    downside_returns = returns[returns < 0]
    excess_returns = returns - (risk_free_rate / periods_per_year)
    return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(periods_per_year)

def calculate_max_drawdown(cumulative_returns: pd.Series):
    rolling_max = cumulative_returns.cummax()
    drawdowns = cumulative_returns / rolling_max - 1
    return drawdowns.min()

def calculate_xirr(cash_flows: list, dates: list):
    from scipy.optimize import newton

    def xnpv(rate):
        return sum(cf / (1 + rate) ** ((d - dates[0]).days / 365) for cf, d in zip(cash_flows, dates))

    try:
        return newton(xnpv, 0.1)
    except Exception:
        return np.nan

def calculate_alpha_beta(returns: pd.Series, benchmark_returns: pd.Series):
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if aligned.shape[0] < 2:
        return np.nan, np.nan
    cov_matrix = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    alpha = aligned.iloc[:, 0].mean() - beta * aligned.iloc[:, 1].mean()
    return alpha, beta

def calculate_var(returns: pd.Series, confidence_level=0.95):
    return np.percentile(returns, (1 - confidence_level) * 100)

def evaluate_strategy(returns: pd.Series, benchmark_returns: pd.Series = None, cash_flows: list = None, dates: list = None):
    cumulative_returns = (1 + returns).cumprod()

    metrics = {
        "CAGR": round(calculate_cagr(returns) * 100, 2),
        "Sharpe Ratio": round(calculate_sharpe_ratio(returns), 2),
        "Sortino Ratio": round(calculate_sortino_ratio(returns), 2),
        "Max Drawdown": round(calculate_max_drawdown(cumulative_returns) * 100, 2),
        "VaR (95%)": round(calculate_var(returns) * 100, 2),
    }

    if benchmark_returns is not None:
        alpha, beta = calculate_alpha_beta(returns, benchmark_returns)
        metrics["Alpha"] = round(alpha, 4)
        metrics["Beta"] = round(beta, 4)

    if cash_flows and dates:
        xirr_value = calculate_xirr(cash_flows, dates)
        metrics["XIRR"] = round(xirr_value * 100, 2) if not np.isnan(xirr_value) else "N/A"

    return metrics