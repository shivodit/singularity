import numpy as np
from math import sqrt

# Generate artificial price series using geometric Brownian motion
def generate_price_series(T=1000, S0=100, mu=0.0001, sigma=0.01):
    dt = 1
    prices = [S0]
    for _ in range(T-1):
        Z = np.random.normal(0, 1)
        S_t = prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        prices.append(S_t)
    returns = np.diff(np.log(prices))  # Log returns
    return np.array(prices), returns

# Sharpe ratio calculation
def sharpe_ratio(Ret):
    T = len(Ret)
    mean_ret = float(sum(Ret)) / T
    mean_sq_ret = float(sum(Ret**2)) / T
    if mean_ret == 0.0 and mean_sq_ret == 0.0:
        return 0
    sharpe = mean_ret / sqrt(mean_sq_ret - mean_ret * mean_ret)
    return sharpe

# Compute trading positions F_t
def update_Ft(X, theta, T, M):
    Ft = np.zeros(T+1)
    for t in range(1, T+1):
        xt = np.zeros(M+2)
        xt[0] = 1  # Bias term
        xt[1:M+1] = X[max(0, t-M-1):t-1] if t-1 < M else X[t-M-1:t-1]
        xt[M+1] = Ft[t-1]
        Ft[t] = np.tanh(np.dot(xt, theta))
    return Ft

# Compute returns and Sharpe ratio
def compute_returns(X, Ft, mu=1, delta=0.001, M=5):
    T = len(Ft) - 1
    Ret = mu * (Ft[:T] * X[M:M+T] - delta * np.abs(Ft[1:T+1] - Ft[:T]))
    sharpe = sharpe_ratio(Ret)
    return Ret, sharpe

# Simplified gradient computation (approximate)
def compute_gradient(X, Ft, Ret, theta, M, T):
    dFdw = np.zeros((T+1, len(theta)))
    for t in range(1, T+1):
        xt = np.zeros(M+2)
        xt[0] = 1
        xt[1:M+1] = X[max(0, t-M-1):t-1] if t-1 < M else X[t-M-1:t-1]
        xt[M+1] = Ft[t-1]
        tanh_deriv = 1 - np.tanh(np.dot(xt, theta))**2
        dFdw[t] = tanh_deriv * xt + tanh_deriv * theta[M+1] * dFdw[t-1]
    # Approximate dS/dw (simplified, assumes dS/dR is approximated)
    dSdw = np.zeros(len(theta))
    for t in range(T):
        dSdw += Ret[t] * dFdw[t]  # Simplified gradient
    return dSdw / T

# Main RRL training loop
def train_rrl(X, M=5, epochs=100, rho=0.1):
    T = len(X) - M
    theta = np.random.randn(M+2) * 0.01
    for epoch in range(epochs):
        Ft = update_Ft(X, theta, T, M)
        Ret, sharpe = compute_returns(X, Ft, mu=1, delta=0.001, M=M)
        dSdw = compute_gradient(X, Ft, Ret, theta, M, T)
        theta += rho * dSdw
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Sharpe Ratio: {sharpe:.4f}")
    return theta, Ft

# Example usage
if __name__ == "__main__":
    prices, returns = generate_price_series(T=1000)
    theta, Ft = train_rrl(returns, M=5, epochs=100)
    print("Final Sharpe Ratio:", sharpe_ratio(compute_returns(returns, Ft)[0]))