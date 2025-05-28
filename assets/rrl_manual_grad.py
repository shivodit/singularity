import numpy as np
import matplotlib.pyplot as plt

# Function to generate artificial prices as per the paper
def generate_artificial_prices(T=10000, alpha=0.9, k=3):
    eps = np.random.normal(0, 1, T)
    nu = np.random.normal(0, 1, T)
    beta = np.zeros(T + 1)
    p = np.zeros(T + 1)
    for t in range(1, T + 1):
        beta[t] = alpha * beta[t-1] + nu[t-1]
        p[t] = p[t-1] + beta[t-1] + k * eps[t-1]
    R = np.max(p[1:]) - np.min(p[1:])
    z = np.exp(p[1:] / R)
    return z

# Parameters
T = 10000  # Time steps
m = 8  # Number of lagged returns
n_params = m + 2  # w (bias), u (F_{t-1}), v_1 to v_8
theta = np.random.normal(0, 0.1, n_params)  # Random initialization
delta = 0.005  # Transaction cost 0.5%
eta = 0.01  # Decay factor for moving averages (1/eta = 100 periods)
rho = 0.001  # Learning rate

# Generate prices and returns
prices = generate_artificial_prices(T=T)
returns = np.diff(prices, prepend=prices[0])  # r_t = z_t - z_{t-1}

# Initialize trader variables
F_prev = 0.0  # Initial position F_0
P_prev = np.zeros(n_params)  # Gradient dF_t/dθ
A = 0.0  # Moving average of returns
B = 0.0  # Moving average of squared returns

# Storage for results
F_list = [F_prev]
R_list = []
A_list = []
B_list = []

# Real-time recurrent learning simulation
for t in range(1, T + 1):
    r_t = returns[t - 1]
    # Construct input vector: [1, F_{t-1}, r_{t-1}, ..., r_{t-8}]
    x_t = np.zeros(n_params)
    x_t[0] = 1.0  # Bias term
    x_t[1] = F_prev
    for i in range(1, m + 1):
        if t - i >= 1:
            x_t[i + 1] = returns[t - i - 1]
        else:
            x_t[i + 1] = 0.0
    # Compute trader output
    a_t = np.dot(theta, x_t)
    F_t = np.tanh(a_t)  # Continuous position in [-1, 1]
    # Compute gradient dF_t/dθ using RTRL
    P_t = (1 - F_t**2) * (x_t + theta[1] * P_prev)
    # Compute trading return
    R_t = F_prev * r_t - delta * np.abs(F_t - F_prev)
    # Update moving averages
    A = A + eta * (R_t - A)
    B = B + eta * (R_t**2 - B)
    # Compute differential Sharpe ratio gradient
    if B - A**2 > 0:
        dD_dR_t = (B - A * R_t) / (B - A**2)**1.5
    else:
        dD_dR_t = 0.0
    # Compute dR_t/dθ
    sign_diff = np.sign(F_t - F_prev) if F_t != F_prev else 0
    dR_dtheta = r_t * P_prev - delta * sign_diff * (P_t - P_prev)
    # Update parameters
    delta_theta = rho * dD_dR_t * dR_dtheta
    theta = theta + delta_theta
    # Store results
    F_list.append(F_t)
    R_list.append(R_t)
    A_list.append(A)
    B_list.append(B)
    # Update previous values
    F_prev = F_t
    P_prev = P_t

# Compute cumulative profit and Sharpe ratio
cum_profit = np.cumsum(R_list)
sharpe_ratio = []
for t in range(T):
    if B_list[t] - A_list[t]**2 > 0:
        S_t = A_list[t] / np.sqrt(B_list[t] - A_list[t]**2)
    else:
        S_t = 0.0
    sharpe_ratio.append(S_t)

# Figure 2: Time series plots
fig, axs = plt.subplots(4, 1, figsize=(12, 16))
axs[0].plot(prices)
axs[0].set_title('Price Series')
axs[1].plot(F_list[1:])
axs[1].set_title('Trading Signals (F_t)')
axs[2].plot(cum_profit)
axs[2].set_title('Cumulative Profit')
axs[3].plot(sharpe_ratio)
axs[3].set_title('Exponential Moving Sharpe Ratio')
plt.tight_layout()
plt.savefig('figure_2.png')
plt.close()

# Figure 4: Histograms
fig, axs = plt.subplots(3, 2, figsize=(12, 12))
# Price changes
axs[0, 0].hist(returns[:5000], bins=50)
axs[0, 0].set_title('Price Changes (First 5000)')
axs[0, 1].hist(returns[5000:], bins=50)
axs[0, 1].set_title('Price Changes (Last 5000)')
# Trading profits
axs[1, 0].hist(R_list[:5000], bins=50)
axs[1, 0].set_title('Trading Profits (First 5000)')
axs[1, 1].hist(R_list[5000:], bins=50)
axs[1, 1].set_title('Trading Profits (Last 5000)')
# Sharpe ratios
axs[2, 0].hist(sharpe_ratio[:5000], bins=50)
axs[2, 0].set_title('Sharpe Ratios (First 5000)')
axs[2, 1].hist(sharpe_ratio[5000:], bins=50)
axs[2, 1].set_title('Sharpe Ratios (Last 5000)')
plt.tight_layout()
plt.savefig('figure_4.png')
plt.close()

print("Simulation complete. Plots saved as 'figure_2.png' and 'figure_4.png'.")