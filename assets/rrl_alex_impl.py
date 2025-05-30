import numpy as np
import matplotlib.pyplot as plt
import math

# Utility Functions (standalone implementations)
def GetReturns(data):
    """
    Compute returns from price data.
    Returns: r_t = (p_t - p_{t-1}) / p_{t-1}
    """
    data = np.asarray(data)
    returns = np.zeros(len(data))
    returns[1:] = (data[1:] - data[:-1]) / data[:-1]
    return returns

def FeatureNormalize(X):
    """
    Normalize features using z-score normalization.
    """
    X = np.asarray(X)
    mean_X = np.mean(X)
    std_X = np.std(X)
    if std_X == 0:
        return X - mean_X
    return (X - mean_X) / std_X

# Core Functions (standalone implementations)
def ComputeF(theta, Xn, T, M, startPos):
    """
    Compute the neuron output F_t using the RRL model.
    
    Parameters:
    - theta: model parameters
    - Xn: normalized returns
    - T: time window size
    - M: number of lagged returns
    - startPos: starting position in the data
    
    Returns:
    - F: array of neuron outputs (positions)
    """
    F = np.zeros(T + 1)  # Initialize with F_0 = 0
    
    for t in range(1, T + 1):
        # Create input vector: [1, F_{t-1}, r_{t-1}, ..., r_{t-M}]
        x_t = np.zeros(M + 2)  # bias + F_{t-1} + M lagged returns
        x_t[0] = 1.0  # bias term
        x_t[1] = F[t-1]  # previous position
        
        # Add lagged returns
        for i in range(M):
            idx = startPos + t - 1 - i - 1  # t-1-i for lagged returns
            if idx >= 0 and idx < len(Xn):
                x_t[i + 2] = Xn[idx]
            else:
                x_t[i + 2] = 0.0
        
        # Compute neuron output
        s_t = np.dot(theta, x_t)
        F[t] = np.tanh(s_t)
    
    return F

def RewardFunction(mu, F, transactionCosts, T, M, X):
    """
    Compute trading rewards with transaction costs.
    
    Parameters:
    - mu: scaling factor
    - F: position array
    - transactionCosts: transaction cost rate
    - T: time window
    - M: starting offset
    - X: returns array
    
    Returns:
    - rewards: array of trading rewards
    """
    rewards = np.zeros(T)
    
    for t in range(T):
        # Trading return: F_{t-1} * r_t
        trading_return = F[t] * X[M + t]
        
        # Transaction cost: |F_t - F_{t-1}|
        transaction_cost = transactionCosts * abs(F[t + 1] - F[t])
        
        # Net reward
        rewards[t] = mu * (trading_return - transaction_cost)
    
    return rewards

def sharpe_ratio(returns):
    """
    Compute Sharpe ratio of returns.
    """
    returns = np.asarray(returns)
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0.0
    
    return mean_return / std_return

def generate_sample_data(n_points=1000, seed=42):
    """
    Generate sample price data for testing.
    """
    np.random.seed(seed)
    
    # Generate random walk with drift
    returns = np.random.normal(0.001, 0.02, n_points)  # 0.1% daily return, 2% volatility
    prices = np.zeros(n_points)
    prices[0] = 100.0  # Starting price
    
    for i in range(1, n_points):
        prices[i] = prices[i-1] * (1 + returns[i])
    
    return prices

# RRL Training Functions
def compute_gradient(theta, Xn, T, M, startPos, mu, transactionCosts, X):
    """
    Compute gradient of the objective function (Sharpe ratio) with respect to theta.
    """
    F = np.zeros(T + 1)
    dF_dtheta = np.zeros((T + 1, len(theta)))
    
    # Forward pass with gradient computation
    for t in range(1, T + 1):
        # Create input vector
        x_t = np.zeros(M + 2)
        x_t[0] = 1.0
        x_t[1] = F[t-1]
        
        for i in range(M):
            idx = startPos + t - 1 - i - 1
            if idx >= 0 and idx < len(Xn):
                x_t[i + 2] = Xn[idx]
        
        # Compute F_t
        s_t = np.dot(theta, x_t)
        F[t] = np.tanh(s_t)
        
        # Compute gradient dF_t/dtheta
        tanh_derivative = 1 - F[t]**2
        dF_dtheta[t] = tanh_derivative * (x_t + theta[1] * dF_dtheta[t-1])
    
    # Compute rewards and their gradients
    rewards = RewardFunction(mu, F, transactionCosts, T, M, X)
    
    # Gradient of Sharpe ratio
    mean_R = np.mean(rewards)
    var_R = np.var(rewards)
    std_R = np.sqrt(var_R)
    
    if std_R == 0:
        return np.zeros_like(theta)
    
    # Gradient of rewards with respect to theta
    dR_dtheta = np.zeros((T, len(theta)))
    for t in range(T):
        # dR_t/dtheta = mu * (X[M+t] * dF_t/dtheta - transactionCosts * sign(F_{t+1} - F_t) * (dF_{t+1}/dtheta - dF_t/dtheta))
        trading_grad = mu * X[M + t] * dF_dtheta[t]
        
        if t < T - 1:
            position_change = F[t + 2] - F[t + 1] if t + 2 <= T else 0
            sign_change = np.sign(position_change) if position_change != 0 else 0
            transaction_grad = mu * transactionCosts * sign_change * (dF_dtheta[t + 1] - dF_dtheta[t])
        else:
            transaction_grad = np.zeros_like(dF_dtheta[t])
        
        dR_dtheta[t] = trading_grad - transaction_grad
    
    # Gradient of Sharpe ratio
    dSharpe_dtheta = np.zeros_like(theta)
    for i in range(len(theta)):
        dMean_dtheta_i = np.mean(dR_dtheta[:, i])
        dVar_dtheta_i = 2 * np.mean((rewards - mean_R) * dR_dtheta[:, i])
        
        numerator = dMean_dtheta_i * std_R - mean_R * (0.5 / std_R) * dVar_dtheta_i
        dSharpe_dtheta[i] = numerator / var_R
    
    return dSharpe_dtheta

def train_rrl(theta_init, Xn, T, M, startPos, X, mu=1, transactionCosts=0.001, 
              learning_rate=0.01, max_iterations=100, tolerance=1e-6):
    """
    Train RRL model using gradient ascent on Sharpe ratio.
    """
    theta = theta_init.copy()
    sharpe_history = []
    
    for iteration in range(max_iterations):
        # Compute current Sharpe ratio
        F = ComputeF(theta, Xn, T, M, startPos)
        rewards = RewardFunction(mu, F, transactionCosts, T, M, X)
        current_sharpe = sharpe_ratio(rewards)
        sharpe_history.append(current_sharpe)
        
        # Compute gradient
        gradient = compute_gradient(theta, Xn, T, M, startPos, mu, transactionCosts, X)
        
        # Update parameters
        theta_new = theta + learning_rate * gradient
        
        # Check convergence
        if np.linalg.norm(theta_new - theta) < tolerance:
            print(f"Converged after {iteration + 1} iterations")
            break
        
        theta = theta_new
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Sharpe Ratio: {current_sharpe:.4f}")
    
    return theta, sharpe_history

# Main execution function (replicating the notebook functionality)
def main():
    # Generate or load testing data
    print("Generating sample data...")
    testingData = generate_sample_data(n_points=500, seed=42)
    
    # Parameters (matching the notebook)
    T = 100
    M = 10
    mu = 1
    transactionCosts = 0.001
    startPos = M
    finishPos = startPos + T
    
    # Initialize theta (model parameters)
    np.random.seed(42)
    theta = np.random.normal(0, 0.1, M + 2)  # bias + F_{t-1} + M lagged returns
    
    print("Processing data...")
    # Generate the returns
    X = GetReturns(testingData)
    Xn = FeatureNormalize(X)
    
    print("Training RRL model...")
    # Train the model (optional - you can skip this and use random theta)
    # theta_trained, sharpe_history = train_rrl(theta, Xn, T, M, startPos, X, mu, transactionCosts)
    # theta = theta_trained
    
    print("Computing neuron output...")
    # Compute the neuron output
    F = ComputeF(theta, Xn, T, M, startPos)
    
    print("Computing rewards...")
    # Compute the rewards
    rewards = RewardFunction(mu, F, transactionCosts, T, M, X)
    
    print("Processing cumulative returns...")
    # Compute the cumulative reward
    cumulative_rewards = rewards + 1  # Add one to prevent vanishing
    for i in range(1, len(cumulative_rewards)):
        cumulative_rewards[i] = cumulative_rewards[i-1] * cumulative_rewards[i]
    
    # Get the cumulative returns
    returns = X[startPos:finishPos] + 1
    for i in range(1, len(returns)):
        returns[i] = returns[i-1] * returns[i]
    
    print("Creating plots...")
    # Create figure for plotting (replicating the notebook visualization)
    fig = plt.figure(figsize=(18, 15))
    
    # Plot rewards and policy decisions
    ax1a = fig.add_subplot(3, 1, 1)
    ## Plot neuron output
    t = np.arange(0.0, len(F[1:]), 1)
    ax1a.plot(F[1:], 'y', label='Neuron Output')
    ax1a.set_xlabel('Days')
    ax1a.set_ylabel('Neuron Output', color='b')
    for tl in ax1a.get_yticklabels():
        tl.set_color('b')
    
    ## Plot the policy output
    B = F[1:] > 0
    for i, b in enumerate(B):
        if b:
            plt.plot([i, i], [-1, 1], color='b', linestyle='-', linewidth=2, alpha=0.3)
        else:
            plt.plot([i, i], [-1, 1], color='g', linestyle='-', linewidth=2, alpha=0.3)
    
    ## Plot returns
    ax1b = ax1a.twinx()
    ax1b.plot(returns, 'r-', linewidth=3, label='Cumulative Returns')
    ax1b.set_ylabel('Returns', color='r')
    for tl in ax1b.get_yticklabels():
        tl.set_color('r')
    
    # Plot Neuron output
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.set_title("Neuron Output")
    ax2.plot([0, T], [0, 0], color='k', linestyle='-', linewidth=2)
    ax2.plot(F[1:], color='r', linestyle='--', linewidth=2)
    ax2.set_ylim([-1.1, 1.1])
    ax2.set_xlim([0, len(F[1:])])
    ax2.grid(True, alpha=0.3)
    
    # Plot agent rewards
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.set_title("Agent Rewards")
    ax3.plot(cumulative_rewards, 'g-', linewidth=2)
    ax3.set_xlim([0, len(cumulative_rewards)])
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(f"Training Period: {T} days")
    print(f"Number of lagged returns: {M}")
    print(f"Transaction costs: {transactionCosts*100:.3f}%")
    print(f"Final Sharpe Ratio: {sharpe_ratio(rewards):.4f}")
    print(f"Total Cumulative Reward: {cumulative_rewards[-1]:.4f}")
    print(f"Total Cumulative Return: {returns[-1]:.4f}")
    print(f"Average Daily Reward: {np.mean(rewards):.6f}")
    print(f"Reward Volatility: {np.std(rewards):.6f}")
    print(f"Max Position: {np.max(F[1:]):.4f}")
    print(f"Min Position: {np.min(F[1:]):.4f}")
    print(f"Average Position: {np.mean(F[1:]):.4f}")
    
    # Return results for further analysis
    return {
        'theta': theta,
        'F': F,
        'rewards': rewards,
        'cumulative_rewards': cumulative_rewards,
        'returns': returns,
        'X': X,
        'Xn': Xn,
        'testingData': testingData,
        'sharpe_ratio': sharpe_ratio(rewards)
    }

main()
