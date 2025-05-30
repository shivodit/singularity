import numpy as np
import matplotlib.pyplot as plt
import math

# Fixed Utility Functions
def GetReturns(X):
    """
    Compute returns from price data.
    Fixed to handle edge cases and prevent negative dimensions.
    """
    X = np.asarray(X)
    
    # Check if we have enough data points
    if X.size < 2:
        raise ValueError(f"Need at least 2 data points to compute returns, got {X.size}")
    
    # Compute returns: r_t = (p_t - p_{t-1}) / p_{t-1}
    returns = np.zeros(X.size)
    returns[1:] = (X[1:] - X[:-1]) / X[:-1]
    
    return returns

def FeatureNormalize(X):
    """
    Normalize features using z-score normalization.
    Fixed to handle edge cases.
    """
    X = np.asarray(X)
    
    if X.size == 0:
        return X
    
    mean_X = np.mean(X)
    std_X = np.std(X)
    
    # Handle case where all values are the same (std = 0)
    if std_X == 0:
        return X - mean_X  # This will be all zeros
    
    return (X - mean_X) / std_X

# Data Generation Function
def generate_sufficient_training_data(n_points=5000, seed=42):
    """
    Generate sufficient training data for RRL.
    """
    np.random.seed(seed)
    
    # Generate price series using geometric Brownian motion
    dt = 1/252  # Daily time step
    mu_drift = 0.05  # 5% annual drift
    sigma = 0.2  # 20% annual volatility
    
    # Initialize price array
    prices = np.zeros(n_points)
    prices[0] = 100.0  # Starting price
    
    # Generate random increments
    random_increments = np.random.normal(0, 1, n_points-1)
    
    # Generate price path
    for i in range(1, n_points):
        drift = mu_drift * dt
        diffusion = sigma * np.sqrt(dt) * random_increments[i-1]
        prices[i] = prices[i-1] * np.exp(drift + diffusion)
    
    return prices

# Core RRL Functions
def ComputeF(theta, Xn, T, M, startPos):
    """
    Compute the neuron output F_t using the RRL model.
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

# Generate training data
trainingData = generate_sufficient_training_data(n_points=2000)
print(f"Generated training data with {len(trainingData)} points")

# Fixed fit function
def fit(M, T, optimizationFunction, initialTheta, transactionCosts, mu):
    """
    Fixed version of the fit function with proper error handling.
    """
    # Validate inputs
    required_points = M + T + 50  # Add extra buffer
    if len(trainingData) < required_points:
        print(f"Warning: Need at least {required_points} data points, got {len(trainingData)}")
        print("Generating more data...")
        global trainingData
        trainingData = generate_sufficient_training_data(n_points=required_points + 100)
    
    # Initialize arrays
    F = np.zeros(T + 1)
    dF = np.zeros(T + 1)
    
    try:
        # Generate the returns with error handling
        X = GetReturns(trainingData)
        print(f"✓ Generated {len(X)} returns from {len(trainingData)} prices")
        
        if len(X) < M + T:
            raise ValueError(f"Insufficient returns after processing: need {M + T}, got {len(X)}")
        
        Xn = FeatureNormalize(X)
        print(f"✓ Normalized returns successfully")
        
        # Initialize theta
        if initialTheta == "random":
            theta = np.random.normal(0, 0.1, M + 2)
            print(f"✓ Initialized random theta with {len(theta)} parameters")
        else:
            theta = np.array(initialTheta)
            if len(theta) != M + 2:
                raise ValueError(f"Initial theta must have {M + 2} parameters, got {len(theta)}")
            print(f"✓ Using provided theta with {len(theta)} parameters")
        
        # Set starting position
        startPos = M
        
        # Compute neuron output
        F = ComputeF(theta, Xn, T, M, startPos)
        print(f"✓ Computed neuron outputs")
        
        # Compute rewards
        rewards = RewardFunction(mu, F, transactionCosts, T, M, X)
        print(f"✓ Computed rewards")
        
        # Calculate performance metrics
        sharpe = sharpe_ratio(rewards)
        total_return = np.sum(rewards)
        
        print(f"✓ Sharpe Ratio: {sharpe:.4f}")
        print(f"✓ Total Return: {total_return:.4f}")
        
        return theta, F, X, Xn, rewards
        
    except Exception as e:
        print(f"✗ Error in fit function: {str(e)}")
        raise

# Test the function with sample parameters
try:
    print("Testing fit function...")
    M = 10
    T = 100
    optimizationFunction = "sharpe"
    initialTheta = "random"
    transactionCosts = 0.001
    mu = 1
    
    theta, F, X, Xn, rewards = fit(M, T, optimizationFunction, initialTheta, transactionCosts, mu)
    print("✓ Function executed successfully!")
    
    # Replicate the original notebook visualization
    print("\nGenerating visualization...")
    
    # Parameters for visualization
    startPos = M
    finishPos = startPos + T
    
    # Compute cumulative rewards
    cumulative_rewards = rewards + 1  # Add one to prevent vanishing
    for i in range(1, len(cumulative_rewards)):
        cumulative_rewards[i] = cumulative_rewards[i-1] * cumulative_rewards[i]
    
    # Get cumulative returns
    returns_viz = X[startPos:finishPos] + 1
    for i in range(1, len(returns_viz)):
        returns_viz[i] = returns_viz[i-1] * returns_viz[i]
    
    # Create figure for plotting (replicating the original notebook)
    fig = plt.figure(figsize=(18, 15))
    
    # Plot 1: Rewards and policy decisions
    ax1a = fig.add_subplot(3, 1, 1)
    ax1a.plot(F[1:], 'y', linewidth=2, label='Neuron Output')
    ax1a.set_xlabel('Days')
    ax1a.set_ylabel('Neuron Output', color='b')
    ax1a.set_title('Trading Positions and Returns')
    for tl in ax1a.get_yticklabels():
        tl.set_color('b')
    
    # Plot the policy output (buy/sell signals)
    B = F[1:] > 0
    for i, b in enumerate(B):
        if b:
            plt.plot([i, i], [-1, 1], color='b', linestyle='-', linewidth=2, alpha=0.3)
        else:
            plt.plot([i, i], [-1, 1], color='g', linestyle='-', linewidth=2, alpha=0.3)
    
    # Plot returns on secondary y-axis
    ax1b = ax1a.twinx()
    ax1b.plot(returns_viz, 'r-', linewidth=3, label='Cumulative Returns')
    ax1b.set_ylabel('Returns', color='r')
    for tl in ax1b.get_yticklabels():
        tl.set_color('r')
    
    # Plot 2: Neuron output
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.set_title("Neuron Output")
    ax2.plot([0, T], [0, 0], color='k', linestyle='-', linewidth=2)
    ax2.plot(F[1:], color='r', linestyle='--', linewidth=2)
    ax2.set_ylim([-1.1, 1.1])
    ax2.set_xlim([0, len(F[1:])])
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel('Position')
    ax2.set_xlabel('Time Steps')
    
    # Plot 3: Agent rewards
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.set_title("Agent Rewards")
    ax3.plot(cumulative_rewards, 'g-', linewidth=2)
    ax3.set_xlim([0, len(cumulative_rewards)])
    ax3.grid(True, alpha=0.3)
    ax3.set_ylabel('Cumulative Rewards')
    ax3.set_xlabel('Time Steps')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Training Period: {T} days")
    print(f"Number of lagged returns: {M}")
    print(f"Transaction costs: {transactionCosts*100:.3f}%")
    print(f"Sharpe Ratio: {sharpe_ratio(rewards):.4f}")
    print(f"Total Cumulative Reward: {cumulative_rewards[-1]:.4f}")
    print(f"Total Cumulative Return: {returns_viz[-1]:.4f}")
    print(f"Average Daily Reward: {np.mean(rewards):.6f}")
    print(f"Reward Volatility: {np.std(rewards):.6f}")
    print(f"Max Position: {np.max(F[1:]):.4f}")
    print(f"Min Position: {np.min(F[1:]):.4f}")
    print(f"Average Position: {np.mean(F[1:]):.4f}")
    print(f"Number of position changes: {np.sum(np.abs(np.diff(F[1:])) > 0.01)}")
    
except Exception as e:
    print(f"✗ Error occurred: {e}")
    import traceback
    traceback.print_exc()
