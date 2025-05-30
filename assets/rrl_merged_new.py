import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_ncg
import ipywidgets as widgets
from ipywidgets import interact
# Import RRL modules
from RRL.rrl_class import RRL
from RRL import CoreFunctions as cf
from RRL import UtilityFunctions as uf

class ArtificialPriceGenerator:
    """Generate artificial prices as per the RRL paper"""
    
    @staticmethod
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

class RRLTrainer:
    """Enhanced RRL trainer with artificial price support"""
    
    def __init__(self):
        self.theta = None
        self.training_data = None
        self.test_data = None
        
    def generate_training_data(self, T=10000, alpha=0.9, k=3):
        """Generate artificial training data"""
        self.training_data = ArtificialPriceGenerator.generate_artificial_prices(T, alpha, k)
        return self.training_data
    
    def load_real_data(self, data):
        """Load real market data"""
        self.training_data = np.asarray(data)
        return self.training_data
    
    def train_rrl(self, M, T, optimization_function, initial_theta, transaction_costs, mu, 
                  iterations=15, use_artificial=True, **artificial_params):
        """Train RRL model with either artificial or real data"""
        
        # Generate or use existing data
        if use_artificial:
            if self.training_data is None:
                self.generate_training_data(**artificial_params)
        
        if self.training_data is None:
            raise ValueError("No training data available. Generate artificial data or load real data.")
        
        # Set parameters
        start_pos = M
        finish_pos = start_pos + T
        F = np.zeros(T + 1)
        
        # Generate returns
        X = uf.GetReturns(self.training_data)
        Xn = uf.FeatureNormalize(X)
        
        # Initialize theta
        self.theta = np.ones(M + 2) * initial_theta
        
        # Training loop
        start_pos += T
        print("Training RRL model...")
        
        for i in range(1, iterations):
            print(f"Iteration: {i}/{iterations-1}")
            
            # Option 1: Use gradient-based optimization
            self.theta = cf.train(self.theta, X, Xn, T, M, mu, transaction_costs, 
                                start_pos, 1, 500, optimization_function)
            
            # Option 2: Use scipy optimization (uncomment if preferred)
            # self.theta = fmin_ncg(cf.ObjectiveFunction, self.theta, cf.GradientFunctionM, 
            #                      args=(X, Xn, T, M, mu, transaction_costs, start_pos, optimization_function), 
            #                      avextol=1e-8, maxiter=50)
            
            start_pos += T
            
        print("Training completed.")
        
        # Compute results
        start_pos = M
        F = cf.ComputeF(self.theta, Xn, T, M, start_pos)
        rewards = cf.RewardFunction(mu, F, transaction_costs, T, M, X)
        
        return self.theta, F, rewards, X[start_pos:finish_pos]
    
    def plot_results(self, F, rewards, returns, T):
        """Plot training results"""
        # Compute cumulative rewards
        cum_rewards = rewards + 1
        for i in range(1, cum_rewards.size):
            cum_rewards[i] = cum_rewards[i-1] * cum_rewards[i]
        
        # Compute cumulative returns
        cum_returns = returns + 1
        for i in range(1, cum_returns.size):
            cum_returns[i] = cum_returns[i-1] * cum_returns[i]
        
        # Create comprehensive plot
        fig = plt.figure(figsize=(18, 15))
        
        # Plot 1: Trading signals and returns
        ax1a = fig.add_subplot(4, 1, 1)
        ax1a.plot(F[1:], 'y', label='Neuron Output')
        ax1a.set_xlabel('Days')
        ax1a.set_ylabel('Neuron Output', color='b')
        ax1a.set_title('Trading Signals and Policy Decisions')
        
        # Plot policy decisions as colored bars
        B = F[1:] > 0
        for i, b in enumerate(B):
            color = 'b' if b else 'g'
            ax1a.axvline(x=i, color=color, alpha=0.3, linewidth=1)
        
        # Plot returns on secondary axis
        ax1b = ax1a.twinx()
        ax1b.plot(cum_returns, 'r-', linewidth=2, label='Cumulative Returns')
        ax1b.set_ylabel('Returns', color='r')
        
        # Plot 2: Neuron output detail
        ax2 = fig.add_subplot(4, 1, 2)
        ax2.set_title("Neuron Output (Trading Position)")
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax2.plot(F[1:], color='r', linestyle='--', linewidth=2)
        ax2.set_ylabel('Position')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Agent rewards
        ax3 = fig.add_subplot(4, 1, 3)
        ax3.set_title("Cumulative Agent Rewards")
        ax3.plot(cum_rewards, 'g-', linewidth=2)
        ax3.set_ylabel('Cumulative Reward')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Price series
        ax4 = fig.add_subplot(4, 1, 4)
        ax4.set_title("Price Series")
        start_idx = len(self.training_data) - len(returns)
        ax4.plot(self.training_data[start_idx:start_idx + len(returns)], 'k-', linewidth=1)
        ax4.set_ylabel('Price')
        ax4.set_xlabel('Time')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print performance metrics
        final_reward = cum_rewards[-1] if len(cum_rewards) > 0 else 0
        sharpe_ratio = cf.SharpeRatio(rewards) if len(rewards) > 0 else 0
        
        print(f"\nPerformance Metrics:")
        print(f"Final Cumulative Reward: {final_reward:.4f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Total Trading Days: {len(F)-1}")

# Initialize trainer
trainer = RRLTrainer()

# Interactive training function
def interactive_fit(M, T, optimization_function, initial_theta, transaction_costs, mu, 
                   iterations, use_artificial, T_artificial, alpha, k):
    """Interactive function for widget-based training"""
    
    artificial_params = {
        'T': T_artificial,
        'alpha': alpha,
        'k': k
    }
    
    try:
        theta, F, rewards, returns = trainer.train_rrl(
            M=M, T=T, optimization_function=optimization_function,
            initial_theta=initial_theta, transaction_costs=transaction_costs,
            mu=mu, iterations=iterations, use_artificial=use_artificial,
            **artificial_params
        )
        
        trainer.plot_results(F, rewards, returns, T)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")

# Real-time learning simulation (as per the paper)
def real_time_learning_simulation(T=10000, m=8, delta=0.005, eta=0.01, rho=0.001):
    """Implement real-time recurrent learning as described in the paper"""
    
    # Generate artificial prices
    prices = ArtificialPriceGenerator.generate_artificial_prices(T=T)
    returns = np.diff(prices, prepend=prices[0])
    
    # Initialize parameters
    n_params = m + 2
    theta = np.random.normal(0, 0.1, n_params)
    
    # Initialize trader variables
    F_prev = 0.0
    P_prev = np.zeros(n_params)
    A = 0.0  # Moving average of returns
    B = 0.0  # Moving average of squared returns
    
    # Storage
    F_list = [F_prev]
    R_list = []
    A_list = []
    B_list = []
    
    print("Running real-time learning simulation...")
    
    for t in range(1, T + 1):
        if t % 1000 == 0:
            print(f"Processing step {t}/{T}")
            
        r_t = returns[t - 1]
        
        # Construct input vector
        x_t = np.zeros(n_params)
        x_t[0] = 1.0  # Bias
        x_t[1] = F_prev
        for i in range(1, m + 1):
            if t - i >= 1:
                x_t[i + 1] = returns[t - i - 1]
            else:
                x_t[i + 1] = 0.0
        
        # Compute trader output
        a_t = np.dot(theta, x_t)
        F_t = np.tanh(a_t)
        
        # Compute gradient using RTRL
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
    
    # Plot results
    plot_real_time_results(prices, F_list, R_list, A_list, B_list)
    
    return theta, F_list, R_list, prices

def plot_real_time_results(prices, F_list, R_list, A_list, B_list):
    """Plot results from real-time learning simulation"""
    
    cum_profit = np.cumsum(R_list)
    
    # Compute Sharpe ratio
    sharpe_ratio = []
    for t in range(len(A_list)):
        if B_list[t] - A_list[t]**2 > 0:
            S_t = A_list[t] / np.sqrt(B_list[t] - A_list[t]**2)
        else:
            S_t = 0.0
        sharpe_ratio.append(S_t)
    
    # Time series plots
    fig, axs = plt.subplots(4, 1, figsize=(12, 16))
    
    axs[0].plot(prices)
    axs[0].set_title('Price Series')
    axs[0].grid(True, alpha=0.3)
    
    axs[1].plot(F_list[1:])
    axs[1].set_title('Trading Signals (F_t)')
    axs[1].grid(True, alpha=0.3)
    
    axs[2].plot(cum_profit)
    axs[2].set_title('Cumulative Profit')
    axs[2].grid(True, alpha=0.3)
    
    axs[3].plot(sharpe_ratio)
    axs[3].set_title('Exponential Moving Sharpe Ratio')
    axs[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Histograms
    T = len(R_list)
    mid_point = T // 2
    
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    
    # Price changes
    returns = np.diff(prices)
    axs[0, 0].hist(returns[:mid_point], bins=50, alpha=0.7)
    axs[0, 0].set_title('Price Changes (First Half)')
    axs[0, 1].hist(returns[mid_point:], bins=50, alpha=0.7)
    axs[0, 1].set_title('Price Changes (Second Half)')
    
    # Trading profits
    axs[1, 0].hist(R_list[:mid_point], bins=50, alpha=0.7)
    axs[1, 0].set_title('Trading Profits (First Half)')
    axs[1, 1].hist(R_list[mid_point:], bins=50, alpha=0.7)
    axs[1, 1].set_title('Trading Profits (Second Half)')
    
    # Sharpe ratios
    axs[2, 0].hist(sharpe_ratio[:mid_point], bins=50, alpha=0.7)
    axs[2, 0].set_title('Sharpe Ratios (First Half)')
    axs[2, 1].hist(sharpe_ratio[mid_point:], bins=50, alpha=0.7)
    axs[2, 1].set_title('sharpe ratios (Second Half)')
    plt.tight_layout()
    plt.show()
def main():
    """Main function to run RRL training and simulation"""
    
    # Interactive widget-based training
    print("=== Interactive RRL Training ===")
    interact(interactive_fit, 
             M=widgets.IntSlider(min=5, max=50, step=1, value=8, description='Input Window (M)'),
             T=widgets.IntSlider(min=100, max=2000, step=50, value=500, description='Training Window (T)'),
             optimization_function=widgets.Dropdown(options=["return", "sharpe_ratio"], value="sharpe_ratio", description='Optimization'),
             initial_theta=widgets.FloatSlider(min=0.1, max=2.0, step=0.1, value=1.0, description='Initial Theta'),
             transaction_costs=widgets.FloatSlider(min=0.001, max=0.01, step=0.001, value=0.005, description='Transaction Costs'),
             mu=widgets.IntSlider(min=1, max=10, step=1, value=1, description='Shares'),
             iterations=widgets.IntSlider(min=5, max=30, step=1, value=15, description='Iterations'),
             use_artificial=widgets.Checkbox(value=True, description='Use Artificial Data'),
             T_artificial=widgets.IntSlider(min=5000, max=20000, step=1000, value=10000, description='Artificial Data Length'),
             alpha=widgets.FloatSlider(min=0.1, max=0.99, step=0.01, value=0.9, description='Alpha'),
             k=widgets.FloatSlider(min=1, max=10, step=0.5, value=3, description='K'))
    
    # Run real-time learning simulation
    print("\n=== Real-Time Learning Simulation ===")
    theta, F_list, R_list, prices = real_time_learning_simulation(T=10000, m=8, delta=0.005, eta=0.01, rho=0.001)
    
    print("Simulation completed successfully!")

if __name__ == "__main__":
    main()
