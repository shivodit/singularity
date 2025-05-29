import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate artificial prices
def generate_artificial_prices(n=10000, alpha=0.9, k=3):
    p = np.zeros(n)
    beta = np.zeros(n)
    for t in range(1, n):
        beta[t] = alpha * beta[t-1] + np.random.normal(0, 1)
        p[t] = p[t-1] + beta[t-1] + k * np.random.normal(0, 1)
    R = p.max() - p.min()
    z = np.exp(p / R)
    return z

z_np = generate_artificial_prices()
# Compute returns
r_np = np.zeros(10000)
r_np[1:] = z_np[1:] - z_np[:-1]
r_torch = torch.tensor(r_np, dtype=torch.float32)

# Define model
class TraderModel(nn.Module):
    def __init__(self):
        super(TraderModel, self).__init__()
        self.linear = nn.Linear(9, 1)  # F_{t-1} + 8 returns

    def forward(self, x):
        s_t = self.linear(x)
        F_t = torch.tanh(s_t)
        return F_t

# Initialize model and optimizer
model = TraderModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 100
T_train = 2000

# Training loop
for epoch in range(num_epochs):
    F_prev = torch.tensor(0.0)
    R_list = []
    for t in range(1, T_train + 1):
        past_r_indices = [max(0, t - k) for k in range(8)]
        past_r = [r_torch[idx] for idx in past_r_indices]
        past_r_tensor = torch.stack(past_r)
        X_t = torch.cat([F_prev.unsqueeze(0), past_r_tensor])
        F_t = model(X_t).squeeze()
        r_t = r_torch[t]
        R_t = F_prev * r_t
        R_list.append(R_t)
        F_prev = F_t
    R_tensor = torch.stack(R_list)
    mean_R = R_tensor.mean()
    std_R = R_tensor.std(unbiased=False)
    S_T = mean_R / std_R if std_R > 0 else torch.tensor(0.0)
    loss = -S_T
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Sharpe Ratio: {S_T.item()}")

# Testing on t=2001 to 4000
T_test_start = T_train + 1
T_test_end = T_train + 2000
with torch.no_grad():
    # Get F_T from training set
    F_prev = torch.tensor(0.0)
    for t in range(1, T_train + 1):
        past_r_indices = [max(0, t - k) for k in range(8)]
        past_r = [r_torch[idx] for idx in past_r_indices]
        past_r_tensor = torch.stack(past_r)
        X_t = torch.cat([F_prev.unsqueeze(0), past_r_tensor])
        F_t = model(X_t).squeeze()
        F_prev = F_t
    # Now test on t=T_test_start to T_test_end
    R_list_test = []
    for t in range(T_test_start, T_test_end + 1):
        if t >= len(r_torch):
            break
        past_r_indices = [max(0, t - k) for k in range(8)]
        past_r = [r_torch[idx] if idx < len(r_torch) else torch.tensor(0.0) for idx in past_r_indices]
        past_r_tensor = torch.stack([val if isinstance(val, torch.Tensor) else torch.tensor(val) for val in past_r])
        X_t = torch.cat([F_prev.unsqueeze(0), past_r_tensor])
        F_t = model(X_t).squeeze()
        r_t = r_torch[t]
        R_t = F_prev * r_t
        R_list_test.append(R_t)
        F_prev = F_t
    if R_list_test:
        R_tensor_test = torch.stack(R_list_test)
        mean_R_test = R_tensor_test.mean()
        std_R_test = R_tensor_test.std(unbiased=False)
        S_T_test = mean_R_test / std_R_test if std_R_test > 0 else torch.tensor(0.0)
        print(f"Test Sharpe Ratio (t={T_test_start} to {T_test_end}): {S_T_test.item()}")
    else:
        print("No test data available")
