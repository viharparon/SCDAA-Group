import numpy as np
import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class LQR:
    def __init__(self, H, M, sigma, C, D, R, T, time_grid):
        self.H = H.numpy()
        self.M = M.numpy()
        self.sigma = sigma
        self.C = C.numpy()
        self.D = D.numpy()
        self.R = R.numpy()
        self.T = T
        self.time_grid = time_grid.numpy()
        self.S = self.solve_riccati()
    
    def riccati_ode(self, t, S_flat):
        """ Computes dS/dt for the Riccati ODE using NumPy arrays. """
        S = S_flat.reshape(2, 2)  # Reshape flattened S
        D_inv = np.linalg.inv(self.D)  # Compute inverse using NumPy
        dSdt = S @ self.M @ D_inv @ self.M.T @ S - self.H.T @ S - S @ self.H - self.C
        return dSdt.flatten()
    
    def solve_riccati(self):
        """ Solves the Riccati equation backwards in time. """
        sol = solve_ivp(
            self.riccati_ode,
            [self.T, 0],  # Integrate backward from T to 0
            self.R.flatten(),  # Ensure NumPy array
            t_eval=self.time_grid[::-1],  # Reverse time grid for backward integration
            method='RK45',
        )
        return {t: torch.tensor(S.reshape(2, 2), dtype=torch.float32) for t, S in zip(sol.t[::-1], sol.y.T[::-1])}  # Reverse solution
    
    def value_function(self, t, x):
        """ Computes v(t, x) = x^T S(t) x + integral term. """
        S_t = self.S[min(self.time_grid, key=lambda tn: abs(tn - t))]  # Closest grid point
        integral_term = sum(torch.trace(self.sigma @ self.sigma.T @ self.S[tau]) * (self.time_grid[1] - self.time_grid[0]) for tau in self.time_grid if tau >= t)
        return x.T @ S_t @ x + integral_term
    
    def optimal_control(self, t, x):
        """ Computes a(t, x) = -D^(-1) M^T S(t) x. """
        S_t = self.S[min(self.time_grid, key=lambda tn: abs(tn - t))]
        return -torch.linalg.inv(torch.tensor(self.D)) @ torch.tensor(self.M.T) @ S_t @ x
    
    def explicit_scheme(self, x0, N):
        dt = self.T / N
        x = x0.clone()
        for _ in range(N):
            a = self.optimal_control(0, x)
            x += (torch.tensor(self.H) @ x + torch.tensor(self.M) @ a) * dt + self.sigma @ torch.randn(2, 1) * np.sqrt(dt)
        return x
    
    def implicit_scheme(self, x0, N):
        dt = self.T / N
        x = x0.clone()
        I = torch.eye(2)
        for _ in range(N):
            a = self.optimal_control(0, x)
            x = torch.linalg.solve(I - dt * torch.tensor(self.H), x + dt * torch.tensor(self.M) @ a + self.sigma @ torch.randn(2, 1) * np.sqrt(dt))
        return x

# Test Cases for Exercise 1.2
def test_lqr_monte_carlo():
    H = torch.tensor([[0.5, 0.5], [0.0, 0.5]])
    M = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    sigma = torch.eye(2) * 0.5
    C = torch.tensor([[1.0, 0.1], [0.1, 1.0]])
    D = torch.tensor([[1.0, 0.1], [0.1, 1.0]]) * 0.1
    R = torch.tensor([[1.0, 0.3], [0.3, 1.0]]) * 10.0
    T = 0.5
    time_grid = torch.linspace(0, T, 100)
    lqr = LQR(H, M, sigma, C, D, R, T, time_grid)
    
    x0 = torch.tensor([[1.0], [1.0]])
    N_values = [2**i for i in range(1, 12)]
    explicit_errors = []
    implicit_errors = []
    
    for N in N_values:
        x_explicit = lqr.explicit_scheme(x0, N)
        x_implicit = lqr.implicit_scheme(x0, N)
        true_value = lqr.value_function(0, x0)
        explicit_errors.append(torch.norm(true_value - x_explicit.T @ lqr.S[0] @ x_explicit).item())
        implicit_errors.append(torch.norm(true_value - x_implicit.T @ lqr.S[0] @ x_implicit).item())
    
    plt.figure(figsize=(8, 5))
    plt.loglog(N_values, explicit_errors, '-o', label='Explicit Scheme')
    plt.loglog(N_values, implicit_errors, '-s', label='Implicit Scheme')
    plt.xlabel('Number of Time Steps')
    plt.ylabel('Error')
    plt.title('Convergence of Explicit vs Implicit Schemes')
    plt.legend()
    plt.show()

test_lqr_monte_carlo()