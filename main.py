import numpy as np
import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class LQR:
    def __init__(self, H, M, sigma, C, D, R, T, time_grid):
        """Initialize the LQR solver with problem parameters."""
        # Store parameters as NumPy arrays for solve_ivp compatibility
        self.H = H.numpy() if isinstance(H, torch.Tensor) else H
        self.M = M.numpy() if isinstance(M, torch.Tensor) else M
        self.sigma = sigma
        self.C = C.numpy() if isinstance(C, torch.Tensor) else C
        self.D = D.numpy() if isinstance(D, torch.Tensor) else D
        self.R = R.numpy() if isinstance(R, torch.Tensor) else R
        self.T = T
        self.time_grid = time_grid.numpy() if isinstance(time_grid, torch.Tensor) else time_grid
        
        # Store torch versions for later use
        self.H_torch = H if isinstance(H, torch.Tensor) else torch.tensor(H, dtype=torch.float32)
        self.M_torch = M if isinstance(M, torch.Tensor) else torch.tensor(M, dtype=torch.float32)
        self.D_torch = D if isinstance(D, torch.Tensor) else torch.tensor(D, dtype=torch.float32)
        
        # Solve Riccati equation
        self.S = self.solve_riccati()
        
        # Precompute integral term for efficiency
        self.integral_terms = self._precompute_integral_terms()
    
    def riccati_ode(self, t, S_flat):
        """Computes dS/dt for the Riccati ODE."""
        S = S_flat.reshape(2, 2)
        D_inv = np.linalg.inv(self.D)
        dSdt = (S @ self.M @ D_inv @ self.M.T @ S - 
                self.H.T @ S - S @ self.H - self.C)
        return dSdt.flatten()
    
    def solve_riccati(self):
        """Solves the Riccati equation backwards in time."""
        sol = solve_ivp(
            self.riccati_ode,
            [self.T, 0],  # Integrate backward from T to 0
            self.R.flatten(),
            t_eval=self.time_grid[::-1],  # Reverse time grid for backward integration
            method='RK45',
            rtol=1e-8,
            atol=1e-8
        )
        
        # Create dictionary mapping time points to S matrices
        S_dict = {}
        for t, S_flat in zip(sol.t[::-1], sol.y.T[::-1]):
            S_dict[t] = torch.tensor(S_flat.reshape(2, 2), dtype=torch.float32)
        
        return S_dict
    
    def _precompute_integral_terms(self):
        """Precomputes integral terms for value function for efficiency."""
        dt = self.time_grid[1] - self.time_grid[0]
        integral_terms = {}
        
        # Convert sigma to torch tensor if needed
        sigma_torch = self.sigma if isinstance(self.sigma, torch.Tensor) else torch.tensor(self.sigma, dtype=torch.float32)
        sigma_sigma_T = sigma_torch @ sigma_torch.T
        
        # Compute integral terms for each time point
        for t_idx, t in enumerate(self.time_grid):
            integral_term = 0.0
            for tau_idx in range(t_idx, len(self.time_grid)):
                tau = self.time_grid[tau_idx]
                S_tau = self.S[tau]
                integral_term += torch.trace(sigma_sigma_T @ S_tau).item() * dt
            
            integral_terms[t] = integral_term
        
        return integral_terms
    
    def get_nearest_time(self, t):
        """Gets the nearest time point in the time grid."""
        return min(self.time_grid, key=lambda tn: abs(tn - t))
    
    def value_function(self, t, x):
        """Computes the value function v(t, x) = x^T S(t) x + integral term."""
        t_val = t.item() if isinstance(t, torch.Tensor) else t
        nearest_t = self.get_nearest_time(t_val)
        
        # Get S matrix for nearest time point
        S_t = self.S[nearest_t]
        
        # Use precomputed integral term
        integral_term = self.integral_terms[nearest_t]
        
        # Reshape x if needed
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        
        return x.T @ S_t @ x + integral_term
    
    def optimal_control(self, t, x):
        """Computes the optimal control a(t, x) = -D^(-1) M^T S(t) x."""
        t_val = t.item() if isinstance(t, torch.Tensor) else t
        nearest_t = self.get_nearest_time(t_val)
        
        # Get S matrix for nearest time point
        S_t = self.S[nearest_t]
        
        # Compute D^(-1) M^T S(t) x
        D_inv = torch.linalg.inv(self.D_torch)
        
        # Reshape x if needed
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        
        return -D_inv @ self.M_torch.T @ S_t @ x
    
    def explicit_scheme(self, x0, N):
        """Simulates a trajectory using the explicit scheme."""
        dt = self.T / N
        x = x0.clone()
        
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        
        for _ in range(N):
            a = self.optimal_control(0, x)
            x = x + (self.H_torch @ x + self.M_torch @ a) * dt + self.sigma @ torch.randn(2, 1) * np.sqrt(dt)
        
        return x
    
    def implicit_scheme(self, x0, N):
        """Simulates a trajectory using the implicit scheme."""
        dt = self.T / N
        x = x0.clone()
        
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        
        I = torch.eye(2)
        
        for _ in range(N):
            a = self.optimal_control(0, x)
            b = x + dt * self.M_torch @ a + self.sigma @ torch.randn(2, 1) * np.sqrt(dt)
            x = torch.linalg.solve(I - dt * self.H_torch, b)
        
        return x
    
    def monte_carlo_time_steps(self, x0, schemes, N_values, num_samples, seed=None):
        """Analyzes convergence with respect to time steps."""
        if seed is not None:
            torch.manual_seed(seed)
        
        # Compute true value
        true_value = self.value_function(0, x0)
        errors = {scheme: [] for scheme in schemes}
        
        for N in N_values:
            for scheme in schemes:
                total_error = 0.0
                for _ in range(num_samples):
                    if scheme == 'explicit':
                        x_final = self.explicit_scheme(x0, N)
                    else:  # implicit
                        x_final = self.implicit_scheme(x0, N)
                        
                    # Compute final value and error
                    final_value = x_final.T @ self.S[self.T] @ x_final
                    total_error += torch.abs(true_value - final_value).item()
                
                errors[scheme].append(total_error / num_samples)
        
        return errors, true_value
    
    def monte_carlo_samples(self, x0, schemes, N, sample_values, seed=None):
        """Analyzes convergence with respect to number of samples."""
        if seed is not None:
            torch.manual_seed(seed)
        
        true_value = self.value_function(0, x0)
        errors = {scheme: [] for scheme in schemes}
        
        # Pre-generate full simulation data for each scheme
        sim_results = {scheme: [] for scheme in schemes}
        
        # Maximum number of samples needed
        max_samples = max(sample_values)
        
        # Generate simulation results for all needed samples
        for scheme in schemes:
            for _ in range(max_samples):
                if scheme == 'explicit':
                    x_final = self.explicit_scheme(x0, N)
                else:  # implicit
                    x_final = self.implicit_scheme(x0, N)
                    
                # Compute final value
                final_value = x_final.T @ self.S[self.T] @ x_final
                error = torch.abs(true_value - final_value).item()
                sim_results[scheme].append(error)
        
        # Calculate errors for different sample sizes
        for scheme in schemes:
            for num_samples in sample_values:
                errors[scheme].append(sum(sim_results[scheme][:num_samples]) / num_samples)
        
        return errors, true_value

def test_lqr_monte_carlo():
    """Test case for Exercise 1.2"""
    # Define problem parameters
    H = torch.tensor([[0.5, 0.5], [0.0, 0.5]])
    M = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    sigma = torch.eye(2) * 0.5
    C = torch.tensor([[1.0, 0.1], [0.1, 1.0]])
    D = torch.tensor([[1.0, 0.1], [0.1, 1.0]]) * 0.1
    R = torch.tensor([[1.0, 0.3], [0.3, 1.0]]) * 10.0
    T = 0.5
    time_grid = torch.linspace(0, T, 100)
    
    # Create LQR solver
    lqr = LQR(H, M, sigma, C, D, R, T, time_grid)
    
    # Test states
    x0 = torch.tensor([[1.0], [1.0]])
    x1 = torch.tensor([[2.0], [2.0]])
    
    # Test time step convergence
    schemes = ['explicit', 'implicit']
    N_values = [2**i for i in range(1, 12)]
    num_samples = 10000
    
    errors, true_value = lqr.monte_carlo_time_steps(x0, schemes, N_values, num_samples, seed=42)
    
    # Plot time step convergence
    plt.figure(figsize=(10, 6))
    for scheme in schemes:
        plt.loglog(N_values, errors[scheme], '-o', label=f'{scheme.capitalize()} Scheme')
    
    # Add reference slopes
    x_ref = np.array([N_values[0], N_values[-1]])
    y_ref_half = true_value.item() * x_ref[0]**0.5 / x_ref**0.5
    y_ref_one = true_value.item() * x_ref[0] / x_ref
    
    plt.loglog(x_ref, y_ref_half, '--', label='O(N^(-1/2))')
    plt.loglog(x_ref, y_ref_one, '-.', label='O(N^(-1))')
    
    plt.xlabel('Number of Time Steps (N)')
    plt.ylabel('Error')
    plt.title('Convergence of Schemes with Respect to Time Steps')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()
    
    # Test sample size convergence
    N = 10000  # Fixed large number of time steps
    sample_values = [2 * 4**i for i in range(6)]
    
    errors, true_value = lqr.monte_carlo_samples(x0, schemes, N, sample_values, seed=42)
    
    # Plot sample size convergence
    plt.figure(figsize=(10, 6))
    for scheme in schemes:
        plt.loglog(sample_values, errors[scheme], '-o', label=f'{scheme.capitalize()} Scheme')
    
    # Add reference slope
    x_ref = np.array([sample_values[0], sample_values[-1]])
    y_ref = true_value.item() * np.sqrt(x_ref[0]) / np.sqrt(x_ref)
    
    plt.loglog(x_ref, y_ref, '--', label='O(M^(-1/2))')
    
    plt.xlabel('Number of Monte Carlo Samples (M)')
    plt.ylabel('Error')
    plt.title('Convergence of Schemes with Respect to Sample Size')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()

test_lqr_monte_carlo()