import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import time
from .q2 import SoftLQR
import math

class ValueNN(nn.Module):
    """Neural network to approximate the value function for soft LQR.
    The value function is parameterized as v(t,x) = x^T K(t)x + R(t)."""
    
    def __init__(self, hidden_size=512, device=torch.device("cpu")):
        super(ValueNN, self).__init__()
        self.device = device
        
        # Explicitly set dtype for all layers
        dtype = torch.float64
        
        self.time_embedding = nn.Linear(1, hidden_size, device=device, dtype=dtype)
        self.hidden_layer1 = nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype)
        self.matrix_output = nn.Linear(hidden_size, 2*2, device=device, dtype=dtype)
        self.offset_output = nn.Linear(hidden_size, 1, device=device, dtype=dtype)
    
    def forward(self, t):
        """
        Forward pass through the network.
        
        Args:
            t: Time tensor (batch)
            
        Returns:
            Tuple of (matrix, offset) for value function approximation
        """
        # Ensure t is properly shaped
        t = t.view(-1, 1)
        
        # Forward pass through time embedding and hidden layers
        x = torch.relu(self.time_embedding(t))
        x = torch.relu(self.hidden_layer1(x))
        x = torch.relu(self.hidden_layer2(x))
        
        # Get matrix elements and reshape to batch of 2x2 matrices
        matrix_elements = self.matrix_output(x)
        batch_matrices = matrix_elements.view(-1, 2, 2)
        
        # Make the matrices positive semi-definite (symmetric and positive)
        matrices = torch.matmul(batch_matrices, batch_matrices.transpose(1, 2))
        matrices = matrices + 1e-3 * torch.eye(2, device=self.device).unsqueeze(0)
        
        # Get offset term
        offset = self.offset_output(x)
        
        return matrices, offset
    
    def value_function(self, t, x):
        """
        Compute the value function v(t,x) = x^T K(t)x + R(t) using vectorized operations.
        
        Args:
            t: Time tensor (batch) 
            x: State tensor (batch x 2)
            
        Returns:
            Value function at (t, x) (batch)
        """
        # Ensure consistent dtypes
        t = t.to(torch.float64)
        x = x.to(torch.float64)
        
        matrices, offsets = self.forward(t)
        
        # Ensure matrices and offsets are float64
        matrices = matrices.to(torch.float64)
        offsets = offsets.to(torch.float64)
        
        # Vectorized computation of quadratic term x^T K(t)x
        # Reshape x to [batch, 2, 1] for batch matrix multiplication
        x_reshaped = x.unsqueeze(2)  # [batch, 2, 1]
        
        # Compute x^T K(t)x for all batch elements at once:
        # First multiply matrices by x: [batch, 2, 2] @ [batch, 2, 1] -> [batch, 2, 1]
        # Then multiply x^T by the result: [batch, 1, 2] @ [batch, 2, 1] -> [batch, 1, 1]
        quadratic_terms = torch.bmm(
            torch.bmm(x_reshaped.transpose(1, 2), matrices), 
            x_reshaped
        ).squeeze(2).squeeze(1)  # Remove extra dimensions to get [batch]
        
        # Add offset term
        values = quadratic_terms + offsets.squeeze()
        
        return values


class CriticAlgorithm:
    """Implementation of the critic algorithm for soft LQR."""
    
    def __init__(self, 
                 soft_lqr: SoftLQR, 
                 hidden_size: int = 512, 
                 learning_rate: float = 1e-3,
                 device: torch.device = torch.device("cpu")):
        """
        Initialize the critic algorithm.
        """
        self.soft_lqr = soft_lqr
        self.device = device
        
        # Create value network - explicitly set dtype to float64
        self.value_network = ValueNN(hidden_size=hidden_size, device=device).to(device)
        
        # Convert network parameters to double precision
        for param in self.value_network.parameters():
            param.data = param.data.to(torch.float64)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        
        # MSE loss function
        self.criterion = nn.MSELoss()
        
        # For tracking progress
        self.loss_history = []
    
    def _generate_batch(self, batch_size: int, state_range: float = 3.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of random states and times for training."""
        # Sample times uniformly from [0, T] - explicitly specify dtype
        times = torch.rand(batch_size, device=self.device, dtype=torch.float64) * self.soft_lqr.T
        
        # Sample states uniformly from [-state_range, state_range]^2
        states = (torch.rand(batch_size, 2, device=self.device, dtype=torch.float64) * 2 - 1) * state_range
        
        return times, states
    
    def _compute_target_values(self, t: torch.Tensor, x: torch.Tensor, num_steps: int = 100) -> torch.Tensor:
        """
        Compute target values for critic training by simulating trajectories using the current policy.
        Vectorized implementation to process all batch elements simultaneously.
        
        Args:
            t: Time tensor (batch)
            x: State tensor (batch x d)
            num_steps: Number of time steps for simulation
            
        Returns:
            Target values tensor (batch)
        """
        batch_size = t.shape[0]
        device = self.device
        dtype = torch.float64
        
        # Initialize target values tensor
        target_values = torch.zeros(batch_size, device=device, dtype=dtype)
        
        # Handle states at terminal time directly (vectorized)
        terminal_mask = (t >= self.soft_lqr.T - 1e-6)
        if terminal_mask.any():
            x_terminal = x[terminal_mask]
            
            # Vectorized computation of terminal cost x^T R x
            x_term_reshaped = x_terminal.unsqueeze(2)  # [batch, 2, 1]
            R_tensor = self.soft_lqr.R.to(dtype).unsqueeze(0).expand(x_terminal.size(0), -1, -1)
            
            terminal_costs = torch.bmm(
                torch.bmm(x_term_reshaped.transpose(1, 2), R_tensor),
                x_term_reshaped
            ).squeeze()
            
            target_values[terminal_mask] = terminal_costs
        
        # Process non-terminal states
        non_terminal_mask = ~terminal_mask
        if not non_terminal_mask.any():
            return target_values
            
        # Extract non-terminal states for batch processing
        t_batch = t[non_terminal_mask]
        x_batch = x[non_terminal_mask]
        non_term_batch_size = t_batch.size(0)
        
        # Compute remaining time and step size
        remaining_time = self.soft_lqr.T - t_batch
        dt = remaining_time / num_steps
        
        # Initialize current states and costs
        states = x_batch.clone()
        costs = torch.zeros(non_term_batch_size, device=device, dtype=dtype)
        
        # Pre-allocate tensors for matrix operations to avoid repeated creation
        C_tensor = self.soft_lqr.C.to(dtype).unsqueeze(0).expand(non_term_batch_size, -1, -1)
        D_tensor = self.soft_lqr.D.to(dtype).unsqueeze(0).expand(non_term_batch_size, -1, -1)
        R_tensor = self.soft_lqr.R.to(dtype).unsqueeze(0).expand(non_term_batch_size, -1, -1)
        H_tensor = self.soft_lqr.H.to(dtype).unsqueeze(0).expand(non_term_batch_size, -1, -1)
        M_tensor = self.soft_lqr.M.to(dtype).unsqueeze(0).expand(non_term_batch_size, -1, -1)
        
        
        # Simulate all trajectories together
        for step in range(num_steps):
            # Current time for all batch elements
            current_time = t_batch + step * dt
            current_time_tensor = current_time.unsqueeze(1)  # Ensure proper shape
            
            # Get control distribution parameters
            means, covariance = self.soft_lqr.optimal_control_distribution(current_time_tensor, states)
            means = means.to(dtype)
            covariance = covariance.to(dtype)
            
            # Vectorized sampling from multivariate normal distribution
            L = torch.linalg.cholesky(covariance)  # Calculate Cholesky once
            
            # Generate random samples for all batch elements
            z = torch.randn(non_term_batch_size, self.soft_lqr.m, device=device, dtype=dtype)
            
            # Apply transformation to all samples: mean + L @ z
            controls = means + torch.bmm(
                L.unsqueeze(0).expand(non_term_batch_size, -1, -1), 
                z.unsqueeze(2)
            ).squeeze(2)
            
            # Vectorized computation of state costs using batch matrix multiplication
            states_reshaped = states.unsqueeze(2)  # [batch, 2, 1]
            state_costs = torch.bmm(
                torch.bmm(states_reshaped.transpose(1, 2), C_tensor),
                states_reshaped
            ).squeeze()
            
            # Vectorized computation of control costs
            controls_reshaped = controls.unsqueeze(2)  # [batch, 2, 1]
            control_costs = torch.bmm(
                torch.bmm(controls_reshaped.transpose(1, 2), D_tensor),
                controls_reshaped
            ).squeeze()
            
            # Entropy term calculation (same for all states if covariance is constant)
            entropy_term = self.soft_lqr.tau * torch.log(torch.det(2 * math.pi * math.e * covariance)) / 2
            entropy_terms = torch.ones(non_term_batch_size, device=device, dtype=dtype) * entropy_term
            
            # Accumulate costs with proper broadcasting
            dt_expanded = dt.unsqueeze(-1) if dt.dim() == 1 else dt
            costs += (state_costs + control_costs + entropy_terms) * dt
            
            # Vectorized state update using Euler step
            # Calculate drift: HX + Mα
            drift = torch.bmm(states.unsqueeze(1), H_tensor.transpose(1, 2)).squeeze(1) + \
                torch.bmm(controls.unsqueeze(1), M_tensor.transpose(1, 2)).squeeze(1)
            
            # Generate random noise for all states at once
            noise = torch.randn(non_term_batch_size, self.soft_lqr.sigma.shape[1], 
                            device=device, dtype=dtype)
            diffusion = torch.matmul(noise, self.soft_lqr.sigma.T) * torch.sqrt(dt.unsqueeze(-1))
            
            # Update all states in parallel
            states = states + drift * dt.unsqueeze(-1) + diffusion
        
        # Vectorized calculation of terminal costs
        states_reshaped = states.unsqueeze(2)  # [batch, 2, 1]
        terminal_costs = torch.bmm(
            torch.bmm(states_reshaped.transpose(1, 2), R_tensor),
            states_reshaped
        ).squeeze()
        costs += terminal_costs
        
        # Assign costs to target values
        target_values[non_terminal_mask] = costs
        
        return target_values

    def train_step(self, batch_size: int) -> float:
        """Perform one training step of the critic algorithm."""
        # Generate batch
        times, states = self._generate_batch(batch_size)
        
        # Compute target values
        with torch.no_grad():
            target_values = self._compute_target_values(times, states)
        
        # Compute predicted values
        predicted_values = self.value_network.value_function(times, states)
        
        # Ensure consistent dtype before computing loss
        predicted_values = predicted_values.to(torch.float64)
        target_values = target_values.to(torch.float64)
        
        # Compute loss
        loss = self.criterion(predicted_values, target_values)
        
        # Perform optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_epochs: int, batch_size: int, 
              eval_interval: int = 10, eval_grid_size: int = 5) -> None:
        """
        Train the critic network.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            eval_interval: Interval for evaluation
            eval_grid_size: Grid size for evaluation
        """
        print("Starting critic training...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            loss = self.train_step(batch_size)
            self.loss_history.append(loss)
            
            if (epoch + 1) % eval_interval == 0 or epoch == 0:
                # Evaluate current performance
                max_error = self.evaluate(eval_grid_size)
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.6e}, Max Error: {max_error:.6e}, Time: {elapsed:.2f}s")
        
        print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    
    def evaluate(self, grid_size: int = 5, state_range: float = 3.0, 
             time_points: Optional[List[float]] = None) -> float:
        """
        Evaluate the critic network against the exact value function.
        
        Args:
            grid_size: Number of points in each dimension for evaluation
            state_range: Range for evaluation grid
            time_points: Specific time points for evaluation, defaults to [0, T/3, 2T/3, T]
            
        Returns:
            Maximum absolute error
        """
        # Set default time points if not provided
        if time_points is None:
            time_points = [0, self.soft_lqr.T/3, 2*self.soft_lqr.T/3, self.soft_lqr.T]
        
        # Create evaluation grid - explicitly specify dtype as float64
        x_grid = torch.linspace(-state_range, state_range, grid_size, device=self.device, dtype=torch.float64)
        y_grid = torch.linspace(-state_range, state_range, grid_size, device=self.device, dtype=torch.float64)
        
        # Convert time points to tensor with explicit dtype
        time_tensor = torch.tensor(time_points, device=self.device, dtype=torch.float64)
        
        # Store maximum error
        max_error = 0.0
        
        # Evaluate on grid
        with torch.no_grad():
            for t_idx, t in enumerate(time_tensor):
                t_batch = t.unsqueeze(0)
                for i in range(grid_size):
                    for j in range(grid_size):
                        # Current state with explicit dtype
                        x = torch.tensor([[x_grid[i], y_grid[j]]], device=self.device, dtype=torch.float64)
                        
                        # Compute exact value function
                        exact_value = self.soft_lqr.value_function(t_batch, x)
                        
                        # Compute approximate value function
                        approx_value = self.value_network.value_function(t_batch, x)
                        
                        # Ensure both values have the same dtype before comparison
                        exact_value = exact_value.to(torch.float64)
                        approx_value = approx_value.to(torch.float64)
                        
                        # Update maximum error
                        error = torch.abs(exact_value - approx_value).item()
                        max_error = max(max_error, error)
        
        return max_error
    
    def plot_loss_history(self) -> None:
        """Plot the loss history during training."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history)
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Critic Loss History')
        plt.grid(True)
        plt.draw()
    
    def plot_value_function_comparison(self, grid_size: int = 20, state_range: float = 3.0):
        """
        Plot comparison between exact and learned value function.
        
        Args:
            grid_size: Number of points in each dimension for plotting
            state_range: Range for plotting grid
        """
        # Create evaluation grid with explicit float64 dtype
        x_grid = torch.linspace(-state_range, state_range, grid_size, device=self.device, dtype=torch.float64)
        y_grid = torch.linspace(-state_range, state_range, grid_size, device=self.device, dtype=torch.float64)
        
        # Evaluation time (t=0)
        t = torch.tensor([0.0], device=self.device, dtype=torch.float64)
        
        # Initialize arrays for values
        X, Y = np.meshgrid(x_grid.cpu().numpy(), y_grid.cpu().numpy())
        exact_values = np.zeros((grid_size, grid_size))
        approx_values = np.zeros((grid_size, grid_size))
        errors = np.zeros((grid_size, grid_size))
        
        # Compute values on grid
        with torch.no_grad():
            for i in range(grid_size):
                for j in range(grid_size):
                    # Current state with explicit float64 dtype
                    x = torch.tensor([[x_grid[i].item(), y_grid[j].item()]], 
                                    device=self.device, dtype=torch.float64)
                    
                    # Compute exact value function - must convert to float64 first
                    exact_value = self.soft_lqr.value_function(t, x).item()
                    
                    # Compute approximate value function
                    # Make sure network inputs are float64
                    approx_value = self.value_network.value_function(
                        self._ensure_float64(t), 
                        self._ensure_float64(x)
                    ).item()
                    
                    # Store values
                    exact_values[j, i] = exact_value
                    approx_values[j, i] = approx_value
                    errors[j, i] = np.abs(exact_value - approx_value)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot exact value function
        im1 = axes[0].contourf(X, Y, exact_values, 50, cmap='viridis')
        axes[0].set_title('Exact Value Function at t=0')
        axes[0].set_xlabel('x1')
        axes[0].set_ylabel('x2')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot approximate value function
        im2 = axes[1].contourf(X, Y, approx_values, 50, cmap='viridis')
        axes[1].set_title('Learned Value Function at t=0')
        axes[1].set_xlabel('x1')
        axes[1].set_ylabel('x2')
        plt.colorbar(im2, ax=axes[1])
        
        # Plot error
        im3 = axes[2].contourf(X, Y, errors, 50, cmap='hot')
        axes[2].set_title('Absolute Error')
        axes[2].set_xlabel('x1')
        axes[2].set_ylabel('x2')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.draw()

    def _ensure_float64(self, tensor):
        """Helper method to ensure tensors are consistently float64."""
        if tensor.dtype != torch.float64:
            return tensor.to(torch.float64)
        return tensor


def run_critic_algorithm():
    """Run the critic algorithm for soft LQR."""
    # Set the problem matrices as specified in assignment
    H = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64) * 0.5
    M = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    sigma = torch.eye(2, dtype=torch.float64) * 0.5
    C = torch.tensor([[1.0, 0.1], [0.1, 1.0]], dtype=torch.float64) * 1.0
    
    # Use identity matrix for D as specified in the hints
    D = torch.eye(2, dtype=torch.float64) 
    
    R = torch.tensor([[1.0, 0.3], [0.3, 1.0]], dtype=torch.float64) * 10.0
    
    # Set parameters as specified in the hints
    T = 0.5
    tau = 0.5
    gamma = 1.0
    
    # Create time grid with N=100 as specified
    N = 100
    time_grid = torch.linspace(0, T, N+1, dtype=torch.float64)
    
    # Create soft LQR instance and solve Ricatti ODE
    soft_lqr = SoftLQR(H, M, sigma, C, D, R, T, time_grid, tau, gamma)
    soft_lqr.solve_ricatti()
    
    # Create critic algorithm
    critic = CriticAlgorithm(soft_lqr, hidden_size=512, learning_rate=1e-3)
    
    # Train critic
    critic.train(num_epochs=500, batch_size=64, eval_interval=20)
    
    # Plot results
    critic.plot_loss_history()
    critic.plot_value_function_comparison()
    
    # Evaluate final performance
    max_error = critic.evaluate(grid_size=10)
    print(f"Final maximum error: {max_error:.6e}")
    
    return critic

