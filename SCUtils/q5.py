import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import time
from .q2 import SoftLQR
from .q3 import ValueNN
from .q4 import PolicyNN

class ActorCriticAlgorithm:
    """
    Implementation of the actor-critic algorithm for soft LQR,
    which simultaneously learns the policy and value function.
    """
    
    def __init__(self, 
                 soft_lqr,
                 actor_hidden_size: int = 256,
                 critic_hidden_size: int = 512,
                 actor_lr: float = 1e-4,
                 critic_lr: float = 1e-3,
                 device: torch.device = torch.device("cpu")):
        """
        Initialize the actor-critic algorithm.
        
        Args:
            soft_lqr: SoftLQR instance with problem setup
            actor_hidden_size: Size of hidden layers in actor network
            critic_hidden_size: Size of hidden layers in critic network
            actor_lr: Learning rate for actor optimization
            critic_lr: Learning rate for critic optimization
            device: Device to use for computation
        """
        self.soft_lqr = soft_lqr
        self.device = device
        self.d = soft_lqr.d  # State dimension
        self.m = soft_lqr.m  # Control dimension
        
        # Create actor network (policy)
        self.actor = PolicyNN(d=self.m, hidden_size=actor_hidden_size, device=device)
        self.actor.to(device=device, dtype=torch.float64)
        
        # Create critic network (value function)
        self.critic = ValueNN(hidden_size=critic_hidden_size, device=device)
        self.critic.to(device=device, dtype=torch.float64)
        
        # Create optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # For tracking progress
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.eval_epochs = []
        self.actor_error_history = []
        self.critic_error_history = []
        
    def _ensure_float64(self, tensor):
        """Helper method to ensure tensors are consistently float64."""
        if tensor.dtype != torch.float64:
            return tensor.to(torch.float64)
        return tensor
    
    def _generate_batch(self, batch_size: int, state_range: float = 3.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of random states and times for training.
        
        Args:
            batch_size: Number of samples to generate
            state_range: Range for uniform sampling of states
            
        Returns:
            Tuple of (times, states) tensors
        """
        # Sample times uniformly from [0, T]
        times = torch.rand(batch_size, device=self.device, dtype=torch.float64) * self.soft_lqr.T
        
        # Sample states uniformly from [-state_range, state_range]^d
        states = (torch.rand(batch_size, self.d, device=self.device, dtype=torch.float64) * 2 - 1) * state_range
        
        return times, states
    
    def compute_actor_loss(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the KL divergence loss for the actor.
        
        Args:
            t: Time tensor (batch)
            x: State tensor (batch x d)
            
        Returns:
            Actor loss (scalar tensor)
        """
        # Get action distributions from current policy
        mu_current, Sigma_current = self.actor.action_distribution(t, x)
        
        # Get approximate value function gradient (using critic)
        # We need to compute the advantage using critic's value function
        t_clone = t.clone().detach().requires_grad_(True)
        x_clone = x.clone().detach().requires_grad_(True)
        
        # Get critic's value function for the current state
        v_t_x = self.critic.value_function(t_clone, x_clone)
        
        # Use KL divergence between current policy and optimal policy as a surrogate loss
        # This is similar to what's done in Exercise 4.1
        # Get action distributions from both policies
        mu_optimal, Sigma_optimal = self.soft_lqr.optimal_control_distribution(t, x)
        
        # Ensure optimal_covariance is properly shaped for batch operations
        batch_size = t.shape[0]
        if Sigma_optimal.dim() < 3 and batch_size > 0:
            Sigma_optimal = Sigma_optimal.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Compute KL divergence
        # Add small identity for numerical stability before inverse
        eps_eye = torch.eye(self.m, device=self.device, dtype=torch.float64) * 1e-6
        Sigma_optimal_stable = Sigma_optimal + eps_eye
        
        try:
            optimal_precision = torch.inverse(Sigma_optimal_stable)
        except torch.linalg.LinAlgError:
            optimal_precision = torch.linalg.pinv(Sigma_optimal_stable)
        
        # Compute mean difference
        mean_diff = mu_optimal - mu_current
        
        # Compute KL divergence terms (vectorized)
        # Term 1: tr(Sigma_optimal_inv @ Sigma_current)
        term1 = torch.diagonal(
            torch.bmm(optimal_precision, Sigma_current), 
            dim1=-2, dim2=-1
        ).sum(dim=-1)
        
        # Term 2: (mu_optimal - mu_current)^T @ Sigma_optimal_inv @ (mu_optimal - mu_current)
        term2 = torch.bmm(
            mean_diff.unsqueeze(1), 
            torch.bmm(optimal_precision, mean_diff.unsqueeze(-1))
        ).squeeze()
        
        # Term 3: -d (dimensionality)
        term3 = -torch.tensor(self.m, device=self.device, dtype=torch.float64)
        
        # Term 4: log(det(Sigma_optimal) / det(Sigma_current))
        Sigma_current_stable = Sigma_current + eps_eye
        logdet_optimal = torch.logdet(Sigma_optimal_stable)
        logdet_current = torch.logdet(Sigma_current_stable)
        term4 = logdet_optimal - logdet_current
        
        # Compute KL divergence: 0.5 * (term1 + term2 + term3 + term4)
        kl_div = 0.5 * (term1 + term2 + term3 + term4)
        
        # Return mean KL divergence over the batch
        return kl_div.mean()
    
    def compute_critic_loss(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the critic loss (MSE between predicted value and target value).
        
        Args:
            t: Time tensor (batch)
            x: State tensor (batch x d)
            
        Returns:
            Critic loss (scalar tensor)
        """
        # Generate target values using Monte Carlo simulations
        with torch.no_grad():
            target_values = self._compute_target_values(t, x)
        
        # Compute predicted values
        predicted_values = self.critic.value_function(t, x)
        
        # Compute MSE loss
        mse_loss = nn.MSELoss()
        loss = mse_loss(predicted_values, target_values)
        
        return loss
    
    def _compute_target_values(self, t: torch.Tensor, x: torch.Tensor, num_steps: int = 100) -> torch.Tensor:
        """
        Compute target values for critic training by simulating trajectories.
        
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
        
        # Handle states at terminal time directly
        terminal_mask = (t >= self.soft_lqr.T - 1e-6)
        if terminal_mask.any():
            x_terminal = x[terminal_mask]
            
            # Terminal cost: x^T R x
            x_term_reshaped = x_terminal.unsqueeze(2)
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
        
        # Pre-allocate matrices for computations
        C_tensor = self.soft_lqr.C.to(dtype).unsqueeze(0).expand(non_term_batch_size, -1, -1)
        D_tensor = self.soft_lqr.D.to(dtype).unsqueeze(0).expand(non_term_batch_size, -1, -1)
        R_tensor = self.soft_lqr.R.to(dtype).unsqueeze(0).expand(non_term_batch_size, -1, -1)
        H_tensor = self.soft_lqr.H.to(dtype).unsqueeze(0).expand(non_term_batch_size, -1, -1)
        M_tensor = self.soft_lqr.M.to(dtype).unsqueeze(0).expand(non_term_batch_size, -1, -1)
        
        # Simulate trajectories
        for step in range(num_steps):
            # Current time
            current_time = t_batch + step * dt
            current_time_tensor = current_time.unsqueeze(1)
            
            # Get actions from current policy
            means, covariances = self.actor.action_distribution(current_time_tensor, states)
            
            # Sample actions
            try:
                L = torch.linalg.cholesky(covariances)
                z = torch.randn(non_term_batch_size, self.m, 1, device=device, dtype=dtype)
                controls = means + torch.bmm(L, z).squeeze(2)
            except:
                # Fallback if Cholesky fails
                controls = means + torch.sqrt(torch.diagonal(covariances, dim1=1, dim2=2)) * torch.randn_like(means)
            
            # Calculate costs
            states_reshaped = states.unsqueeze(2)
            controls_reshaped = controls.unsqueeze(2)
            
            # State costs: x^T C x
            state_costs = torch.bmm(
                torch.bmm(states_reshaped.transpose(1, 2), C_tensor),
                states_reshaped
            ).squeeze()
            
            # Control costs: u^T D u
            control_costs = torch.bmm(
                torch.bmm(controls_reshaped.transpose(1, 2), D_tensor),
                controls_reshaped
            ).squeeze()
            
            # Add entropy term
            entropy_term = self.soft_lqr.tau * torch.log(torch.det(2 * np.pi * np.e * covariances)).view(-1) / 2
            
            # Accumulate costs
            costs += (state_costs + control_costs + entropy_term) * dt.view(-1)
            
            # Calculate state updates
            drift = torch.bmm(H_tensor, states.unsqueeze(2)).squeeze(2) + torch.bmm(M_tensor, controls.unsqueeze(2)).squeeze(2)
            
            # Generate noise for diffusion
            noise = torch.randn(non_term_batch_size, self.d, device=device, dtype=dtype) * torch.sqrt(dt.view(-1, 1))
            diffusion = torch.matmul(self.soft_lqr.sigma, noise.unsqueeze(2)).squeeze(2)
            
            # Update states
            states = states + drift * dt.view(-1, 1) + diffusion
        
        # Add terminal costs
        states_reshaped = states.unsqueeze(2)
        terminal_costs = torch.bmm(
            torch.bmm(states_reshaped.transpose(1, 2), R_tensor),
            states_reshaped
        ).squeeze()
        costs += terminal_costs
        
        # Assign costs to target values
        target_values[non_terminal_mask] = costs
        
        return target_values
    
    def train_step(self, batch_size: int) -> Tuple[float, float]:
        """
        Perform one training step of the actor-critic algorithm.
        
        Args:
            batch_size: Batch size for training
            
        Returns:
            Tuple of (Actor loss, Critic loss)
        """
        # Generate batch
        times, states = self._generate_batch(batch_size)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss = self.compute_critic_loss(times, states)
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss = self.compute_actor_loss(times, states)
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()
    
    def train(self, num_epochs: int, batch_size: int, eval_interval: int = 10) -> None:
        """
        Train the actor-critic algorithm.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            eval_interval: Interval for evaluation
        """
        print(f"Starting actor-critic training for {num_epochs} epochs...")
        start_time = time.time()
        
        # Initial evaluation
        actor_error, critic_error = self.evaluate()
        self.eval_epochs.append(0)
        self.actor_error_history.append(actor_error)
        self.critic_error_history.append(critic_error)
        print(f"Epoch 0/{num_epochs}, Initial Actor Error: {actor_error:.6e}, Initial Critic Error: {critic_error:.6e}")
        
        for epoch in range(num_epochs):
            actor_loss, critic_loss = self.train_step(batch_size)
            self.actor_loss_history.append(actor_loss)
            self.critic_loss_history.append(critic_loss)
            
            if (epoch + 1) % eval_interval == 0 or epoch == num_epochs - 1:
                actor_error, critic_error = self.evaluate()
                self.eval_epochs.append(epoch + 1)
                self.actor_error_history.append(actor_error)
                self.critic_error_history.append(critic_error)
                
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{num_epochs}, Actor Loss: {actor_loss:.6e}, Critic Loss: {critic_loss:.6e}, "
                      f"Actor Error: {actor_error:.6e}, Critic Error: {critic_error:.6e}, Time: {elapsed:.2f}s")
        
        print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    
    def evaluate(self, grid_size: int = 5, state_range: float = 3.0, 
                time_points: Optional[List[float]] = None) -> Tuple[float, float]:
        """
        Evaluate the actor-critic algorithm against the optimal solution.
        
        Args:
            grid_size: Number of points in each dimension for evaluation
            state_range: Range for evaluation grid
            time_points: Specific time points for evaluation, defaults to [0, T/3, 2T/3, T]
            
        Returns:
            Tuple of (Actor error, Critic error)
        """
        # Set default time points if not provided
        if time_points is None:
            time_points = [0, self.soft_lqr.T/3, 2*self.soft_lqr.T/3, self.soft_lqr.T]
            
        # Create evaluation grid
        x_grid = torch.linspace(-state_range, state_range, grid_size, device=self.device, dtype=torch.float64)
        y_grid = torch.linspace(-state_range, state_range, grid_size, device=self.device, dtype=torch.float64)
        
        # Convert time points to tensor
        time_tensor = torch.tensor(time_points, device=self.device, dtype=torch.float64)
        
        # Storage for errors
        actor_errors = []
        critic_errors = []
        
        with torch.no_grad():
            for t in time_tensor:
                t_batch = t.repeat(grid_size * grid_size)
                
                states = []
                for i in range(grid_size):
                    for j in range(grid_size):
                        states.append([x_grid[i], y_grid[j]])
                        
                states_tensor = torch.tensor(states, device=self.device, dtype=torch.float64)
                
                # Evaluate actor (policy) error
                means_learned, cov_learned = self.actor.action_distribution(t_batch, states_tensor)
                means_optimal, cov_optimal = self.soft_lqr.optimal_control_distribution(t_batch, states_tensor)
                
                # Compute errors for actor
                mean_error = torch.mean(torch.norm(means_learned - means_optimal, dim=1)).item()
                
                if cov_optimal.dim() < 3:
                    cov_optimal = cov_optimal.unsqueeze(0).expand(grid_size * grid_size, -1, -1)
                    
                cov_error = torch.mean(torch.norm(cov_learned - cov_optimal, dim=(1, 2))).item()
                actor_error = (mean_error + cov_error) / 2
                actor_errors.append(actor_error)
                
                # Evaluate critic (value function) error
                v_learned = self.critic.value_function(t_batch, states_tensor)
                v_optimal = self.soft_lqr.value_function(t_batch, states_tensor)
                
                critic_error = torch.mean(torch.abs(v_learned - v_optimal)).item()
                critic_errors.append(critic_error)
        
        return np.mean(actor_errors), np.mean(critic_errors)
    
    def compare_to_optimal(self, x0_list: list, num_steps: int = 200, scheme: str = 'explicit'):
        """
        Compare trajectories and costs from learned policy vs optimal policy.
        
        Args:
            x0_list: List of initial states
            num_steps: Number of time steps for simulation
            scheme: 'explicit' or 'implicit' integration scheme
        """
        dt = self.soft_lqr.T / num_steps
        t_grid = torch.linspace(0, self.soft_lqr.T, num_steps + 1, dtype=torch.float64, device=self.device)
        
        # Initialize plot
        fig, axes = plt.subplots(len(x0_list), 2, figsize=(16, 4*len(x0_list)), squeeze=False)
        
        for i, x0 in enumerate(x0_list):
            # Convert initial state to tensor
            x0_tensor = torch.tensor([x0], dtype=torch.float64, device=self.device)
            
            # Initialize trajectories and costs
            X_learned = torch.zeros((num_steps + 1, 2), dtype=torch.float64, device=self.device)
            X_optimal = torch.zeros((num_steps + 1, 2), dtype=torch.float64, device=self.device)
            X_learned[0] = x0_tensor[0]
            X_optimal[0] = x0_tensor[0]
            
            costs_learned = torch.zeros(num_steps + 1, dtype=torch.float64, device=self.device)
            costs_optimal = torch.zeros(num_steps + 1, dtype=torch.float64, device=self.device)
            
            # Generate same Brownian increments for both simulations
            dW = torch.randn((num_steps, self.soft_lqr.sigma.shape[1]), 
                            dtype=torch.float64, device=self.device) * np.sqrt(dt)
            
            # Pre-calculate necessary matrices
            H = self.soft_lqr.H.to(torch.float64)
            M = self.soft_lqr.M.to(torch.float64)
            sigma = self.soft_lqr.sigma.to(torch.float64)
            C = self.soft_lqr.C.to(torch.float64)
            D = self.soft_lqr.D.to(torch.float64)
            R = self.soft_lqr.R.to(torch.float64)
            
            # Simulate trajectories
            if scheme == 'explicit':
                for n in range(num_steps):
                    t_n = t_grid[n]
                    t_tensor = torch.tensor([t_n], dtype=torch.float64, device=self.device)
                    
                    # Learned policy step
                    control_learned = self.actor.sample_action(t_tensor, X_learned[n:n+1])[0]
                    drift_learned = H @ X_learned[n] + M @ control_learned
                    X_learned[n+1] = X_learned[n] + drift_learned * dt + sigma @ dW[n]
                    
                    # Compute running cost for learned policy
                    learned_state_cost = X_learned[n] @ C @ X_learned[n]
                    learned_control_cost = control_learned @ D @ control_learned
                    
                    # Add entropy regularization term for learned policy
                    _, cov_learned = self.actor.action_distribution(t_tensor, X_learned[n:n+1])
                    entropy_term = self.soft_lqr.tau * torch.log(torch.det(2 * np.pi * np.e * cov_learned[0])) / 2
                    
                    costs_learned[n+1] = costs_learned[n] + (learned_state_cost + learned_control_cost + entropy_term) * dt
                    
                    # Optimal policy step
                    control_optimal = self.soft_lqr.optimal_control(t_tensor, X_optimal[n:n+1])[0]
                    drift_optimal = H @ X_optimal[n] + M @ control_optimal
                    X_optimal[n+1] = X_optimal[n] + drift_optimal * dt + sigma @ dW[n]
                    
                    # Compute running cost for optimal policy
                    optimal_state_cost = X_optimal[n] @ C @ X_optimal[n]
                    optimal_control_cost = control_optimal @ D @ control_optimal
                    
                    # Add entropy regularization term for optimal policy
                    _, cov_optimal = self.soft_lqr.optimal_control_distribution(t_tensor, X_optimal[n:n+1])
                    entropy_term_opt = self.soft_lqr.tau * torch.log(torch.det(2 * np.pi * np.e * cov_optimal)) / 2
                    
                    costs_optimal[n+1] = costs_optimal[n] + (optimal_state_cost + optimal_control_cost + entropy_term_opt) * dt
            else:
                # Implicit scheme (would be implemented similarly)
                pass
            
            # Add terminal costs
            learned_terminal = X_learned[-1] @ R @ X_learned[-1]
            optimal_terminal = X_optimal[-1] @ R @ X_optimal[-1]
            costs_learned[-1] += learned_terminal
            costs_optimal[-1] += optimal_terminal
            
            # Plot trajectories
            ax1 = axes[i, 0]
            ax1.plot(X_learned.cpu().detach().numpy()[:, 0], X_learned.cpu().detach().numpy()[:, 1], 'b-', label='Learned Policy')
            ax1.plot(X_optimal.cpu().detach().numpy()[:, 0], X_optimal.cpu().detach().numpy()[:, 1], 'r-', label='Optimal Policy')
            ax1.scatter([x0[0]], [x0[1]], color='g', s=100, marker='o', label='Initial State')
            ax1.set_title(f'Trajectories from Initial State {x0}')
            ax1.set_xlabel('X1')
            ax1.set_ylabel('X2')
            ax1.grid(True)
            ax1.legend()
            
            # Plot costs over time
            ax2 = axes[i, 1]
            ax2.plot(t_grid.cpu().numpy(), costs_learned.cpu().detach().numpy(), 'b-', label='Learned Policy Cost')
            ax2.plot(t_grid.cpu().numpy(), costs_optimal.cpu().detach().numpy(), 'r-', label='Optimal Policy Cost')
            ax2.set_title(f'Cost Over Time from Initial State {x0}')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Cost')
            ax2.grid(True)
            ax2.legend()
            
            print(f"Initial state {x0}:")
            print(f"  Learned policy final cost: {costs_learned[-1].item():.4f}")
            print(f"  Optimal policy final cost: {costs_optimal[-1].item():.4f}")
            print(f"  Cost difference: {(costs_learned[-1] - costs_optimal[-1]).item():.4f}")
        
        plt.tight_layout()
        plt.draw()
    
    def plot_loss_history(self):
        """Plot the loss history during training."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot actor loss
        axes[0].plot(self.actor_loss_history)
        axes[0].set_yscale('log')
        axes[0].set_ylabel('Actor Loss (KL Divergence)')
        axes[0].set_title('Actor Loss History')
        axes[0].grid(True)
        
        # Plot critic loss
        axes[1].plot(self.critic_loss_history)
        axes[1].set_yscale('log')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Critic Loss (MSE)')
        axes[1].set_title('Critic Loss History')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.draw()
    
    def plot_error_history(self):
        """Plot the error history during training."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.eval_epochs, self.actor_error_history, 'b-o', label='Actor Error')
        plt.plot(self.eval_epochs, self.critic_error_history, 'r-s', label='Critic Error')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Actor and Critic Error History')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.draw()

    def plot_value_function_comparison(self, grid_size: int = 20, state_range: float = 3.0, time_point: float = 0.0):
        """
        Plot comparison between exact and learned value function as contour plots.
        
        Args:
            grid_size: Number of points in each dimension for plotting
            state_range: Range for plotting grid
            time_point: Time point at which to evaluate the value function (default: t=0)
        """
        # Create evaluation grid with explicit float64 dtype
        x_grid = torch.linspace(-state_range, state_range, grid_size, device=self.device, dtype=torch.float64)
        y_grid = torch.linspace(-state_range, state_range, grid_size, device=self.device, dtype=torch.float64)
        
        # Create meshgrid for evaluation
        X, Y = np.meshgrid(x_grid.cpu().detach().numpy(), y_grid.cpu().detach().numpy())
        
        # Evaluation time (t=time_point)
        t = torch.tensor([time_point], device=self.device, dtype=torch.float64)
        
        # Initialize arrays for values
        exact_values = np.zeros((grid_size, grid_size))
        learned_values = np.zeros((grid_size, grid_size))
        errors = np.zeros((grid_size, grid_size))
        
        # Compute values on grid
        with torch.no_grad():
            for i in range(grid_size):
                for j in range(grid_size):
                    # Current state with explicit float64 dtype
                    x = torch.tensor([[x_grid[i].item(), y_grid[j].item()]], 
                                    device=self.device, dtype=torch.float64)
                    
                    # Compute exact value function from soft_lqr
                    exact_value = self.soft_lqr.value_function(t, x).item()
                    
                    # Compute approximate value function from critic
                    learned_value = self.critic.value_function(t, x).item()
                    
                    # Store values
                    exact_values[j, i] = exact_value
                    learned_values[j, i] = learned_value
                    errors[j, i] = np.abs(exact_value - learned_value)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), squeeze=True)
        
        # Plot exact value function
        contour1 = axes[0].contourf(X, Y, exact_values, 50, cmap='viridis')
        axes[0].set_title(f'Exact Value Function at t={time_point}')
        axes[0].set_xlabel('x₁')
        axes[0].set_ylabel('x₂')
        plt.colorbar(contour1, ax=axes[0])
        
        # Plot approximate value function
        contour2 = axes[1].contourf(X, Y, learned_values, 50, cmap='viridis')
        axes[1].set_title(f'Learned Value Function at t={time_point}')
        axes[1].set_xlabel('x₁')
        axes[1].set_ylabel('x₂')
        plt.colorbar(contour2, ax=axes[1])
        
        # Plot absolute error
        contour3 = axes[2].contourf(X, Y, errors, 50, cmap='hot')
        axes[2].set_title('Absolute Error')
        axes[2].set_xlabel('x₁')
        axes[2].set_ylabel('x₂')
        plt.colorbar(contour3, ax=axes[2])
        
        # Add some statistics
        max_error = np.max(errors)
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        fig.suptitle(f'Value Function Comparison (Max Error: {max_error:.4f}, Mean: {mean_error:.4f}, Median: {median_error:.4f})', fontsize=16)
        
        plt.tight_layout()
        plt.draw()

def run_actor_critic_algorithm(initial_states=None):
    """Run the actor-critic algorithm for soft LQR."""
    # Set the problem matrices
    H = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64) * 0.5
    M = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    sigma = torch.eye(2, dtype=torch.float64) * 0.5
    C = torch.tensor([[1.0, 0.1], [0.1, 1.0]], dtype=torch.float64) * 1.0
    D = torch.eye(2, dtype=torch.float64)  # Use identity as recommended in the hints
    R = torch.tensor([[1.0, 0.3], [0.3, 1.0]], dtype=torch.float64) * 10.0
    
    # Set parameters as specified in the hints
    T = 0.5
    tau = 0.5
    gamma = 1.0
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create time grid with N=100 as specified
    N = 1000
    time_grid = torch.linspace(0, T, N+1, dtype=torch.float64)
    
    # Create soft LQR instance and solve Ricatti ODE
    soft_lqr = SoftLQR(H, M, sigma, C, D, R, T, time_grid, tau, gamma)
    soft_lqr.solve_ricatti()
    
    # Create actor-critic algorithm
    actor_critic = ActorCriticAlgorithm(
        soft_lqr=soft_lqr,
        actor_hidden_size=256,
        critic_hidden_size=512,
        actor_lr=1e-4,
        critic_lr=1e-3,
        device=device
    )
    
    # Train actor-critic
    actor_critic.train(num_epochs=500, batch_size=256, eval_interval=25)
    
    # Plot results
    actor_critic.plot_loss_history()
    actor_critic.plot_error_history()

    # Plot value function comparison
    actor_critic.plot_value_function_comparison(grid_size=30, state_range=3.0, time_point=0.0)
    # You can also visualize at different time points if desired
    actor_critic.plot_value_function_comparison(grid_size=30, state_range=3.0, time_point=T/2)
    
    # Compare with optimal solution on test states
    if initial_states is None:
        initial_states = [
            [2.0, 2.0],
            [2.0, -2.0],
            [-2.0, -2.0],
            [-2.0, 2.0]
        ]
    actor_critic.compare_to_optimal(initial_states)
    
    return actor_critic

actor_critic = run_actor_critic_algorithm()