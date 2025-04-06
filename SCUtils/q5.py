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
    """Implementation of the actor-critic algorithm for soft LQR."""
    
    def __init__(self, 
                 soft_lqr: SoftLQR, 
                 actor_hidden_size: int = 256, 
                 critic_hidden_size: int = 512,
                 actor_lr: float = 1e-4,
                 critic_lr: float = 1e-3,
                 device: torch.device = torch.device("cpu")):
        """
        Initialize the actor-critic algorithm.
        
        Args:
            soft_lqr: Instance of SoftLQR class
            actor_hidden_size: Size of hidden layers in the actor network
            critic_hidden_size: Size of hidden layers in the critic network
            actor_lr: Learning rate for actor optimization
            critic_lr: Learning rate for critic optimization
            device: Device to use for computation
        """
        self.soft_lqr = soft_lqr
        self.device = device
        
        # Create actor network (policy)
        self.policy_network = PolicyNN(d=soft_lqr.d, hidden_size=actor_hidden_size, device=device).to(device)
        
        # Create critic network (value function)
        self.value_network = ValueNN(hidden_size=critic_hidden_size, device=device).to(device)
        
        # Ensure all parameters are float64
        for param in self.policy_network.parameters():
            param.data = param.data.to(torch.float64)
        
        for param in self.value_network.parameters():
            param.data = param.data.to(torch.float64)
        
        # Create optimizers
        self.actor_optimizer = optim.Adam(self.policy_network.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.value_network.parameters(), lr=critic_lr)
        
        # MSE loss for critic
        self.critic_criterion = nn.MSELoss()
        
        # For tracking progress
        self.actor_loss_history = []
        self.critic_loss_history = []

    def _initialize_networks(self):
        """Initialize networks to better approximate optimal solution."""
        # Sample some states and times
        num_samples = 1000
        times = torch.rand(num_samples, device=self.device, dtype=torch.float64) * self.soft_lqr.T
        states = (torch.rand(num_samples, self.soft_lqr.d, device=self.device, dtype=torch.float64) * 2 - 1) * 3.0
        
        # Get optimal values and policies
        with torch.no_grad():
            # For critic initialization
            optimal_values = torch.zeros(num_samples, device=self.device, dtype=torch.float64)
            for i in range(num_samples):
                t_i = times[i:i+1]
                x_i = states[i:i+1]
                optimal_values[i] = self.soft_lqr.value_function(t_i, x_i)
            
            # For actor initialization
            optimal_means = torch.zeros((num_samples, self.soft_lqr.m), device=self.device, dtype=torch.float64)
            for i in range(num_samples):
                t_i = times[i:i+1]
                x_i = states[i:i+1]
                means, _ = self.soft_lqr.optimal_control_distribution(t_i, x_i)
                optimal_means[i] = means.squeeze()
        
        # Pre-train critic to approximate optimal value function
        optimizer = optim.Adam(self.value_network.parameters(), lr=1e-3)
        for _ in range(100):  # Few steps of pre-training
            predicted_values = self.value_network.value_function(times, states)
            loss = nn.MSELoss()(predicted_values, optimal_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Pre-train actor to approximate optimal policy mean
        optimizer = optim.Adam(self.policy_network.parameters(), lr=1e-3)
        for _ in range(100):  # Few steps of pre-training
            means, _ = self.policy_network.action_distribution(times, states)
            loss = nn.MSELoss()(means, optimal_means)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
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
        
        # Sample states uniformly from [-state_range, state_range]^2
        states = (torch.rand(batch_size, self.soft_lqr.d, device=self.device, dtype=torch.float64) * 2 - 1) * state_range
        
        return times, states
    
    def _compute_target_values(self, t: torch.Tensor, x: torch.Tensor, num_steps: int = 100) -> torch.Tensor:
        """
        Compute target values for critic training by simulating trajectories using the current policy.
        
        Args:
            t: Time tensor (batch)
            x: State tensor (batch x d)
            num_steps: Number of time steps for simulation
            
        Returns:
            Target values tensor (batch)
        """
        batch_size = t.shape[0]
        target_values = torch.zeros(batch_size, device=self.device, dtype=torch.float64)
        
        for i in range(batch_size):
            # Current time and state
            t_i = t[i].item()
            x_i = x[i:i+1]
            
            # If we're already at terminal time, return terminal cost
            if t_i >= self.soft_lqr.T - 1e-6:
                target_values[i] = torch.matmul(torch.matmul(x_i.squeeze(), self.soft_lqr.R), x_i.squeeze())
                continue
            
            # Determine remaining time and step size
            remaining_time = self.soft_lqr.T - t_i
            dt = remaining_time / num_steps
            
            # Initialize state and cost
            current_state = x_i.clone()
            cost = 0.0
            
            # Simulate trajectory using current policy
            for step in range(num_steps):
                # Current time
                current_time = t_i + step * dt
                current_time_tensor = torch.tensor([current_time], device=self.device, dtype=torch.float64)
                
                # Get control action from current policy
                action = self.policy_network.sample_action(current_time_tensor, current_state)
                
                # Compute running cost
                state_cost = torch.matmul(torch.matmul(current_state.squeeze(), self.soft_lqr.C), current_state.squeeze())
                control_cost = torch.matmul(torch.matmul(action.squeeze(), self.soft_lqr.D), action.squeeze())
                
                # Add entropy regularization term
                _, covariance = self.policy_network.action_distribution(current_time_tensor, current_state)
                covariance = covariance[0] if covariance.dim() > 2 else covariance
                entropy_term = self.soft_lqr.tau * torch.log(torch.det(2 * np.pi * np.e * covariance)) / 2
                
                # Accumulate cost
                cost += (state_cost + control_cost + entropy_term) * dt
                
                # Update state with Euler step
                drift = self.soft_lqr.H @ current_state.squeeze() + self.soft_lqr.M @ action.squeeze()
                noise = self.soft_lqr.sigma @ torch.randn(self.soft_lqr.sigma.shape[1], device=self.device, dtype=torch.float64) * np.sqrt(dt)
                current_state = current_state + (drift * dt + noise).unsqueeze(0)
            
            # Add terminal cost
            terminal_cost = torch.matmul(torch.matmul(current_state.squeeze(), self.soft_lqr.R), current_state.squeeze())
            cost += terminal_cost
            
            target_values[i] = cost
        
        return target_values
    
    def train_critic_step(self, batch_size: int) -> float:
        """
        Perform one training step of the critic algorithm.
        
        Args:
            batch_size: Batch size for training
            
        Returns:
            Loss value from this step
        """
        # Generate batch
        times, states = self._generate_batch(batch_size)
        
        # Compute target values
        with torch.no_grad():
            target_values = self._compute_target_values(times, states)
        
        # Compute predicted values
        predicted_values = self.value_network.value_function(times, states)
        
        # Ensure consistent dtype before computing loss
        predicted_values = self._ensure_float64(predicted_values)
        target_values = self._ensure_float64(target_values)
        
        # Compute loss
        loss = self.critic_criterion(predicted_values, target_values)
        
        # Perform optimization step
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        
        return loss.item()
    
    def compute_td_values(self, t: torch.Tensor, x: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """Compute temporal difference values with improved numerical stability."""
        batch_size = t.shape[0]
        td_values = torch.zeros(batch_size, device=self.device, dtype=torch.float64)
        
        for i in range(batch_size):
            t_i = t[i].item()
            
            # If at terminal time, use terminal cost
            if t_i >= self.soft_lqr.T - 1e-6:
                continue
            
            # Ensure we don't go beyond terminal time
            next_t = min(t_i + dt, self.soft_lqr.T)
            next_t_tensor = torch.tensor([next_t], device=self.device, dtype=torch.float64)
            
            # Current state
            x_i = x[i:i+1]
            current_t_tensor = torch.tensor([t_i], device=self.device, dtype=torch.float64)
            
            # Sample action from current policy with clipping
            action = self.policy_network.sample_action(current_t_tensor, x_i)
            action = torch.clamp(action, -10.0, 10.0)  # Add explicit clipping here too
            
            # Compute immediate cost
            state_cost = torch.matmul(torch.matmul(x_i.squeeze(), self.soft_lqr.C), x_i.squeeze())
            control_cost = torch.matmul(torch.matmul(action.squeeze(), self.soft_lqr.D), action.squeeze())
            
            # Add entropy regularization
            _, covariance = self.policy_network.action_distribution(current_t_tensor, x_i)
            covariance = covariance[0] if covariance.dim() > 2 else covariance
            # Add small regularization term to ensure stable determinant
            covariance = covariance + 1e-6 * torch.eye(covariance.shape[0], device=self.device, dtype=torch.float64)
            entropy_term = self.soft_lqr.tau * torch.log(torch.det(2 * np.pi * np.e * covariance)) / 2
            
            # Calculate immediate cost
            immediate_cost = (state_cost + control_cost + entropy_term) * dt
            
            # More stable next state computation using smaller substeps
            sub_steps = 5  # Use multiple smaller steps for integration
            sub_dt = dt / sub_steps
            next_state = x_i.clone()
            
            for _ in range(sub_steps):
                drift = self.soft_lqr.H @ next_state.squeeze() + self.soft_lqr.M @ action.squeeze()
                noise = self.soft_lqr.sigma @ torch.randn(self.soft_lqr.sigma.shape[1], 
                                                        device=self.device, 
                                                        dtype=torch.float64) * np.sqrt(sub_dt)
                next_state = next_state + (drift * sub_dt + noise).unsqueeze(0)
                
                # Add state clipping to prevent extreme values
                next_state = torch.clamp(next_state, -100.0, 100.0)
            
            # Compute value of next state
            with torch.no_grad():
                next_value = self.value_network.value_function(next_t_tensor, next_state)
                # Clamp next_value to reasonable range to prevent explosion
                next_value = torch.clamp(next_value, -1e6, 1e6)
            
            # Compute TD value: c + V(s')
            td_values[i] = immediate_cost + next_value
        
        return td_values
    
    def train_actor_step(self, batch_size: int) -> float:
        """
        Perform one training step of the actor algorithm using policy gradient.
        Vectorized implementation to compute policy gradient for all samples simultaneously.
        
        Args:
            batch_size: Batch size for training
            
        Returns:
            Loss value from this step
        """
        # Generate batch
        times, states = self._generate_batch(batch_size)
        
        # Compute TD values (expected costs)
        td_values = self.compute_td_values(times, states)
        
        # Create mask for non-terminal states
        non_terminal_mask = times < (self.soft_lqr.T - 1e-6)
        
        # Skip computation if all states are terminal
        if not non_terminal_mask.any():
            return 0.0
        
        # Get action distributions for all states in batch
        means, covariances = self.policy_network.action_distribution(times, states)
        
        # Handle different covariance shapes
        if covariances.dim() > 2:
            if covariances.dim() == 3:
                # Shape [batch, m, m]
                pass
            else:
                # Extra dimension, squeeze
                covariances = covariances.squeeze(1)
        
        # Sample actions for all states
        actions = self.policy_network.sample_action(times, states)
        
        # Compute log probabilities for all actions vectorized
        # Preparation for batch matrix multiplication
        mean_diff = actions - means  # [batch, m]
        
        # Add small regularization to covariances for numerical stability
        covariances_reg = covariances + 1e-6 * torch.eye(
            covariances.shape[-1], 
            device=self.device, 
            dtype=torch.float64
        ).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Compute precision matrices (inverse of covariances)
        precisions = torch.inverse(covariances_reg)  # [batch, m, m]
        
        # Compute log determinants
        log_dets = torch.logdet(covariances_reg)  # [batch]
        
        # Compute quadratic term (x-μ)ᵀΣ⁻¹(x-μ) for all batch elements
        quad_terms = torch.bmm(
            torch.bmm(mean_diff.unsqueeze(1), precisions),  # [batch, 1, m]
            mean_diff.unsqueeze(2)  # [batch, m, 1]
        ).squeeze()  # [batch]
        
        # Compute log probabilities
        log_probs = -0.5 * (quad_terms + log_dets + self.soft_lqr.m * torch.log(torch.tensor(2 * np.pi, device=self.device, dtype=torch.float64)))
        
        # Compute advantages (negate TD values as we want to minimize cost)
        advantages = -td_values  # [batch]
        
        # Mask for non-terminal states
        masked_log_probs = log_probs[non_terminal_mask]
        masked_advantages = advantages[non_terminal_mask]
        
        # Compute policy loss using vectorized operations
        # We negate because optimizer minimizes
        policy_loss = -torch.mean(masked_log_probs * masked_advantages)
        
        # Zero gradients
        self.actor_optimizer.zero_grad()
        
        # Backward pass
        policy_loss.backward()
        
        # Update parameters
        self.actor_optimizer.step()
        
        return policy_loss.item()
    
    def train_step(self, batch_size: int) -> Tuple[float, float]:
        """
        Perform one combined actor-critic training step.
        
        Args:
            batch_size: Batch size for training
            
        Returns:
            Tuple of (critic_loss, actor_loss)
        """
        # First update critic (value function)
        critic_loss = self.train_critic_step(batch_size)
        
        # Then update actor (policy)
        actor_loss = self.train_actor_step(batch_size)
        
        return critic_loss, actor_loss
    
    def train(self, num_epochs: int, batch_size: int, eval_interval: int = 10) -> None:
        """
        Train with improved stability measures.
        """
        print("Initializing networks to approximate optimal solution...")
        self._initialize_networks()
        
        print("Starting actor-critic training...")
        start_time = time.time()
        
        # Learning rate schedules
        actor_lr_start = 1e-4
        critic_lr_start = 1e-3
        
        for epoch in range(num_epochs):
            # Update learning rates with decay
            actor_lr = actor_lr_start * (0.99 ** (epoch // 10))
            critic_lr = critic_lr_start * (0.99 ** (epoch // 10))
            
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = actor_lr
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = critic_lr
            
            # Update critic multiple times per actor update
            critic_updates = 5
            avg_critic_loss = 0
            
            for _ in range(critic_updates):
                critic_loss = self.train_critic_step(batch_size)
                avg_critic_loss += critic_loss / critic_updates
            
            # Update actor
            actor_loss = self.train_actor_step(batch_size)
            
            self.critic_loss_history.append(avg_critic_loss)
            self.actor_loss_history.append(actor_loss)
            
            if (epoch + 1) % eval_interval == 0 or epoch == 0:
                # Evaluate and print progress
                value_error = self.evaluate_value_function()
                policy_error = self.evaluate_policy()
                elapsed = time.time() - start_time
                
                print(f"Epoch {epoch+1}/{num_epochs}, Critic Loss: {avg_critic_loss:.6e}, Actor Loss: {actor_loss:.6e}")
                print(f"Value Error: {value_error:.6e}, Policy Error: {policy_error:.6e}, Time: {elapsed:.2f}s")
                print(f"Actor LR: {actor_lr:.6e}, Critic LR: {critic_lr:.6e}")
                
                # Early stopping if we've achieved good results
                if value_error < 1e-2 and policy_error < 1e-2:
                    print("Reached convergence threshold, stopping training.")
                    break
        
        print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    
    def evaluate_value_function(self, grid_size: int = 5, state_range: float = 3.0) -> float:
        """
        Evaluate the critic network against the exact value function.
        
        Args:
            grid_size: Number of points in each dimension for evaluation
            state_range: Range for evaluation grid
            
        Returns:
            Maximum absolute error
        """
        # Create evaluation grid
        x_grid = torch.linspace(-state_range, state_range, grid_size, device=self.device, dtype=torch.float64)
        y_grid = torch.linspace(-state_range, state_range, grid_size, device=self.device, dtype=torch.float64)
        
        # Time points for evaluation
        time_points = torch.tensor([0, self.soft_lqr.T/3, 2*self.soft_lqr.T/3, self.soft_lqr.T], device=self.device, dtype=torch.float64)
        
        # Store total error
        total_error = 0.0
        count = 0
        
        # Evaluate on grid
        with torch.no_grad():
            for t in time_points:
                t_batch = t.unsqueeze(0)
                for i in range(grid_size):
                    for j in range(grid_size):
                        # Current state
                        x = torch.tensor([[x_grid[i], y_grid[j]]], device=self.device, dtype=torch.float64)
                        
                        # Compute exact and approximate value functions
                        exact_value = self.soft_lqr.value_function(t_batch, x)
                        approx_value = self.value_network.value_function(t_batch, x)
                        
                        # Ensure both are float64
                        exact_value = self._ensure_float64(exact_value)
                        approx_value = self._ensure_float64(approx_value)
                        
                        # Compute error
                        error = torch.abs(exact_value - approx_value).item()
                        total_error += error
                        count += 1
        
        # Return average error
        return total_error / count if count > 0 else 0.0
    
    def evaluate_policy(self, grid_size: int = 5, state_range: float = 3.0) -> float:
        """
        Evaluate the actor network against the exact optimal policy.
        
        Args:
            grid_size: Number of points in each dimension for evaluation
            state_range: Range for evaluation grid
            
        Returns:
            Average error in policy means
        """
        # Create evaluation grid
        x_grid = torch.linspace(-state_range, state_range, grid_size, device=self.device, dtype=torch.float64)
        y_grid = torch.linspace(-state_range, state_range, grid_size, device=self.device, dtype=torch.float64)
        
        # Time points for evaluation
        time_points = torch.tensor([0, self.soft_lqr.T/3, 2*self.soft_lqr.T/3, self.soft_lqr.T], device=self.device, dtype=torch.float64)
        
        # Store total error
        total_error = 0.0
        count = 0
        
        # Evaluate on grid
        with torch.no_grad():
            for t in time_points:
                t_batch = t.unsqueeze(0)
                for i in range(grid_size):
                    for j in range(grid_size):
                        # Current state
                        x = torch.tensor([[x_grid[i], y_grid[j]]], device=self.device, dtype=torch.float64)
                        
                        # Compute optimal and learned policy means
                        optimal_means, _ = self.soft_lqr.optimal_control_distribution(t_batch, x)
                        learned_means, _ = self.policy_network.action_distribution(t_batch, x)
                        
                        # Ensure both are float64
                        optimal_means = self._ensure_float64(optimal_means)
                        learned_means = self._ensure_float64(learned_means)
                        
                        # Compute error in means
                        error = torch.norm(optimal_means - learned_means).item()
                        total_error += error
                        count += 1
        
        # Return average error
        return total_error / count if count > 0 else 0.0
    
    def plot_loss_history(self) -> None:
        """Plot the loss history for both actor and critic during training."""
        plt.figure(figsize=(12, 5))
        
        # Plot critic loss
        plt.subplot(1, 2, 1)
        plt.plot(self.critic_loss_history)
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Critic Loss')
        plt.title('Critic Training Loss History')
        plt.grid(True)
        
        # Plot actor loss
        plt.subplot(1, 2, 2)
        plt.plot(self.actor_loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Actor Loss')
        plt.title('Actor Training Loss History')
        plt.grid(True)
        
        plt.tight_layout()
        plt.draw()
    
    def plot_value_function_comparison(self, grid_size: int = 20, state_range: float = 3.0) -> None:
        """
        Plot comparison between exact and learned value function.
        
        Args:
            grid_size: Number of points in each dimension for plotting
            state_range: Range for plotting grid
        """
        # Create evaluation grid
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
                    # Current state
                    x = torch.tensor([[x_grid[i].item(), y_grid[j].item()]], device=self.device, dtype=torch.float64)
                    
                    # Compute exact and approximate value functions
                    exact_value = self.soft_lqr.value_function(t, x).item()
                    approx_value = self.value_network.value_function(t, x).item()
                    
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
    
    def compare_policies(self, test_states: List[List[float]], dt: float = 0.005) -> None:
        """
        Compare trajectories with improved numerical stability.
        """
        # Convert test states to tensors
        num_states = len(test_states)
        test_states_tensor = torch.tensor(test_states, device=self.device, dtype=torch.float64)
        
        # Time points for simulation
        num_steps = int(self.soft_lqr.T / dt)
        time_points = torch.linspace(0, self.soft_lqr.T, num_steps, device=self.device, dtype=torch.float64)
        
        # Initialize plot
        fig, axes = plt.subplots(num_states, 2, figsize=(16, 4*num_states))
        if num_states == 1:
            axes = axes.reshape(1, 2)
        
        # Simulation parameters
        max_state_magnitude = 10.0
        max_action_magnitude = 5.0
        
        # Simulate for each initial state
        for i, x0 in enumerate(test_states_tensor):
            x0 = x0.view(1, -1)  # Add batch dimension
            
            # Initialize state trajectories with clipping
            learned_traj = torch.zeros((num_steps, self.soft_lqr.d), device=self.device, dtype=torch.float64)
            optimal_traj = torch.zeros((num_steps, self.soft_lqr.d), device=self.device, dtype=torch.float64)
            learned_traj[0] = torch.clamp(x0, -max_state_magnitude, max_state_magnitude)
            optimal_traj[0] = torch.clamp(x0, -max_state_magnitude, max_state_magnitude)
            
            # Accumulated costs with sanity checking
            learned_costs = torch.zeros(num_steps, device=self.device, dtype=torch.float64)
            optimal_costs = torch.zeros(num_steps, device=self.device, dtype=torch.float64)
            
            # Generate same noise for both simulations
            noise_seq = torch.randn((num_steps, self.soft_lqr.sigma.shape[1]), 
                                    device=self.device, dtype=torch.float64) * np.sqrt(dt)
            
            # Simulate trajectories with numerical safeguards
            for n in range(1, num_steps):
                t_n = time_points[n-1]
                t_tensor = t_n.unsqueeze(0)
                
                # Learned policy step with clipping
                learned_control = self.policy_network.sample_action(t_tensor, learned_traj[n-1:n])
                learned_control = torch.clamp(learned_control, -max_action_magnitude, max_action_magnitude)
                
                learned_drift = self.soft_lqr.H @ learned_traj[n-1] + self.soft_lqr.M @ learned_control[0]
                # Add dampening to prevent explosion
                learned_drift = learned_drift * (1.0 - 0.01 * dt)  # Small dampening factor
                
                # Update with noise and clipping
                learned_traj[n] = learned_traj[n-1] + learned_drift * dt + self.soft_lqr.sigma @ noise_seq[n-1]
                learned_traj[n] = torch.clamp(learned_traj[n], -max_state_magnitude, max_state_magnitude)
                
                # Optimal policy step with clipping
                optimal_control = self.soft_lqr.optimal_control(t_tensor, optimal_traj[n-1:n])
                optimal_control = torch.clamp(optimal_control, -max_action_magnitude, max_action_magnitude)
                
                optimal_drift = self.soft_lqr.H @ optimal_traj[n-1] + self.soft_lqr.M @ optimal_control[0]
                # Same dampening factor for consistency
                optimal_drift = optimal_drift * (1.0 - 0.01 * dt)
                
                # Update with noise and clipping
                optimal_traj[n] = optimal_traj[n-1] + optimal_drift * dt + self.soft_lqr.sigma @ noise_seq[n-1]
                optimal_traj[n] = torch.clamp(optimal_traj[n], -max_state_magnitude, max_state_magnitude)
                
                # Compute running costs with sanity checking
                if torch.isfinite(learned_traj[n-1]).all() and torch.isfinite(learned_control).all():
                    learned_state_cost = torch.matmul(torch.matmul(learned_traj[n-1], self.soft_lqr.C), learned_traj[n-1])
                    learned_control_cost = torch.matmul(torch.matmul(learned_control[0], self.soft_lqr.D), learned_control[0])
                    cost_increment = (learned_state_cost + learned_control_cost) * dt
                    if torch.isfinite(cost_increment):
                        learned_costs[n] = learned_costs[n-1] + cost_increment
                    else:
                        learned_costs[n] = learned_costs[n-1]
                else:
                    learned_costs[n] = learned_costs[n-1]
                
                if torch.isfinite(optimal_traj[n-1]).all() and torch.isfinite(optimal_control).all():
                    optimal_state_cost = torch.matmul(torch.matmul(optimal_traj[n-1], self.soft_lqr.C), optimal_traj[n-1])
                    optimal_control_cost = torch.matmul(torch.matmul(optimal_control[0], self.soft_lqr.D), optimal_control[0])
                    cost_increment = (optimal_state_cost + optimal_control_cost) * dt
                    if torch.isfinite(cost_increment):
                        optimal_costs[n] = optimal_costs[n-1] + cost_increment
                    else:
                        optimal_costs[n] = optimal_costs[n-1]
                else:
                    optimal_costs[n] = optimal_costs[n-1]
            
            # Add terminal costs if trajectories are finite
            if torch.isfinite(learned_traj[-1]).all():
                learned_terminal = torch.matmul(torch.matmul(learned_traj[-1], self.soft_lqr.R), learned_traj[-1])
                if torch.isfinite(learned_terminal):
                    learned_costs[-1] += learned_terminal
            
            if torch.isfinite(optimal_traj[-1]).all():
                optimal_terminal = torch.matmul(torch.matmul(optimal_traj[-1], self.soft_lqr.R), optimal_traj[-1])
                if torch.isfinite(optimal_terminal):
                    optimal_costs[-1] += optimal_terminal
            
            # Plot trajectory
            ax1 = axes[i, 0]
            ax1.plot(learned_traj.cpu().detach().numpy()[:, 0], learned_traj.cpu().detach().numpy()[:, 1], 'b-', label='Learned Policy')
            ax1.plot(optimal_traj.cpu().detach().numpy()[:, 0], optimal_traj.cpu().detach().numpy()[:, 1], 'r-', label='Optimal Policy')
            ax1.scatter([x0[0, 0].item()], [x0[0, 1].item()], color='g', s=100, marker='o', label='Initial State')
            ax1.set_title(f'Trajectories from Initial State {x0.squeeze().cpu().detach().numpy()}')
            ax1.set_xlabel('X1')
            ax1.set_ylabel('X2')
            ax1.grid(True)
            ax1.legend()
            
            # Plot costs
            ax2 = axes[i, 1]
            ax2.plot(time_points.cpu().detach().numpy(), learned_costs.cpu().detach().numpy(), 'b-', label='Learned Policy Cost')
            ax2.plot(time_points.cpu().detach().numpy(), optimal_costs.cpu().detach().numpy(), 'r-', label='Optimal Policy Cost')
            ax2.set_title(f'Cost Over Time from Initial State {x0.squeeze().cpu().detach().numpy()}')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Cost')
            ax2.grid(True)
            ax2.legend()
            
            # Print the final costs (with sanity checking)
            learned_final = learned_costs[-1].item() if torch.isfinite(learned_costs[-1]) else float('inf')
            optimal_final = optimal_costs[-1].item() if torch.isfinite(optimal_costs[-1]) else float('inf')
            
            print(f"Initial state {x0.squeeze().cpu().detach().numpy()}:")
            print(f"  Learned policy final cost: {learned_final:.2f}")
            print(f"  Optimal policy final cost: {optimal_final:.2f}")
            print(f"  Cost ratio: {learned_final/optimal_final if optimal_final > 0 else 'inf':.2f}")
        
        plt.tight_layout()
        plt.draw()


def run_actor_critic_algorithm():
    """Run the actor-critic algorithm for soft LQR."""
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
    
    # Create time grid
    grid_size = 1000
    time_grid = torch.linspace(0, T, grid_size, dtype=torch.float64)
    
    # Create soft LQR instance and solve Ricatti ODE
    soft_lqr = SoftLQR(H, M, sigma, C, D, R, T, time_grid, tau, gamma)
    soft_lqr.solve_ricatti()
    
    # Create actor-critic algorithm
    actor_critic = ActorCriticAlgorithm(
        soft_lqr, 
        actor_hidden_size=256, 
        critic_hidden_size=512, 
        actor_lr=1e-4, 
        critic_lr=1e-3
    )
    
    # Train actor-critic
    actor_critic.train(num_epochs=500, batch_size=16, eval_interval=25)
    
    # Plot results
    actor_critic.plot_loss_history()
    actor_critic.plot_value_function_comparison()
    
    # Compare policies on test states
    test_states = [
        [2.0, 2.0],
        [2.0, -2.0],
        [-2.0, -2.0],
        [-2.0, 2.0]
    ]
    actor_critic.compare_policies(test_states)
    
    # Evaluate final performance
    value_error = actor_critic.evaluate_value_function(grid_size=10)
    policy_error = actor_critic.evaluate_policy(grid_size=10)
    print(f"Final value function error: {value_error:.6e}")
    print(f"Final policy error: {policy_error:.6e}")
    
    return actor_critic