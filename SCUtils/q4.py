import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import time
from .q2 import SoftLQR

class PolicyNN(nn.Module):
    """Neural network to approximate the optimal policy for soft LQR.
    Outputs the mean and covariance of the control distribution."""

    def __init__(self, d=2, hidden_size=256, device=torch.device("cpu")):
        super(PolicyNN, self).__init__()
        self.d = d  # State dimension
        self.device = device

        # Create neural network with 3 hidden layers, all explicitly using float64
        # Note: Original code used 3 hidden layers (time_embedding + 2 hidden)
        self.layer1 = nn.Linear(1, hidden_size, device=device, dtype=torch.float64)
        self.layer2 = nn.Linear(hidden_size, hidden_size, device=device, dtype=torch.float64)
        self.layer3 = nn.Linear(hidden_size, hidden_size, device=device, dtype=torch.float64)

        # Output layers for action distribution parameters
        # Output for phi(t) - mapping from state to control mean (m x d) -> (d x d) in this case
        self.phi_output = nn.Linear(hidden_size, d * d, device=device, dtype=torch.float64)

        # Output for elements of the L matrix for Sigma(t) - covariance matrix
        self.sigma_output_L = nn.Linear(hidden_size, d * (d + 1) // 2, device=device, dtype=torch.float64)

        # Precompute indices for building the lower triangular matrix
        self.tri_indices = torch.tril_indices(d, d).to(device)
        self.eps = 1e-9 # For numerical stability of Sigma

    def _ensure_float64(self, tensor):
        """Helper method to ensure tensors are consistently float64."""
        if tensor.dtype != torch.float64:
            return tensor.to(torch.float64)
        return tensor

    def forward(self, t):
        """
        Forward pass to compute policy parameters phi(t) and Sigma(t).
        MODIFIED: Always returns 3D tensors [batch, d, d].
        """
        # Ensure t is properly shaped and dtype
        t = self._ensure_float64(t.view(-1, 1))
        batch_size = t.shape[0] # Get batch size early

        # Forward pass through the network
        x = torch.relu(self.layer1(t))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x)) # Changed sigmoid to relu for consistency

        # Compute phi - matrix mapping state to mean of control distribution
        phi_flat = self.phi_output(x)
        phi = phi_flat.view(batch_size, self.d, self.d) # Shape [batch_size, d, d]

        # Compute L for covariance matrix Sigma = LL^T (ensures positive semi-definiteness)
        L_flat = self.sigma_output_L(x)

        # Create batched lower triangular matrices
        L = torch.zeros(batch_size, self.d, self.d, device=self.device, dtype=torch.float64)

        # Fill in the lower triangular elements
        L[:, self.tri_indices[0], self.tri_indices[1]] = L_flat

        # Compute Sigma = LL^T + eps*I for numerical stability
        Sigma = torch.bmm(L, L.transpose(1, 2)) + self.eps * torch.eye(self.d, device=self.device, dtype=torch.float64).unsqueeze(0) # Shape [batch_size, d, d]

        # REMOVED SQUEEZE LOGIC
        # if batch_size == 1:
        #     phi = phi.squeeze(0)
        #     Sigma = Sigma.squeeze(0)

        return phi, Sigma # Always returns [batch_size, d, d] tensors

    def action_distribution(self, t, x):
        """
        Compute the parameters of the control distribution N(mean, Sigma).

        Args:
            t: Time tensor (scalar or batch_size)
            x: State tensor (d or batch_size x d)

        Returns:
            Tuple of (means, covariance) for the control distribution
        """
        # Ensure inputs are float64
        t = self._ensure_float64(t)
        x = self._ensure_float64(x)

        # Ensure t has batch dimension for forward pass
        if t.dim() == 0:
            t = t.unsqueeze(0) # [1]

        # Get policy parameters
        phi, Sigma = self.forward(t) # phi: [b, d, d], Sigma: [b, d, d]

        # Ensure x has batch dimension [batch_size, d]
        if x.dim() == 1:
            x = x.unsqueeze(0) # Make it [1, d]

        # Handle potential batch size mismatches (e.g., single t, multiple x)
        batch_size_t = phi.shape[0]
        batch_size_x = x.shape[0]

        if batch_size_t == 1 and batch_size_x > 1:
            phi = phi.expand(batch_size_x, -1, -1)
            Sigma = Sigma.expand(batch_size_x, -1, -1)
            batch_size = batch_size_x
        elif batch_size_x == 1 and batch_size_t > 1:
             x = x.expand(batch_size_t, -1)
             batch_size = batch_size_t
        elif batch_size_t == batch_size_x:
             batch_size = batch_size_t
        else:
             raise ValueError(f"Batch size mismatch: t leads to batch {batch_size_t}, x has batch {batch_size_x}")


        # Ensure x is 3D for batch matrix multiplication: [batch_size, d, 1]
        x_3d = x.unsqueeze(2)

        # Compute mean = phi(t) @ x
        means = torch.bmm(phi, x_3d).squeeze(2) # Shape [batch_size, d]

        return means, Sigma

    def sample_action(self, t, x):
        """
        Sample from the control distribution N(mean, Sigma).

        Args:
            t: Time tensor (scalar or batch_size)
            x: State tensor (d or batch_size x d)

        Returns:
            Sampled control actions (batch_size x d)
        """
        # Ensure inputs are float64
        t = self._ensure_float64(t)
        x = self._ensure_float64(x)

        means, covariances = self.action_distribution(t, x) # means: [b, d], cov: [b, d, d]
        batch_size = means.shape[0]

        # Create the distribution object(s)
        # Need to handle potential batching issues if means/covariances are single tensors
        if batch_size == 1 and means.dim() == 1: # Single sample case
             dist = torch.distributions.MultivariateNormal(means, covariance_matrix=covariances)
             actions = dist.sample()
        else: # Batch case
            try:
                # Use Cholesky for stability in batch
                L = torch.linalg.cholesky(covariances) # [b, d, d]
                # Generate standard normal samples
                z = torch.randn(batch_size, self.d, 1, device=self.device, dtype=torch.float64) # [b, d, 1]
                # Transform: mean + L @ z
                actions = means + torch.bmm(L, z).squeeze(-1) # [b, d]
            except torch.linalg.LinAlgError as e:
                 print(f"Cholesky failed: {e}. Falling back to sampling (less efficient).")
                 # Fallback for potential numerical issues (less efficient)
                 actions = torch.zeros_like(means)
                 for i in range(batch_size):
                     dist_i = torch.distributions.MultivariateNormal(means[i], covariance_matrix=covariances[i])
                     actions[i] = dist_i.sample()

        return actions


class ActorAlgorithm:
    """Implementation of the actor algorithm for soft LQR, which learns
    the optimal policy using the exact value function."""

    def __init__(self,
                 soft_lqr: SoftLQR, # Assume SoftLQR class is defined and works
                 hidden_size: int = 256,
                 learning_rate: float = 1e-4,
                 device: torch.device = torch.device("cpu")):
        """
        Initialize the actor algorithm.

        Args:
            soft_lqr: Instance of SoftLQR class with exact value function
            hidden_size: Size of hidden layers in the policy network
            learning_rate: Learning rate for optimization
            device: Device to use for computation
        """
        self.soft_lqr = soft_lqr
        self.device = device
        self.d = soft_lqr.d # State dimension

        # Create policy network and ensure it uses float64
        self.policy_network = PolicyNN(d=self.d, hidden_size=hidden_size, device=device)
        self.policy_network.to(dtype=torch.float64) # Ensure parameters are float64

        # Create optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        # --- For tracking progress ---
        self.loss_history = [] # Stores surrogate objective (KL divergence)
        self.gradient_norm_history = [] # Stores L2 norm of gradients
        self.eval_epochs = [] # Stores epochs where evaluation was performed
        self.mean_error_history = [] # Stores parameter error for phi(t)x
        self.cov_error_history = [] # Stores parameter error for Sigma
        self.policy_gradient_objective_history = []

    def _rollout_trajectory(self, x0: torch.Tensor, N_steps: int) -> List[Tuple[float, torch.Tensor, torch.Tensor, float, float, torch.Tensor]]:
        """
        Simulate one episode using the *current learned policy*.
        Internal method for monitoring the policy gradient objective.

        Args:
            x0: Initial state tensor (d)
            N_steps: Number of steps in the rollout

        Returns:
            List of tuples: (t_n, x_n, a_n, cost_n, log_prob_n, x_{n+1})
        """
        self.policy_network.eval() # Use eval mode for consistent sampling during rollout

        x0 = self._ensure_float64(x0).to(self.device)
        trajectory = []
        x_current = x0.clone()
        t_current = 0.0
        dt = self.soft_lqr.T / N_steps
        sqrt_dt = np.sqrt(dt)

        # Pre-generate noise for the episode
        dW_episode = torch.randn((N_steps, self.soft_lqr.sigma.shape[1]), device=self.device, dtype=torch.float64) * sqrt_dt

        # Get necessary matrices (ensure float64)
        H = self._ensure_float64(self.soft_lqr.H)
        M = self._ensure_float64(self.soft_lqr.M)
        C = self._ensure_float64(self.soft_lqr.C)
        D = self._ensure_float64(self.soft_lqr.D) # Use original D for cost
        sigma = self._ensure_float64(self.soft_lqr.sigma)

        with torch.no_grad(): # No gradients needed for rollout simulation
            for n in range(N_steps):
                t_tensor = torch.tensor(t_current, device=self.device, dtype=torch.float64)

                # 1. Get policy distribution and sample action
                # Need batch dim for network methods
                means, covariances = self.policy_network.action_distribution(t_tensor.unsqueeze(0), x_current.unsqueeze(0))
                dist = torch.distributions.MultivariateNormal(means.squeeze(0), covariance_matrix=covariances.squeeze(0))
                a_n = dist.sample() # Shape [d]

                # 2. Calculate log probability ln p^θ(a_n | t_n, x_n)
                log_prob_n = dist.log_prob(a_n) # Scalar

                # 3. Calculate instantaneous cost cost_n = x_n^T C x_n + a_n^T D a_n
                cost_state = torch.dot(x_current, C @ x_current)
                cost_control = torch.dot(a_n, D @ a_n) # Use original D
                cost_n = cost_state + cost_control

                # 4. Simulate next state using Euler-Maruyama
                drift = H @ x_current + M @ a_n
                diffusion = sigma @ dW_episode[n]
                x_next = x_current + drift * dt + diffusion

                # Store step data including x_next
                trajectory.append((t_current, x_current.clone(), a_n, cost_n.item(), log_prob_n.item(), x_next.clone()))

                # Update state and time
                x_current = x_next
                t_current += dt
                t_current = min(t_current, self.soft_lqr.T) # Cap at T

        return trajectory
   
    def _compute_pg_objective_from_rollout(self, trajectory: List[Tuple[float, torch.Tensor, torch.Tensor, float, float, torch.Tensor]]) -> float:
        """
        Compute the average (advantage * log_prob) for monitoring purposes.
        Uses the exact value function v* from self.soft_lqr.

        Args:
            trajectory: List of (t_n, x_n, a_n, cost_n, log_prob_n, x_{n+1}) tuples.

        Returns:
            The average policy gradient objective value for the episode (float).
        """
        N_steps = len(trajectory)
        if N_steps == 0:
            return 0.0

        dt = self.soft_lqr.T / N_steps
        total_objective = 0.0

        # Pre-calculate all needed v* values
        t_vals_v = torch.tensor([traj[0] for traj in trajectory] + [self.soft_lqr.T], device=self.device, dtype=torch.float64)
        x_vals_v = torch.stack([traj[1] for traj in trajectory] + [trajectory[-1][5]], dim=0) # x_0, ..., x_{N-1}, x_N

        # Ensure SoftLQR components are ready for value function calculation
        # self.soft_lqr.to(self.device, torch.float64) # Assumed helper
        v_star_values = self.soft_lqr.value_function(t_vals_v, x_vals_v).to(torch.float64) # [N+1] values

        with torch.no_grad(): # No gradients needed for this calculation
            for n in range(N_steps):
                t_n, x_n, a_n, cost_n, log_prob_n_float, x_next = trajectory[n]

                # Advantage calculation using v*
                v_star_n = v_star_values[n]
                v_star_n_plus_1 = v_star_values[n+1]
                advantage = torch.tensor(cost_n * dt, dtype=torch.float64, device=self.device) + v_star_n_plus_1 - v_star_n

                # Use the stored log_prob (float)
                log_prob_n = torch.tensor(log_prob_n_float, dtype=torch.float64, device=self.device)

                # Accumulate objective term
                objective_term_n = advantage * log_prob_n
                total_objective += objective_term_n.item() # Accumulate as float

        return total_objective / N_steps # Return average

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
            Tuple of (times, states) tensors [batch_size], [batch_size, d]
        """
        # Sample times uniformly from [0, T]
        times = torch.rand(batch_size, device=self.device, dtype=torch.float64) * self.soft_lqr.T

        # Sample states uniformly from [-state_range, state_range]^d
        states = (torch.rand(batch_size, self.d, device=self.device, dtype=torch.float64) * 2 - 1) * state_range

        return times, states

    def compute_kl_divergence_loss(self, t, x) -> torch.Tensor:
        """
        Compute the KL divergence KL( pi_learned || pi_optimal ) averaged over the batch.
        This serves as the surrogate objective function to minimize.

        Args:
            t: Time tensor (batch_size)
            x: State tensor (batch_size x d)

        Returns:
            Mean KL divergence (scalar tensor, requires grad).
        """
        batch_size = t.shape[0]

        # Ensure inputs are float64
        t = self._ensure_float64(t)
        x = self._ensure_float64(x)

        # Get action distributions from both policies
        # Learned: N(mu_learned, Sigma_learned)
        mu_learned, Sigma_learned = self.policy_network.action_distribution(t, x)

        # Optimal: N(mu_optimal, Sigma_optimal)
        # Ensure SoftLQR components are float64 and on the correct device
        # self.soft_lqr.to(self.device, torch.float64) # Helper method assumed in SoftLQR
        mu_optimal, Sigma_optimal = self.soft_lqr.optimal_control_distribution(t, x)

        # Ensure optimal_covariance is properly shaped for batch operations
        if Sigma_optimal.dim() < 3 and batch_size > 0:
            Sigma_optimal = Sigma_optimal.unsqueeze(0).expand(batch_size, -1, -1)
        elif Sigma_optimal.dim() == 2 and batch_size == 0: # Handle empty batch case
             return torch.tensor(0.0, device=self.device, dtype=torch.float64, requires_grad=True)


        # Compute precision matrix (inverse of optimal covariance)
        # Add small identity for stability before inverse
        eps_eye = torch.eye(self.d, device=self.device, dtype=torch.float64) * 1e-6
        Sigma_optimal_stable = Sigma_optimal + eps_eye
        try:
            optimal_precision = torch.inverse(Sigma_optimal_stable) # [b, d, d]
        except torch.linalg.LinAlgError:
             print("Warning: Optimal Sigma inverse failed. Using pseudo-inverse.")
             optimal_precision = torch.linalg.pinv(Sigma_optimal_stable)


        # Compute mean difference
        mean_diff = mu_optimal - mu_learned # [b, d]

        # Compute KL divergence terms (vectorized)
        # Term 1: tr(Sigma_optimal_inv @ Sigma_learned)
        # torch.bmm: [b, d, d] @ [b, d, d] -> [b, d, d]
        # torch.diagonal extracts diagonals for trace sum
        term1 = torch.diagonal(torch.bmm(optimal_precision, Sigma_learned), dim1=-2, dim2=-1).sum(dim=-1) # [b]

        # Term 2: (mu_optimal - mu_learned)^T @ Sigma_optimal_inv @ (mu_optimal - mu_learned)
        # mean_diff.unsqueeze(-1): [b, d, 1]
        # optimal_precision @ mean_diff.unsqueeze(-1): [b, d, 1]
        # mean_diff.unsqueeze(1): [b, 1, d]
        # torch.bmm(mean_diff.unsqueeze(1), ...): [b, 1, 1] -> squeeze -> [b]
        term2 = torch.bmm(mean_diff.unsqueeze(1), torch.bmm(optimal_precision, mean_diff.unsqueeze(-1))).squeeze() # [b]

        # Term 3: -d (dimensionality)
        term3 = -torch.tensor(self.d, device=self.device, dtype=torch.float64) # scalar

        # Term 4: log(det(Sigma_optimal) / det(Sigma_learned)) = logdet(Sigma_optimal) - logdet(Sigma_learned)
        # Add small identity for stability before logdet
        Sigma_learned_stable = Sigma_learned + eps_eye
        logdet_optimal = torch.logdet(Sigma_optimal_stable) # [b]
        logdet_learned = torch.logdet(Sigma_learned_stable) # [b]
        term4 = logdet_optimal - logdet_learned # [b]

        # Compute KL divergence: 0.5 * (term1 + term2 + term3 + term4)
        kl_div = 0.5 * (term1 + term2 + term3 + term4) # [b]

        # Return mean KL divergence over the batch
        return kl_div.mean() # scalar, requires_grad

    def train_step(self, batch_size: int) -> Tuple[float, float]:
        """
        Perform one training step of the actor algorithm.

        Args:
            batch_size: Batch size for training

        Returns:
            Tuple[float, float]: (Loss value from this step, Gradient norm)
        """
        # Generate batch
        times, states = self._generate_batch(batch_size)

        # Compute KL divergence loss (surrogate objective)
        loss = self.compute_kl_divergence_loss(times, states)

        # Perform optimization step
        self.optimizer.zero_grad()
        # Check if loss requires grad (it should)
        if loss.requires_grad:
            loss.backward()

            # --- Calculate Gradient Norm ---
            total_norm = 0.0
            for p in self.policy_network.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm = total_norm ** 0.5
            # ------------------------------

            # Optional: Gradient Clipping
            # torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)

            self.optimizer.step()
        else:
             # Handle cases where loss might not require grad (e.g., batch_size=0)
             grad_norm = 0.0
             print("Warning: Loss did not require grad. Skipping optimizer step.")


        return loss.item(), grad_norm

    def train(self, num_epochs: int, batch_size: int, eval_interval: int = 10, N_steps_rollout: int = 100) -> None: # Added N_steps_rollout
        """
        Train the actor network.

        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for KL divergence loss calculation
            eval_interval: Interval for evaluation and progress reporting
            N_steps_rollout: Number of steps for the monitoring rollout (can differ from N used elsewhere)
        """
        print(f"Starting actor training for {num_epochs} epochs...")
        start_time = time.time()

        # Initial evaluation before training
        self.policy_network.eval() # Eval mode for initial checks
        mean_error, cov_error = self.evaluate()
        # --- Calculate initial PG objective ---
        x0_eval = self._generate_batch(1, state_range=3.0)[1].squeeze(0) # Sample one initial state
        initial_traj = self._rollout_trajectory(x0_eval, N_steps_rollout)
        initial_pg_obj = self._compute_pg_objective_from_rollout(initial_traj)
        # --- Store initial values ---
        self.eval_epochs.append(0)
        self.mean_error_history.append(mean_error)
        self.cov_error_history.append(cov_error)
        self.policy_gradient_objective_history.append(initial_pg_obj) # Store initial PG obj
        print(f"Epoch 0/{num_epochs}, Initial Mean Err: {mean_error:.6e}, Initial Cov Err: {cov_error:.6e}, Initial PG Obj: {initial_pg_obj:.6e}")


        for epoch in range(num_epochs):
            self.policy_network.train() # Set network to training mode
            loss, grad_norm = self.train_step(batch_size)
            self.loss_history.append(loss)
            self.gradient_norm_history.append(grad_norm) # Store the norm

            if (epoch + 1) % eval_interval == 0 or epoch == num_epochs - 1:
                self.policy_network.eval() # Set network to evaluation mode for consistency
                mean_error, cov_error = self.evaluate()
                # --- Calculate PG objective for monitoring ---
                x0_eval = self._generate_batch(1, state_range=3.0)[1].squeeze(0) # Sample one initial state
                current_traj = self._rollout_trajectory(x0_eval, N_steps_rollout)
                current_pg_obj = self._compute_pg_objective_from_rollout(current_traj)
                # --- Store eval results ---
                self.eval_epochs.append(epoch + 1)
                self.mean_error_history.append(mean_error)
                self.cov_error_history.append(cov_error)
                self.policy_gradient_objective_history.append(current_pg_obj) # Store current PG obj
                # --- END Store ---
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{num_epochs}, KL Loss: {loss:.6e}, Grad Norm: {grad_norm:.6e}, Mean Err: {mean_error:.6e}, Cov Err: {cov_error:.6e}, PG Obj: {current_pg_obj:.6e}, Time: {elapsed:.2f}s")

        print(f"Training completed in {time.time() - start_time:.2f} seconds.")


    def evaluate(self, grid_size: int = 5, state_range: float = 3.0, time_points: Optional[List[float]] = None) -> Tuple[float, float]:
        """
        Evaluate the actor network against the exact optimal policy.

        Args:
            grid_size: Number of points in each dimension for evaluation grid
            state_range: Range for evaluation grid [-range, range]
            time_points: Specific time points for evaluation, defaults to [0, T/3, 2T/3, T]

        Returns:
            Tuple of (average_mean_error, average_covariance_error) over the grid
        """
        self.policy_network.eval() # Ensure evaluation mode

        # Set default time points if not provided
        if time_points is None:
            time_points = [0, self.soft_lqr.T/3, 2*self.soft_lqr.T/3, self.soft_lqr.T]

        # Create evaluation grid for states
        x_coords = torch.linspace(-state_range, state_range, grid_size, device=self.device, dtype=torch.float64)
        # Create all combinations of coordinates for d=2
        state_grid = torch.cartesian_prod(x_coords, x_coords) # Shape [grid_size*grid_size, d]

        # Convert time points to tensor
        time_tensor = torch.tensor(time_points, device=self.device, dtype=torch.float64)

        # Store errors
        total_mean_error = 0.0
        total_cov_error = 0.0
        count = 0

        # Evaluate on grid
        with torch.no_grad():
            for t in time_tensor:
                # Process all states for this time point in a batch
                t_batch = t.repeat(state_grid.shape[0]) # [grid_size*grid_size]

                # Compute optimal policy parameters for the batch
                optimal_means, optimal_cov = self.soft_lqr.optimal_control_distribution(t_batch, state_grid)
                # Ensure optimal_cov is batched correctly
                if optimal_cov.dim() < 3:
                     optimal_cov = optimal_cov.unsqueeze(0).expand(state_grid.shape[0], -1, -1)


                # Compute learned policy parameters for the batch
                learned_means, learned_cov = self.policy_network.action_distribution(t_batch, state_grid)

                # Compute errors using vectorized operations
                mean_error = torch.norm(optimal_means - learned_means, dim=1).mean().item() # Avg norm over batch
                cov_error = torch.norm(optimal_cov - learned_cov, dim=(1, 2)).mean().item() # Avg norm over batch

                # Accumulate errors (weighted by number of states evaluated at this time)
                total_mean_error += mean_error * state_grid.shape[0]
                total_cov_error += cov_error * state_grid.shape[0]
                count += state_grid.shape[0]

        # Return average errors over the entire grid (all times, all states)
        avg_mean_error = total_mean_error / count if count > 0 else 0.0
        avg_cov_error = total_cov_error / count if count > 0 else 0.0
        return avg_mean_error, avg_cov_error

    # --- Plotting Functions ---

    def plot_loss_history(self) -> None:
        """Plot the surrogate objective (KL divergence loss) history."""
        if not self.loss_history:
            print("No loss history to plot.")
            return
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history)
        # Use symlog if loss can be negative, otherwise log or linear
        min_loss = min(self.loss_history) if self.loss_history else 0
        if min_loss < 0:
             plt.yscale('symlog')
        else:
             plt.yscale('log')
        plt.xlabel('Training Step (Epoch)')
        plt.ylabel('Surrogate Objective (KL Divergence)')
        plt.title('Actor Training: Surrogate Objective History')
        plt.grid(True)
        plt.tight_layout()
        plt.draw()

    def plot_gradient_norm_history(self) -> None:
        """Plot the policy gradient norm history."""
        if not self.gradient_norm_history:
            print("No gradient norm history to plot.")
            return
        plt.figure(figsize=(10, 5))
        plt.plot(self.gradient_norm_history)
        plt.yscale('log') # Norms are non-negative, log scale is often useful
        plt.xlabel('Training Step (Epoch)')
        plt.ylabel('Policy Gradient L2 Norm')
        plt.title('Actor Training: Gradient Norm History')
        plt.grid(True)
        plt.tight_layout()
        plt.draw()

    def plot_parameter_error_history(self) -> None:
        """Plot the parameter error history recorded during evaluation."""
        if not self.eval_epochs:
            print("No parameter error history to plot (evaluation likely not run periodically).")
            return

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot Mean Error (Error in phi(t)x or equivalent mean)
        axes[0].plot(self.eval_epochs, self.mean_error_history, 'o-', label='Mean Error Norm')
        axes[0].set_ylabel('Avg ||μ_learned - μ*||')
        axes[0].set_title('Actor Training: Parameter Error History')
        axes[0].grid(True)
        axes[0].legend()
        axes[0].set_yscale('log')

        # Plot Covariance Error
        axes[1].plot(self.eval_epochs, self.cov_error_history, 's-', color='red', label='Covariance Error Norm')
        axes[1].set_xlabel('Training Step (Epoch)')
        axes[1].set_ylabel('Avg ||Σ_learned - Σ*||_F')
        axes[1].grid(True)
        axes[1].legend()
        axes[1].set_yscale('log')

        plt.tight_layout()
        plt.draw()

    def compare_policies(self, test_states: List[List[float]], dt: float = 0.01) -> None:
        """
        Compare trajectories using the learned policy versus the optimal policy.
        Optimized implementation with reduced redundant calculations.

        Args:
            test_states: List of initial states to test
            dt: Time step for simulation
        """
        self.policy_network.eval() # Ensure eval mode for sampling

        # Convert test states to tensors
        num_states = len(test_states)
        test_states_tensor = torch.tensor(test_states, device=self.device, dtype=torch.float64)

        # Time points for simulation
        num_steps = int(self.soft_lqr.T / dt) + 1 # Ensure T is included
        time_points = torch.linspace(0, self.soft_lqr.T, num_steps, device=self.device, dtype=torch.float64)

        # Initialize plot
        fig, axes = plt.subplots(num_states, 2, figsize=(16, 4*num_states), squeeze=False) # Ensure axes is always 2D

        # Pre-compute matrices used in every iteration
        sqrt_dt = np.sqrt(dt)
        H = self.soft_lqr.H.to(dtype=torch.float64)
        M = self.soft_lqr.M.to(dtype=torch.float64)
        sigma = self.soft_lqr.sigma.to(dtype=torch.float64)
        C = self.soft_lqr.C.to(dtype=torch.float64)
        D = self.soft_lqr.D.to(dtype=torch.float64) # Use original D for cost
        R = self.soft_lqr.R.to(dtype=torch.float64)

        # Simulation loop for each initial state
        for i, x0_single in enumerate(test_states_tensor):
            x0 = x0_single.view(1, -1)  # Add batch dimension [1, d]

            # Initialize trajectories and costs
            learned_traj = torch.zeros((num_steps, self.d), device=self.device, dtype=torch.float64)
            optimal_traj = torch.zeros((num_steps, self.d), device=self.device, dtype=torch.float64)
            learned_traj[0] = x0_single
            optimal_traj[0] = x0_single

            learned_costs = torch.zeros(num_steps, device=self.device, dtype=torch.float64)
            optimal_costs = torch.zeros(num_steps, device=self.device, dtype=torch.float64)

            # Generate noise once for both simulations
            noise_seq = torch.randn((num_steps - 1, sigma.shape[1]), device=self.device, dtype=torch.float64) * sqrt_dt

            # Simulate trajectories step-by-step
            current_learned_x = x0_single
            current_optimal_x = x0_single
            cumulative_learned_cost = 0.0
            cumulative_optimal_cost = 0.0

            with torch.no_grad(): # No gradients needed for simulation/comparison
                for n in range(num_steps - 1):
                    t_n = time_points[n]
                    t_tensor = t_n.unsqueeze(0) # [1]

                    # Get controls from policies
                    learned_control = self.policy_network.sample_action(t_tensor, current_learned_x.unsqueeze(0)).squeeze(0) # [d]
                    optimal_control = self.soft_lqr.optimal_control(t_tensor, current_optimal_x.unsqueeze(0)).squeeze(0) # [d]

                    # Calculate costs for step n
                    learned_state_cost = torch.dot(current_learned_x, C @ current_learned_x)
                    learned_control_cost = torch.dot(learned_control, D @ learned_control)
                    cumulative_learned_cost += (learned_state_cost + learned_control_cost) * dt
                    learned_costs[n+1] = cumulative_learned_cost # Store cumulative cost up to end of step n

                    optimal_state_cost = torch.dot(current_optimal_x, C @ current_optimal_x)
                    optimal_control_cost = torch.dot(optimal_control, D @ optimal_control)
                    cumulative_optimal_cost += (optimal_state_cost + optimal_control_cost) * dt
                    optimal_costs[n+1] = cumulative_optimal_cost # Store cumulative cost up to end of step n

                    # Calculate drift terms
                    learned_drift = H @ current_learned_x + M @ learned_control
                    optimal_drift = H @ current_optimal_x + M @ optimal_control

                    # Apply common noise term
                    noise_term = sigma @ noise_seq[n]

                    # Update states for next iteration
                    current_learned_x = current_learned_x + learned_drift * dt + noise_term
                    current_optimal_x = current_optimal_x + optimal_drift * dt + noise_term

                    # Store next state
                    learned_traj[n+1] = current_learned_x
                    optimal_traj[n+1] = current_optimal_x

            # Add terminal costs
            learned_terminal = torch.dot(learned_traj[-1], R @ learned_traj[-1])
            optimal_terminal = torch.dot(optimal_traj[-1], R @ optimal_traj[-1])
            learned_costs[-1] += learned_terminal
            optimal_costs[-1] += optimal_terminal

            # --- Plotting ---
            ax1 = axes[i, 0]
            ax1.plot(learned_traj.cpu().numpy()[:, 0], learned_traj.cpu().numpy()[:, 1], 'b-', label='Learned Policy')
            ax1.plot(optimal_traj.cpu().numpy()[:, 0], optimal_traj.cpu().numpy()[:, 1], 'r-', label='Optimal Policy')
            ax1.scatter([x0_single[0].item()], [x0_single[1].item()], color='g', s=100, marker='o', label='Initial State')
            ax1.set_title(f'Trajectories from {x0_single.cpu().numpy()}')
            ax1.set_xlabel('X1')
            ax1.set_ylabel('X2')
            ax1.grid(True)
            ax1.legend()

            ax2 = axes[i, 1]
            ax2.plot(time_points.cpu().numpy(), learned_costs.cpu().numpy(), 'b-', label='Learned Policy Cost')
            ax2.plot(time_points.cpu().numpy(), optimal_costs.cpu().numpy(), 'r-', label='Optimal Policy Cost')
            ax2.set_title(f'Cumulative Cost from {x0_single.cpu().numpy()}')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Cost')
            ax2.grid(True)
            ax2.legend()

            print(f"Initial state {x0_single.cpu().numpy()}:")
            print(f"  Learned policy final cost: {learned_costs[-1].item():.4f}")
            print(f"  Optimal policy final cost: {optimal_costs[-1].item():.4f}")
            print(f"  Cost difference: {(learned_costs[-1] - optimal_costs[-1]).item():.4f}")

        plt.tight_layout()
        plt.draw()

    def plot_policy_gradient_loss_history(self) -> None:
        """Plot the monitored policy gradient objective history."""
        if not self.eval_epochs or not self.policy_gradient_objective_history:
            print("No policy gradient objective history to plot (rollouts likely not run periodically).")
            return
        # Ensure lengths match if initial value was added
        epochs_to_plot = self.eval_epochs
        pg_obj_to_plot = self.policy_gradient_objective_history
        if len(epochs_to_plot) != len(pg_obj_to_plot):
            print(f"Warning: Mismatch in lengths of eval_epochs ({len(epochs_to_plot)}) and pg_objective_history ({len(pg_obj_to_plot)}). Plotting available data.")
            min_len = min(len(epochs_to_plot), len(pg_obj_to_plot))
            epochs_to_plot = epochs_to_plot[:min_len]
            pg_obj_to_plot = pg_obj_to_plot[:min_len]


        plt.figure(figsize=(10, 5))
        plt.plot(epochs_to_plot, pg_obj_to_plot, 'o-')
        plt.yscale('symlog') # This objective can be positive or negative
        plt.xlabel('Training Step (Epoch)')
        plt.ylabel('Avg(Advantage * log_prob)')
        plt.title('Actor Training: Monitored Policy Gradient Objective')
        plt.grid(True)
        plt.tight_layout()
        plt.draw()


# --- Main Execution ---
def run_actor_algorithm():
    """Run the actor algorithm for soft LQR."""
    # Setup device
    if torch.cuda.is_available():
         device = torch.device("cuda")
         print("Using CUDA device")
    else:
         device = torch.device("cpu")
         print("Using CPU device")

    # Set the problem matrices as specified in assignment (Figure 1)
    H = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64) * 0.5
    M = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    sigma = torch.eye(2, dtype=torch.float64) * 0.5
    C = torch.tensor([[1.0, 0.1], [0.1, 1.0]], dtype=torch.float64) * 1.0
    D = torch.tensor([[1.0, 0.1], [0.1, 1.0]], dtype=torch.float64) * 0.1 # Original D
    R = torch.tensor([[1.0, 0.3], [0.3, 1.0]], dtype=torch.float64) * 10.0

    # Set parameters (e.g., from Exercise 2.1 or hints if different)
    T = 0.5
    tau = 0.1   # Example value
    gamma = 10.0 # Example value

    # Create time grid for Riccati solver
    grid_size = 1001 # Finer grid for accuracy
    time_grid = torch.linspace(0, T, grid_size, dtype=torch.float64, device=device)

    # Create soft LQR instance and solve Ricatti ODE for exact v* and pi*
    print("Solving SoftLQR Riccati equation...")
    soft_lqr = SoftLQR(H, M, sigma, C, D, R, T, time_grid, tau, gamma)
    soft_lqr.solve_ricatti()
    print("SoftLQR solved.")

    # Create actor algorithm instance
    actor = ActorAlgorithm(soft_lqr,
                           hidden_size=256,
                           learning_rate=1e-3, # May need tuning
                           device=device)

    # Train actor
    actor.train(num_epochs=500, 
                batch_size=256,
                eval_interval=25, 
                N_steps_rollout=100) 

    # --- Plot results ---
    actor.plot_loss_history()
    actor.plot_gradient_norm_history()
    actor.plot_parameter_error_history()
    actor.plot_policy_gradient_loss_history()

    # Compare policies on test states
    test_states = [
        [2.0, 2.0],
        [2.0, -2.0],
        [-2.0, -2.0],
        [-2.0, 2.0]
    ]
    actor.compare_policies(test_states)

    # Evaluate final performance
    mean_error, cov_error = actor.evaluate(grid_size=10) # Evaluate on a denser grid
    print(f"\nFinal Average Mean Error (over grid): {mean_error:.6e}")
    print(f"Final Average Covariance Error (over grid): {cov_error:.6e}")

    return actor