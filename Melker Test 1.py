import numpy as np
import torch

class LQRStochasticControl:
    def __init__(self, time_grid, T, A, B, Q, R, x0):
        if T <= 0:
            raise ValueError("T must be greater than 0.")
        
        self.time_grid = torch.tensor(time_grid, dtype=torch.float32)
        self.T = T
        self.A = np.asarray(A)
        self.B = np.asarray(B)
        self.Q = np.asarray(Q)
        self.R = np.asarray(R)
        self.x0 = np.asarray(x0)
        
        self.validate_dimensions()
    
    def validate_dimensions(self):
        n = self.A.shape[0]
        m = self.B.shape[1]
        
        if self.A.shape != (n, n):
            raise ValueError("Matrix A must be square with dimensions (n, n).")
        if self.B.shape[0] != n:
            raise ValueError("Matrix B must have dimensions (n, m).")
        if self.Q.shape != (n, n):
            raise ValueError("Matrix Q must be square with dimensions (n, n).")
        if self.R.shape != (m, m):
            raise ValueError("Matrix R must be square with dimensions (m, m).")
        if self.x0.shape != (n,):
            raise ValueError("Initial state x0 must have dimensions (n,).")
    
    def solve_riccati(self):
        N = len(self.time_grid)
        n = self.A.shape[0]
        m = self.B.shape[1]
        
        P = np.zeros((N, n, n))
        P[-1] = self.Q  # Terminal condition
        
        dt = float(self.time_grid[1] - self.time_grid[0])
        
        for i in reversed(range(N - 1)):
            P_next = P[i + 1]
            A_trans_P = self.A.T @ P_next
            BRB_inv = self.B @ np.linalg.inv(self.R) @ self.B.T @ P_next
            P_dot = -(A_trans_P @ self.A - A_trans_P + P_next @ self.A + self.Q - BRB_inv)
            P[i] = P_next - dt * P_dot
        
        return P
    
    def calculate_optimal_control(self):
        # Implement the calculation of the optimal control here
        pass

def is_controllable(A, B):
    n = A.shape[0]
    controllability_matrix = B
    for i in range(1, n):
        controllability_matrix = np.hstack((controllability_matrix, np.linalg.matrix_power(A, i) @ B))
    rank = np.linalg.matrix_rank(controllability_matrix)
    return rank == n


# Check controllability
controllable = is_controllable(A, B)
print("The system is controllable:" if controllable else "The system is not controllable.")

# Example usage
time_grid = np.linspace(0, 10, 1000)
T = 10
A = np.array([[1.0, 2.0], [2.0, 1.0]])
B = np.array([[1.0], [0.0]])
Q = np.array([[1.0, 0.0], [0.0, 1.0]])
R = np.array([[1.0]])
x0 = np.array([1.0, 0.0])

lqr = LQRStochasticControl(time_grid, T, A, B, Q, R, x0)
P = lqr.solve_riccati()

print(P)
