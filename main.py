import jax
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
import numpy as np
from functools import partial

# =========================================================
# 1. B-Spline Basis & Linear Dynamics Utils
# =========================================================

class BSplineBasis:
    """Matrix-based Cubic B-Spline Generator"""
    def __init__(self, n_cp, horizon, degree=3):
        self.n_cp = n_cp
        self.horizon = horizon
        self.degree = degree
        self.basis_matrix = self._precompute_basis_matrix()

    def _basis_function(self, i, k, t, knots):
        if k == 0:
            return jnp.where((t >= knots[i]) & (t < knots[i+1]), 1.0, 0.0)
        denom1 = knots[i+k] - knots[i]
        term1 = 0.0 if denom1 == 0 else ((t - knots[i]) / denom1) * self._basis_function(i, k-1, t, knots)
        denom2 = knots[i+k+1] - knots[i+1]
        term2 = 0.0 if denom2 == 0 else ((knots[i+k+1] - t) / denom2) * self._basis_function(i+1, k-1, t, knots)
        return term1 + term2

    def _precompute_basis_matrix(self):
        knots = jnp.arange(self.n_cp + self.degree + 1)
        t_seq = jnp.linspace(0, self.n_cp - self.degree, self.horizon)
        M = jnp.zeros((self.horizon, self.n_cp))
        for i in range(self.n_cp):
            M = M.at[:, i].set(self._basis_function(i, self.degree, t_seq, knots))
        # Normalize rows
        row_sums = M.sum(axis=1, keepdims=True)
        return M / (row_sums + 1e-10)


class AnalyticalUnicycle:
    """
    Derives Linear Matrices A, B, C from Physics
    State z: [x, y, cos, sin]
    Input u: [v, w]
    """
    def __init__(self, dt=0.1):
        self.dt = dt

    @partial(jax.jit, static_argnums=(0,))
    def get_linear_matrices(self, z, target_vel=None):
        """
        Returns LTI approximation at current state z
        z_next = A @ z + B @ u + C
        """
        c = z[2]
        s = z[3]
        
        # 1. System Matrix A (Identity for discrete time position/orientation)
        A = jnp.eye(4)
        
        # 2. Input Matrix B (Jacobian-like but extracted analytically)
        # x += v * c * dt -> B[0,0] = c*dt
        # y += v * s * dt -> B[1,0] = s*dt
        # c += -s * w * dt -> B[2,1] = -s*dt
        # s += c * w * dt -> B[3,1] = c*dt
        B = jnp.array([
            [c * self.dt, 0.0],
            [s * self.dt, 0.0],
            [0.0, -s * self.dt],
            [0.0,  c * self.dt]
        ])

        # 3. Bias Vector C (Drift / Gravity / Target Velocity)
        # If tracking a moving target, C contains -v_target * dt
        if target_vel is None:
            C = jnp.zeros(4)
        else:
            # Assuming state is error state or we compensate target motion here
            # For this visual demo, we assume C handles the 'natural drift' if any.
            # Here we set C to zero for static frame, but ready for expansion.
            C = jnp.zeros(4)
            
        return A, B, C

# =========================================================
# 2. Fast Matrix Rollout & Prediction
# =========================================================

class MatrixPredictor:
    """
    Converts Iterative Dynamics into Single Matrix Multiplication
    Z_traj = Phi @ U_coeffs + D
    """
    def __init__(self, basis: BSplineBasis, model: AnalyticalUnicycle):
        self.basis = basis
        self.model = model
        self.H = basis.horizon
        self.N_cp = basis.n_cp

    @partial(jax.jit, static_argnums=(0,))
    def build_prediction_matrices(self, z_init):
        """
        Constructs the giant matrices Phi and D such that:
        Trajectory (H x 4) = (Phi @ U_flat) + D
        """
        A, B, C = self.model.get_linear_matrices(z_init)
        
        # Since A is Identity, prediction is simple summation (integration)
        # z_k = z_0 + Sum(B @ u_i) + Sum(C)
        
        # 1. Control Effect (Phi)
        # U_seq (H x 2) = Basis (H x N_cp) @ Coeffs (N_cp x 2)
        # We need to map Coeffs -> Z_traj directly.
        
        # B is (4, 2). Basis is (H, N_cp).
        # We want Phi such that vec(Z) = Phi @ vec(Coeffs)
        # But to keep it simple, let's do it in steps.
        
        # Effective B sequence (Assuming LTI at z0 for the horizon)
        # In reality, B changes with theta. For LTI MPC, we fix B at z0.
        # This is the "Linear" approximation.
        
        # Integration Matrix (Lower Triangular of ones)
        # L = [[1,0..], [1,1..]...]
        L = jnp.tril(jnp.ones((self.H, self.H)))
        
        # U_seq = M @ C_coeffs
        M = self.basis.basis_matrix # (H, N_cp)
        
        # Velocity/Omega Sequence -> Position/Angle Deltas
        # Delta_Z = (U_seq @ B.T)  <-- This applies B to inputs
        # But we need cumulative sum.
        
        # Let's simplify: 
        # Z_pos = z0 + cumsum(B @ U) + cumsum(C)
        
        return A, B, C, M, L

    @partial(jax.jit, static_argnums=(0,))
    def linear_rollout(self, z_init, u_coeffs):
        """
        Predicts trajectory using Matrix ops (No Loop!)
        """
        A, B, C, M, L = self.build_prediction_matrices(z_init)
        
        # 1. Reconstruct Input Sequence
        u_seq = M @ u_coeffs # (H, 2)
        
        # 2. Apply B matrix (Input -> State Delta)
        # delta_z = u_seq @ B.T  => (H, 4)
        delta_z = u_seq @ B.T
        
        # 3. Integrate (Cumulative Sum)
        cum_delta = jnp.cumsum(delta_z, axis=0)
        
        # 4. Add Initial State & Bias (C)
        # bias term (linear drift)
        cum_bias = jnp.cumsum(jnp.tile(C, (self.H, 1)), axis=0)
        
        z_traj = z_init + cum_delta + cum_bias
        
        # Optional: Normalize Orientation (Safe-guard)
        # In pure linear MPC we skip this, but for hybrid we can do it at the end
        norm = jnp.sqrt(z_traj[:, 2]**2 + z_traj[:, 3]**2)
        z_traj = z_traj.at[:, 2].set(z_traj[:, 2]/norm)
        z_traj = z_traj.at[:, 3].set(z_traj[:, 3]/norm)
        
        return z_traj

# =========================================================
# 3. Linearized QP Projector (The Bottleneck Fix)
# =========================================================

class LinearQPProjector:
    """
    Solves QP using Explicit Matrices derived from Analytical Koopman
    Minimize: 0.5 * x.T P x + q.T x
    Subject to: G x <= h  (Converted from l <= Ax <= u for jaxopt)
    """
    def __init__(self, predictor: MatrixPredictor):
        self.predictor = predictor
        # OSQP Solver
        self.qp = jaxopt.OSQP(jit=True)

    @partial(jax.jit, static_argnums=(0,))
    def compute_qp_matrices(self, z_init, target_pos, obs_pos, obs_r, mean_coeffs):
        """
        Constructs P, q, G, h for the QP (Gx <= h format)
        """
        # [수정 1] L_int로 변수명 일치
        A_sys, B_sys, C_sys, M_spline, L_int = self.predictor.build_prediction_matrices(z_init)
        
        H, N_cp = M_spline.shape
        
        # --- 1. Cost Function (Tracking) ---
        P = jnp.eye(N_cp * 2) * 1.0 
        q = -1.0 * mean_coeffs.flatten()
        
        # --- 2. Constraints (Converted to Gx <= h) ---
        
        # A. Input Limits (Box constraints split into two inequalities)
        # Variable x is [v0, w0, v1, w1, ...]
        
        # Upper Bound: I * x <= u_input
        # v <= 2.0, w <= 2.0
        G_upper = jnp.eye(N_cp * 2)
        h_upper = jnp.tile(jnp.array([2.0, 2.0]), N_cp)
        
        # Lower Bound: I * x >= l_input  ->  -I * x <= -l_input
        # v >= 0.0, w >= -2.0
        G_lower = -jnp.eye(N_cp * 2)
        # l_input is [0.0, -2.0], so -l_input is [0.0, 2.0]
        h_lower = jnp.tile(jnp.array([0.0, 2.0]), N_cp)
        
        # B. Obstacle Avoidance (Linearized)
        # Originally: n.T * (p_nom + J(c - c_nom)) >= R
        # Rearranged: - (n.T * J) * c <= - (R - n.T * p_nom + n.T * J * c_nom)
        
        traj_pred = self.predictor.linear_rollout(z_init, mean_coeffs)
        pos_pred = traj_pred[:, :2]
        
        dists = jnp.linalg.norm(pos_pred - obs_pos, axis=1)
        k_crit = jnp.argmin(dists)
        p_crit = pos_pred[k_crit]
        
        denom = jnp.linalg.norm(p_crit - obs_pos)
        n_vec = (p_crit - obs_pos) / (denom + 1e-6)
        
        # Jacobian Calculation
        LM = jnp.matmul(L_int, M_spline) # (H, N_cp)
        influence = LM[k_crit] # (N_cp,)
        
        Jx = influence * B_sys[0,0]
        Jy = influence * B_sys[1,0]
        
        # Row for A_obs * c (Flattened)
        A_obs_block = jnp.stack([Jx * n_vec[0] + Jy * n_vec[1], jnp.zeros(N_cp)], axis=1).flatten()
        A_obs_row = A_obs_block.reshape(1, -1)
        
        # Constraint Value (Right Hand Side for >= formulation)
        val_obs = obs_r + 0.2 - jnp.dot(n_vec, p_crit) + jnp.dot(A_obs_block, mean_coeffs.flatten())
        
        # Convert to <= formulation: -A * c <= -val
        G_obs = -A_obs_row
        h_obs = jnp.array([-val_obs])
        
        # --- Stack All Constraints ---
        G = jnp.vstack([G_upper, G_lower, G_obs])
        h = jnp.concatenate([h_upper, h_lower, h_obs])
        
        return P, q, G, h

    @partial(jax.jit, static_argnums=(0,))
    def project(self, z_init, mean_coeffs, target_pos, obs_pos, obs_r):
        """
        Projects the mean coefficients using standard QP interface
        """
        # [수정 2] G, h 형태로 받음
        P, q, G, h = self.compute_qp_matrices(z_init, target_pos, obs_pos, obs_r, mean_coeffs)
        
        # [수정 3] params_ineq에 (G, h) 전달
        params = dict(params_obj=(P, q), params_eq=None, params_ineq=(G, h))
        
        # Solve
        sol = self.qp.run(init_params=None, **params)
        
        # Reshape back to (N_cp, 2)
        projected_coeffs = sol.params.primal.reshape(-1, 2)
        return projected_coeffs

# =========================================================
# 4. MPPI Controller (Simplified)
# =========================================================

class KoopmanMPPI:
    def __init__(self, basis, predictor, projector):
        self.basis = basis
        self.predictor = predictor
        self.projector = projector
        self.num_samples = 200 # Can handle many more now!
        self.sigma = 0.5 
        self.lambda_ = 1.0

    @partial(jax.jit, static_argnums=(0,))
    def step(self, z_curr, mean_coeffs, target_pos, obs_pos, obs_r, key):
        # 1. QP Projection of Mean (Ensure nominal path is safe)
        safe_mean = self.projector.project(z_curr, mean_coeffs, target_pos, obs_pos, obs_r)
        
        # 2. Sampling (Add noise to safe mean)
        noise = jax.random.normal(key, (self.num_samples, self.basis.n_cp, 2)) * self.sigma
        samples = safe_mean + noise
        
        # 3. Matrix Rollout (Super Fast Parallel Prediction)
        # vmap over samples
        rollout_fn = jax.vmap(self.predictor.linear_rollout, in_axes=(None, 0))
        trajs = rollout_fn(z_curr, samples) # (N, H, 4)
        
        # 4. Cost Calculation
        # Distance to target at end
        final_dist = jnp.linalg.norm(trajs[:, -1, :2] - target_pos, axis=1)
        # Obstacle cost (Soft)
        dists_obs = jnp.linalg.norm(trajs[:, :, :2] - obs_pos, axis=2)
        min_dists = jnp.min(dists_obs, axis=1)
        obs_cost = jnp.where(min_dists < obs_r + 0.2, 1000.0, 0.0)
        
        costs = final_dist + obs_cost
        
        # 5. MPPI Weighting
        weights = jax.nn.softmax(-costs / self.lambda_)
        
        # 6. Weighted Average
        delta_u = samples - safe_mean
        weighted_delta = jnp.sum(weights[:, None, None] * delta_u, axis=0)
        
        # Update
        new_mean = safe_mean + weighted_delta
        
        return new_mean, trajs, weights

# =========================================================
# 5. Main Simulation Loop
# =========================================================

def main():
    # Settings
    H = 30
    N_cp = 8
    dt = 0.1
    
    # Init Modules
    basis = BSplineBasis(N_cp, H)
    model = AnalyticalUnicycle(dt)
    predictor = MatrixPredictor(basis, model)
    projector = LinearQPProjector(predictor)
    mppi = KoopmanMPPI(basis, predictor, projector)
    
    # Init State
    z_curr = jnp.array([0.0, 0.0, 1.0, 0.0]) # x, y, c, s
    mean_coeffs = jnp.zeros((N_cp, 2))
    
    target_pos = jnp.array([8.0, 8.0])
    obs_pos = jnp.array([4.0, 4.0])
    obs_r = 1.0
    
    # Run
    key = jax.random.PRNGKey(0)
    traj_hist = []
    
    print("Starting Linearized Koopman MPPI...")
    
    for t in range(80):
        key, subkey = jax.random.split(key)
        
        # MPPI Step (JIT compiled inside)
        mean_coeffs, trajs, weights = mppi.step(z_curr, mean_coeffs, target_pos, obs_pos, obs_r, subkey)
        
        # Apply Control (First step of spline)
        # In reality, we just take the first u, but here we simulate forward
        u_curr = (basis.basis_matrix @ mean_coeffs)[0] # First control
        
        # Update State (Simulation)
        # Using accurate nonlinear step for 'Real World' simulation
        x, y, c, s = z_curr
        v, w = u_curr
        nx = x + v * c * dt
        ny = y + v * s * dt
        nc = c - s * w * dt
        ns = s + c * w * dt
        norm = np.sqrt(nc**2 + ns**2)
        z_curr = jnp.array([nx, ny, nc/norm, ns/norm])
        
        traj_hist.append(z_curr)
        
        dist = jnp.linalg.norm(z_curr[:2] - target_pos)
        print(f"Step {t}: Pos ({z_curr[0]:.2f}, {z_curr[1]:.2f}) Dist {dist:.2f}")
        
        if dist < 0.5:
            print("Target Reached!")
            break

    # Plotting
    hist = np.array(traj_hist)
    plt.figure(figsize=(8, 8))
    
    # Obstacle
    circle = plt.Circle(obs_pos, obs_r, color='r', alpha=0.5, label='Obstacle')
    plt.gca().add_patch(circle)
    # Safe Margin
    circle_safe = plt.Circle(obs_pos, obs_r + 0.2, color='r', fill=False, linestyle='--', label='Constraint')
    plt.gca().add_patch(circle_safe)
    
    # Paths
    plt.plot(target_pos[0], target_pos[1], 'bx', markersize=12, label='Target')
    plt.plot(hist[:, 0], hist[:, 1], 'k-', linewidth=2, label='Driven Path')
    
    # Samples (Last frame)
    for i in range(20): # Plot 20 samples
        plt.plot(trajs[i, :, 0], trajs[i, :, 1], 'g-', alpha=0.1)
        
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.title("Analytical Koopman MPPI (Matrix Rollout + Linear QP)")
    plt.show()

if __name__ == "__main__":
    main()