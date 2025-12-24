import jax
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
import numpy as np
from functools import partial

# =========================================================
# 1. Look-ahead Dynamics & B-Spline Matrix Utils
# =========================================================

class LookAheadPrecomputer:
    """
    Precomputes the Mapping from B-Spline Coefficients to Trajectory
    Since the model is LINEAR (Point Mass), this mapping is CONSTANT.
    Trajectory = Phi @ Coeffs + Bias
    """
    def __init__(self, n_cp, horizon, dt, degree=3):
        self.n_cp = n_cp
        self.H = horizon
        self.dt = dt
        self.degree = degree
        # Precompute Basis Matrix (H x N_cp)
        self.M = self._compute_bspline_matrix()
        # Precompute Integration Matrix (H x H)
        self.L = jnp.tril(jnp.ones((horizon, horizon))) * dt
        # Precompute Linear Mapping Matrix (H x N_cp)
        self.Phi = self.L @ self.M

    def _compute_bspline_matrix(self):
        # Standard cubic B-spline basis generation
        knots = jnp.arange(self.n_cp + self.degree + 1)
        t_seq = jnp.linspace(0, self.n_cp - self.degree, self.H)
        
        def basis_fn(i, k, t):
            if k == 0: return jnp.where((t >= knots[i]) & (t < knots[i+1]), 1.0, 0.0)
            d1 = knots[i+k] - knots[i]
            t1 = 0.0 if d1 == 0 else ((t - knots[i]) / d1) * basis_fn(i, k-1, t)
            d2 = knots[i+k+1] - knots[i+1]
            t2 = 0.0 if d2 == 0 else ((knots[i+k+1] - t) / d2) * basis_fn(i+1, k-1, t)
            return t1 + t2

        M = jnp.zeros((self.H, self.n_cp))
        for i in range(self.n_cp):
            M = M.at[:, i].set(basis_fn(i, self.degree, t_seq))
        return M / (M.sum(axis=1, keepdims=True) + 1e-10)

    @partial(jax.jit, static_argnums=(0,))
    def get_linear_matrices(self):
        """
        Returns Phi such that: Pos_Trajectory = Phi @ Coeffs + Initial_Pos
        """
        # Velocity Trajectory V = M @ C
        # Position Trajectory P = P0 + cumsum(V * dt)
        # P = P0 + L @ M @ C
        Phi = self.L @ self.M  # (H, N_cp)
        return Phi

# =========================================================
# 2. Linear QP Projector (Constraint Satisfaction)
# =========================================================

class LookAheadQP:
    """
    Projects B-Spline Coefficients into Safe Set.
    Because the model is Look-ahead (Linear), obstacle constraints
    become Linear Inequalities on the coefficients C.
    """
    def __init__(self, precomputer: LookAheadPrecomputer):
        self.pc = precomputer
        self.qp = jaxopt.OSQP(jit=True)
        self.Phi = self.pc.Phi # Constant Matrix!

    @partial(jax.jit, static_argnums=(0,))
    def get_constraints(self, z_lookahead, obs_pos, obs_r, mean_coeffs):
        """
        Linearizes Obstacle Constraints around the mean trajectory.
        Constraint: n^T * (p_traj - obs) >= r
        Sub p_traj = z0 + Phi @ C
        => n^T * Phi @ C >= r - n^T * (z0 - obs)
        """
        # 1. Predict Nominal Trajectory (Look-ahead point)
        # z_lookahead: [x_p, y_p]
        p_nom = z_lookahead + self.Phi @ mean_coeffs
        
        # 2. Find Critical Point (Closest to obstacle)
        dists = jnp.linalg.norm(p_nom - obs_pos, axis=1)
        k_crit = jnp.argmin(dists)
        
        # 3. Linearize (Compute Normal)
        p_c = p_nom[k_crit]
        diff = p_c - obs_pos
        norm_val = jnp.linalg.norm(diff)
        n_vec = diff / (norm_val + 1e-6) # Normal vector
        
        # 4. Formulate A @ c <= b
        # Inequality: -n^T * p(c) <= -r_safe
        # -n^T * (z0 + Phi_row * c) <= -r_safe
        # (-n^T * Phi_row) * c <= -r_safe + n^T * z0
        
        # Extract row of Phi corresponding to critical time step
        phi_row = self.Phi[k_crit] # (N_cp,)
        
        # A_obs (1, 2*N_cp) - Flattened coefficients [cx0, cy0, cx1, cy1...]
        # We need to construct the matrix carefully for flattened C
        # cx terms: n_x * phi_row
        # cy terms: n_y * phi_row
        
        A_x = n_vec[0] * phi_row
        A_y = n_vec[1] * phi_row
        
        # Interleave A_x and A_y to match flattened C: [cx0, cy0, cx1, cy1 ...]
        A_flat = jnp.stack([A_x, A_y], axis=1).flatten()
        A_obs = -A_flat.reshape(1, -1)
        
        # RHS
        safe_r = obs_r + 0.3 # Buffer
        b_val = -safe_r + jnp.dot(n_vec, z_lookahead)
        b_obs = jnp.array([b_val])
        
        return A_obs, b_obs

    @partial(jax.jit, static_argnums=(0,))
    def project(self, noisy_coeffs, z_lookahead, obs_pos, obs_r, mean_coeffs):
        """
        Project a SINGLE sample of coefficients to be safe.
        """
        N_vars = self.pc.n_cp * 2
        
        # 1. Get Linear Constraints from Mean Trajectory (Approximation)
        G_obs, h_obs = self.get_constraints(z_lookahead, obs_pos, obs_r, mean_coeffs)
        
        # 2. Input Limits (Box Constraints)
        # Limit speed of look-ahead point (u_x, u_y)
        v_max = 1.5
        G_upper = jnp.eye(N_vars)
        h_upper = jnp.full(N_vars, v_max)
        G_lower = -jnp.eye(N_vars)
        h_lower = jnp.full(N_vars, v_max) # -x <= v_max -> x >= -v_max
        
        # Stack
        G = jnp.vstack([G_obs, G_upper, G_lower])
        h = jnp.concatenate([h_obs, h_upper, h_lower])
        
        # 3. Objective: Minimize deviation from noisy sample
        # min 0.5 * ||c - c_noisy||^2
        # equivalent to min 0.5 c'Ic - c_noisy'c
        P = jnp.eye(N_vars)
        q = -noisy_coeffs.flatten()
        
        # 4. Solve QP
        sol = self.qp.run(init_params=None, params_obj=(P, q), params_ineq=(G, h))
        return sol.params.primal.reshape(-1, 2)

# =========================================================
# 3. MPPI Controller (Parallelized)
# =========================================================

class LookAheadMPPI:
    def __init__(self, precomputer, projector):
        self.pc = precomputer
        self.proj = projector
        self.num_samples = 200 # GPU can handle thousands
        self.lambda_ = 0.5

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, z_lookahead, mean_coeffs, target_pos, obs_pos, obs_r):
        # 1. Sampling
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, (self.num_samples, self.pc.n_cp, 2)) * 0.5
        raw_samples = mean_coeffs + noise
        
        # 2. Parallel QP Projection (The Magic Step)
        # vmap ensures we solve 200 QPs in parallel on GPU
        project_fn = jax.vmap(self.proj.project, in_axes=(0, None, None, None, None))
        safe_samples = project_fn(raw_samples, z_lookahead, obs_pos, obs_r, mean_coeffs)
        
        # 3. Fast Linear Rollout
        # Pos = z0 + Phi @ C
        # (N, H, 2) = (N, 2) + (H, N_cp) @ (N, N_cp, 2) -> Broadcasting
        trajs = z_lookahead + jnp.einsum('hc,ncd->nhd', self.pc.Phi, safe_samples)
        
        # 4. Cost Calculation
        # Distance to target
        dist_cost = jnp.linalg.norm(trajs[:, -1, :] - target_pos, axis=1) * 10.0
        path_cost = jnp.sum(jnp.linalg.norm(trajs - target_pos, axis=2), axis=1) * 0.1
        
        total_cost = dist_cost + path_cost
        
        # 5. Update Mean
        min_cost = jnp.min(total_cost)
        weights = jax.nn.softmax(-(total_cost - min_cost) / self.lambda_)
        
        new_mean = jnp.sum(weights[:, None, None] * safe_samples, axis=0)
        
        # Exponential Moving Average
        final_mean = 0.8 * new_mean + 0.2 * mean_coeffs
        
        return final_mean, trajs, weights

# =========================================================
# 4. Simulation Loop (Nonlinear Robot, Linear Controller)
# =========================================================

def feedback_linearization_control(z_robot, u_lookahead, L=0.5):
    """
    Maps Look-ahead input (ux, uy) -> Unicycle input (v, w)
    """
    theta = z_robot[2]
    c, s = jnp.cos(theta), jnp.sin(theta)
    
    # Jacobian Inverse
    # [v] = [ c   s ] [ux]
    # [w]   [-s/L c/L] [uy]
    
    ux, uy = u_lookahead
    v = c * ux + s * uy
    w = (-s * ux + c * uy) / L
    return jnp.array([v, w])

def main():
    # Setup
    H = 30
    N_CP = 15
    DT = 0.1
    L_dist = 1.0 # Look-ahead distance
    
    pc = LookAheadPrecomputer(N_CP, H, DT)
    proj = LookAheadQP(pc)
    mppi = LookAheadMPPI(pc, proj)
    
    # Initial State (x, y, theta)
    z_robot = jnp.array([0.0, 0.0, 0.0])
    # Look-ahead State
    z_la = jnp.array([z_robot[0] + L_dist*jnp.cos(z_robot[2]), 
                      z_robot[1] + L_dist*jnp.sin(z_robot[2])])
    
    mean_coeffs = jnp.zeros((N_CP, 2))
    target_pos = jnp.array([8.0, 6.0])
    obs_pos = jnp.array([4.0, 3.0])
    obs_r = 0.5
    
    key = jax.random.PRNGKey(42)
    hist_robot = []
    
    print("Starting Look-ahead MPPI with QP Projection...")
    
    plt.figure(figsize=(8,8))
    
    for t in range(500):
        # 1. Update Look-ahead State
        z_la = jnp.array([z_robot[0] + L_dist*jnp.cos(z_robot[2]), 
                          z_robot[1] + L_dist*jnp.sin(z_robot[2])])
        
        # 2. Run MPPI (Optimizes ux, uy trajectories)
        key, subkey = jax.random.split(key)
        mean_coeffs, trajs, weights = mppi.step(subkey, z_la, mean_coeffs, target_pos, obs_pos, obs_r)
        
        # 3. Get Control Input (First step of spline)
        # u_la_seq = M @ C -> take first row
        u_la_curr = (pc.M @ mean_coeffs)[0] / DT # Scale for velocity
        
        # 4. Convert to Unicycle Input (v, w)
        u_robot = feedback_linearization_control(z_robot, u_la_curr, L_dist)
        v, w = u_robot
        
        # 5. Simulate Nonlinear Robot
        # x += v cos theta * dt
        # y += v sin theta * dt
        # theta += w * dt
        nx = z_robot[0] + v * jnp.cos(z_robot[2]) * DT
        ny = z_robot[1] + v * jnp.sin(z_robot[2]) * DT
        nt = z_robot[2] + w * DT
        z_robot = jnp.array([nx, ny, nt])
        
        hist_robot.append(z_robot)
        
        # Check Goal
        if jnp.linalg.norm(z_robot[:2] - target_pos) < 0.5:
            print("Goal Reached!")
            break
            
        # Visualization
        if t % 5 == 0:
            plt.clf()
            # Obstacle
            plt.gca().add_patch(plt.Circle(obs_pos, obs_r, color='r', alpha=0.5))
            plt.gca().add_patch(plt.Circle(obs_pos, obs_r+0.3, fill=False, linestyle='--', color='r'))
            
            # Trajectories (Look-ahead Points)
            for i in range(0, 200, 20): # Plot 10 samples
                plt.plot(trajs[i, :, 0], trajs[i, :, 1], 'g-', alpha=0.1)
            
            # Robot Path
            h = np.array(hist_robot)
            plt.plot(h[:,0], h[:,1], 'ko', label='Robot Body')
            
            # Current Look-ahead Point
            plt.plot(z_la[0], z_la[1], 'ro', label='Look-ahead Point')
            
            plt.plot(target_pos[0], target_pos[1], 'bx', markersize=10)
            plt.xlim(-2, 10); plt.ylim(-2, 10)
            plt.legend()
            plt.pause(0.01)

    plt.show()

if __name__ == "__main__":
    main()