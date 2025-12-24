import jax
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
from functools import partial
import numpy as np

# =========================================================
# 1. B-Spline Basis & Koopman Dynamics
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
        m = self.n_cp + self.degree + 1
        knots = jnp.concatenate([
            jnp.zeros(self.degree), 
            jnp.linspace(0, 1, m - 2 * self.degree), 
            jnp.ones(self.degree + 1)
        ])
        t_vals = jnp.linspace(0, 1, self.horizon)
        def compute_row(t):
            return jnp.array([self._basis_function(i, self.degree, t, knots) for i in range(self.n_cp)])
        return jax.vmap(compute_row)(t_vals)

    @partial(jax.jit, static_argnums=(0,))
    def get_sequence(self, coeffs):
        # (Horizon, N_CP) @ (N_CP, 2) -> (Horizon, 2)
        return self.basis_matrix @ coeffs

@jax.jit
def lift_state(state_std):
    x, y, theta = state_std
    return jnp.array([x, y, jnp.cos(theta), jnp.sin(theta)])

@jax.jit
def koopman_step(z, u, dt):
    x, y, c, s = z
    v, w = u
    next_x = x + v * c * dt
    next_y = y + v * s * dt
    next_c = c - s * w * dt
    next_s = s + c * w * dt
    norm = jnp.sqrt(next_c**2 + next_s**2 + 1e-10)
    return jnp.array([next_x, next_y, next_c/norm, next_s/norm])

# =========================================================
# 2. QP Projector with Limits & Warm Start (Fixed)
# =========================================================

class KoopmanQPProjector:
    def __init__(self, horizon, n_cp, dt, bspline_gen):
        self.H = horizon
        self.N_cp = n_cp
        self.dt = dt
        self.bspline = bspline_gen
        self.qp = jaxopt.OSQP()
        
        # 물리적 제어 한계 설정 (v: 0~2.0 m/s, w: -1.5~1.5 rad/s)
        self.u_min = jnp.array([0.0, -1.5]) 
        self.u_max = jnp.array([2.0, 1.5])

    @partial(jax.jit, static_argnums=(0,))
    def rollout_fn(self, coeffs, z0):
        u_seq = self.bspline.get_sequence(coeffs)
        def step_fn(carry, u):
            z_next = koopman_step(carry, u, self.dt)
            return z_next, z_next[:2]
        _, pos_traj = jax.lax.scan(step_fn, z0, u_seq)
        return pos_traj

    @partial(jax.jit, static_argnums=(0,))
    def jac_fn(self, coeffs, z0):
        return jax.jacfwd(self.rollout_fn, argnums=0)(coeffs, z0)

    @partial(jax.jit, static_argnums=(0,))
    def project_single_sample(self, coeffs_noisy, z0, obs_pos, obs_r, solver_init_state=None):
        """
        [Fix] 양방향 제약조건(Box)을 단방향(G x <= h) 형태로 변환하여 해결
        """
        # --- A. Setup Obstacle Constraint ---
        p_traj = self.rollout_fn(coeffs_noisy, z0)
        J_tensor = self.jac_fn(coeffs_noisy, z0)
        
        diff = p_traj - obs_pos
        dist_sq = jnp.sum(diff**2, axis=1)
        min_idx = jnp.argmin(dist_sq)
        
        p_crit = p_traj[min_idx]
        d_crit = diff[min_idx]
        dist_val = jnp.sqrt(jnp.sum(d_crit**2)) + 1e-6
        normal = d_crit / dist_val
        
        # 1. Obstacle Constraint: A_obs * delta <= b_obs
        J_flat = J_tensor[min_idx].reshape(2, -1)
        A_obs = - (normal @ J_flat).reshape(1, -1)
        
        safe_margin = 0.6 
        req_dist = obs_r + safe_margin
        b_obs = jnp.array([-(req_dist - dist_val)])
        
        # --- B. Setup Input Limit Constraints (Converted to G x <= h) ---
        coeffs_flat = coeffs_noisy.reshape(-1)
        dim_var = self.N_cp * 2
        
        # 타일링하여 상/하한 생성
        u_min_tiled = jnp.tile(self.u_min, self.N_cp)
        u_max_tiled = jnp.tile(self.u_max, self.N_cp)
        
        # 원래 식: lower <= c + delta <= upper
        # 변환 1 (Upper): delta <= upper - c
        upper_bound = u_max_tiled - coeffs_flat
        
        # 변환 2 (Lower): delta >= lower - c  ->  -delta <= -(lower - c)
        lower_bound = u_min_tiled - coeffs_flat
        neg_lower_bound = -lower_bound
        
        # --- C. Stack Constraints (G, h) ---
        # G = [ A_obs ]   h = [ b_obs      ]
        #     [   I   ]       [ upper_bound]
        #     [  -I   ]       [ neg_lower  ]
        
        G = jnp.vstack([
            A_obs,                  # Obstacle
            jnp.eye(dim_var),       # Limit (Upper)
            -jnp.eye(dim_var)       # Limit (Lower)
        ])
        
        h = jnp.concatenate([
            b_obs,
            upper_bound,
            neg_lower_bound
        ])
        
        # --- D. Objective ---
        P = jnp.eye(dim_var)
        q = jnp.zeros(dim_var)
        
        # --- E. Solve ---
        # params_ineq에는 이제 (G, h) 2개만 들어갑니다.
        init_params = solver_init_state if solver_init_state is not None else None
        
        sol = self.qp.run(init_params=init_params, params_obj=(P, q), params_ineq=(G, h))
        
        delta = sol.params.primal.reshape(self.N_cp, 2)
        return coeffs_noisy + delta, sol.params

# =========================================================
# 3. Koopman MPPI (Updated Costs & Logic)
# =========================================================

class KoopmanMPPI:
    def __init__(self, horizon, n_cp, dt, n_samples, temperature, bspline_gen):
        self.H = horizon
        self.N_cp = n_cp
        self.dt = dt
        self.K = n_samples
        self.lambda_ = temperature
        self.projector = KoopmanQPProjector(horizon, n_cp, dt, bspline_gen)

    @partial(jax.jit, static_argnums=(0,))
    def compute_cost(self, coeffs, z0, target_pos, obs_pos, obs_r):
        traj = self.projector.rollout_fn(coeffs, z0)
        
        # 1. Target Tracking Cost
        term_err = jnp.sum((traj[-1] - target_pos)**2)
        stage_err = jnp.sum(jnp.sum((traj - target_pos)**2, axis=1))
        
        # 2. Obstacle Cost (Soft Constraint)
        # QP가 해결해주지만, 추가적인 안전장치 및 Gradient 제공
        dist = jnp.sqrt(jnp.sum((traj - obs_pos)**2, axis=1))
        # 안전거리 안쪽으로 들어오면 지수함수로 비용 폭발
        obs_cost = jnp.sum(jnp.exp(-5.0 * (dist - obs_r - 0.2)))
        
        # 3. Smoothness Cost (Jerk minimization)
        # 제어점 간의 변화량이 클수록 페널티
        # (N_CP-1, 2)
        diffs = jnp.diff(coeffs, axis=0) 
        smoothness_cost = jnp.sum(diffs**2)
        
        total_cost = (
            20.0 * term_err + 
            0.5 * stage_err + 
            10.0 * obs_cost + 
            5.0 * smoothness_cost # 가중치 조절
        )
        return total_cost

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, mean_coeffs, z0, target_pos, obs_pos, obs_r, prev_solver_params):
        # 1. Sampling
        noise_v = jax.random.normal(key, (self.K, self.N_cp)) * 0.5
        noise_w = jax.random.normal(key, (self.K, self.N_cp)) * 0.5
        noise = jnp.stack([noise_v, noise_w], axis=2)
        raw_samples = mean_coeffs + noise
        
        # 2. Project ALL Samples with Limits & Obstacles
        # Warm Start: 이전 스텝의 Mean에 대한 솔버 상태를 모든 샘플의 초기값으로 사용 (근사적 Warm Start)
        # (샘플마다 개별 상태를 유지하는 건 메모리 낭비가 심하므로 Mean의 해를 공유)
        
        # vmap: project_single_sample(coeffs, z0, obs, r, solver_state)
        project_fn = jax.vmap(self.projector.project_single_sample, in_axes=(0, None, None, None, None))
        
        # prev_solver_params가 None이면 None이 전달됨
        safe_samples, solver_states = project_fn(raw_samples, z0, obs_pos, obs_r, prev_solver_params)
        
        # 3. Cost Evaluation
        cost_fn = jax.vmap(self.compute_cost, in_axes=(0, None, None, None, None))
        costs = cost_fn(safe_samples, z0, target_pos, obs_pos, obs_r)
        
        # 4. Weight Update
        min_cost = jnp.min(costs)
        weights = jax.nn.softmax(-(costs - min_cost) / self.lambda_)
        
        weights_expanded = weights[:, None, None]
        new_mean = jnp.sum(weights_expanded * safe_samples, axis=0)
        
        alpha = 0.9 # Smooth update
        final_mean = alpha * new_mean + (1 - alpha) * mean_coeffs
        
        # 다음 스텝 Warm Start를 위해, 가장 비용이 낮은(Best) 샘플의 솔버 상태를 반환
        best_idx = jnp.argmin(costs)
        next_solver_params = jax.tree.map(lambda x: x[best_idx], solver_states)
        
        return final_mean, safe_samples, weights, next_solver_params

# =========================================================
# 4. Main Simulation Loop
# =========================================================
def run_advanced_mppi():
    DT = 0.1
    HORIZON = 30
    N_CP = 10
    N_SAMPLES = 50 
    TEMP = 0.5
    
    bspline_gen = BSplineBasis(N_CP, HORIZON)
    mppi = KoopmanMPPI(HORIZON, N_CP, DT, N_SAMPLES, TEMP, bspline_gen)
    
    # Initial Conditions
    start_pose = jnp.array([0.0, 0.0, 0.0])
    z_curr = lift_state(start_pose)
    target_pos = jnp.array([5.0, 0.5])
    obs_pos = jnp.array([2.5, -0.1]) # 경로상에 있는 장애물
    obs_r = 1.0
    
    # Initial Guess (Straight forward)
    # v=1.0, w=0.0
    mean_coeffs = jnp.ones((N_CP, 2)) * jnp.array([1.0, 0.0])
    
    # Warm Start용 변수 초기화
    solver_params = None 
    
    key = jax.random.PRNGKey(0)
    traj_hist = [z_curr[:2]]
    
    print("Running Advanced Koopman MPPI (Smooth + Limits + WarmStart)...")
    
    plt.figure(figsize=(10, 6))
    
    for t in range(500):
        key, subkey = jax.random.split(key)
        
        # MPPI Step with Warm Start passing
        mean_coeffs, safe_samples, weights, solver_params = mppi.step(
            subkey, mean_coeffs, z_curr, target_pos, obs_pos, obs_r, solver_params
        )
        
        # Execute Control (First step of spline)
        u_seq = bspline_gen.get_sequence(mean_coeffs)
        u_curr = u_seq[0]
        
        # Physics Update
        z_curr = koopman_step(z_curr, u_curr, DT)
        traj_hist.append(z_curr[:2])
        
        # --- Visualization ---
        if t % 1 == 0:
            plt.clf()
            
            # Obstacle & Margin
            plt.gca().add_patch(plt.Circle(obs_pos, obs_r, color='r', alpha=0.5))
            plt.gca().add_patch(plt.Circle(obs_pos, obs_r + 0.6, color='r', fill=False, linestyle=':', label='QP Margin'))
            
            plt.plot(target_pos[0], target_pos[1], 'bx', markersize=10, label='Target')
            
            # Samples Visualization
            rollout_viz = jax.jit(jax.vmap(mppi.projector.rollout_fn, in_axes=(0, None)))
            top_idx = jnp.argsort(weights)[-20:] # Best 20 only
            top_samples = safe_samples[top_idx]
            top_trajs = rollout_viz(top_samples, z_curr)
            
            for k in range(len(top_samples)):
                alp = 0.2 + 0.8 * (k / len(top_samples))
                plt.plot(top_trajs[k, :, 0], top_trajs[k, :, 1], 'g-', alpha=alp)
            
            # Mean Trajectory
            mean_traj = mppi.projector.rollout_fn(mean_coeffs, z_curr)
            plt.plot(mean_traj[:, 0], mean_traj[:, 1], 'b-', linewidth=3, label='Mean Control')
            
            # History
            hist = np.array(traj_hist)
            plt.plot(hist[:, 0], hist[:, 1], 'k--', label='Driven Path')
            plt.plot(z_curr[0], z_curr[1], 'ko')
            
            plt.title(f"Step {t} | V:{u_curr[0]:.2f}, W:{u_curr[1]:.2f} (Limits Checked)")
            plt.axis('equal')
            plt.xlim(-4, 9)
            plt.ylim(-5, 5)
            plt.grid(True)
            plt.legend()
            plt.pause(0.01)

        if jnp.linalg.norm(z_curr[:2] - target_pos) < 0.1:
            print("Goal Reached!")
            break
            
    plt.show()

if __name__ == "__main__":
    run_advanced_mppi()