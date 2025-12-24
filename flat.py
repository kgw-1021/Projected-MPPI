import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import time

# =========================================================
# 1. Differential Flatness B-Spline Engine
# =========================================================

class FlatnessBSpline:
    """
    미분 평탄성을 지원하는 B-Spline 엔진
    (JAX 호환성을 위해 if-else를 jnp.where로 변경함)
    """
    def __init__(self, n_cp, horizon, dt, degree=3):
        self.n_cp = n_cp
        self.horizon = horizon
        self.dt = dt
        self.degree = degree
        # Knots 설정 (Uniform B-Spline)
        self.knots = jnp.arange(n_cp + degree + 1, dtype=jnp.float32)
        
        # 시간 벡터 (0 ~ T)
        times = jnp.linspace(0, horizon * dt, horizon)
        
        # 기저함수 행렬 미리 계산
        self.M = self._precompute_basis(times, derivative=0)
        self.M_vel = self._precompute_basis(times, derivative=1)
        self.M_acc = self._precompute_basis(times, derivative=2)

    def _basis_function(self, i, k, t):
        """Cox-de Boor recursion (JAX Safe Version)"""
        # k는 Python 정수(recursion depth, static)이므로 Python if 사용 가능
        if k == 0:
            return jnp.where((t >= self.knots[i]) & (t < self.knots[i+1]), 1.0, 0.0)
        
        # i는 vmap을 통해 들어오는 Traced Array이므로, 
        # knots[i] 연산 결과도 Traced Array가 됨 -> Python if 사용 불가!
        
        # 1. 첫 번째 항 (Term 1)
        denom1 = self.knots[i+k] - self.knots[i]
        # 분모가 0일 경우 0으로 처리, 아닐 경우 공식 적용
        # 나눗셈 0 방지를 위해 안전한 분모(safe_denom) 생성
        safe_denom1 = jnp.where(denom1 == 0, 1.0, denom1)
        term1_val = ((t - self.knots[i]) / safe_denom1) * self._basis_function(i, k-1, t)
        term1 = jnp.where(denom1 == 0, 0.0, term1_val)
        
        # 2. 두 번째 항 (Term 2)
        denom2 = self.knots[i+k+1] - self.knots[i+1]
        safe_denom2 = jnp.where(denom2 == 0, 1.0, denom2)
        term2_val = ((self.knots[i+k+1] - t) / safe_denom2) * self._basis_function(i+1, k-1, t)
        term2 = jnp.where(denom2 == 0, 0.0, term2_val)
        
        return term1 + term2

    def _precompute_basis(self, times, derivative=0):
        """JAX 자동 미분을 이용해 기저함수 및 도함수 행렬 생성"""
        
        # t와 i를 받아서 스칼라 값을 리턴하는 래퍼 함수
        def basis_wrapper(t, i):
            return self._basis_function(i, self.degree, t)

        # 미분 차수에 따라 함수 변환 (autodiff)
        target_fn = basis_wrapper
        for _ in range(derivative):
            target_fn = jax.grad(target_fn, argnums=0) # time(t)에 대해 미분

        # 행렬 생성을 위한 이중 vmap
        # vmap(target_fn, (None, 0)) -> 시간 t 고정, 모든 제어점 i에 대해 계산 (Vector)
        # vmap(..., (0, None)) -> 모든 시간 t에 대해 계산 (Matrix)
        
        # 1. 먼저 i에 대해 병렬화 (Shape: [N_cp])
        vmap_i = jax.vmap(target_fn, in_axes=(None, 0))
        # 2. 그 다음 t에 대해 병렬화 (Shape: [Time, N_cp])
        vmap_all = jax.vmap(vmap_i, in_axes=(0, None))
        
        return vmap_all(times, jnp.arange(self.n_cp))

# =========================================================
# 2. Geometric MPPI Controller (Output Space)
# =========================================================

class FlatnessMPPI:
    def __init__(self, spline, Q, R_smooth, lam=1.0, n_samples=500):
        self.spline = spline
        self.Q = Q             # Goal Tracking Cost
        self.R_smooth = R_smooth # Smoothness Cost
        self.lam = lam         # Temperature
        self.n_samples = n_samples
        
        # 장애물 설정 (원형 장애물)
        self.obs_pos = jnp.array([[3.0, 3.0], [5.0, 5.0], [7.0, 2.0]])
        self.obs_r = 0.8

    @partial(jax.jit, static_argnums=(0,))
    def rollout_geometry(self, cps):
        """
        [핵심] 적분(scan) 없이 행렬 곱으로 궤적 생성
        cps: (N_samples, N_cp, 2) -> 제어점 좌표
        Returns: (N_samples, Horizon, 2) -> (x, y) 궤적
        """
        # Einsum: (Time, CP) @ (Sample, CP, Dim) -> (Sample, Time, Dim)
        return jnp.einsum('tc,ncd->ntd', self.spline.M, cps)

    @partial(jax.jit, static_argnums=(0,))
    def get_flatness_input(self, cps, t_idx=0):
        """
        미분 평탄성을 이용해 제어점(CP)에서 v, w 역산
        """
        # t_idx 시점의 속도, 가속도 벡터 계산
        # M_vel[t_idx]: (N_cp,)
        vel_vec = self.spline.M_vel[t_idx] @ cps  # [vx, vy]
        acc_vec = self.spline.M_acc[t_idx] @ cps  # [ax, ay]
        
        dx, dy = vel_vec[0], vel_vec[1]
        ddx, ddy = acc_vec[0], acc_vec[1]
        
        # Flatness Transform (Unicycle)
        v_cmd = jnp.sqrt(dx**2 + dy**2 + 1e-6)
        
        # 곡률 기반 각속도 계산: w = (x'y'' - y'x'') / (x'^2 + y'^2)
        w_cmd = (dx * ddy - dy * ddx) / (v_cmd**2 + 1e-6)
        
        return jnp.array([v_cmd, w_cmd])

    @partial(jax.jit, static_argnums=(0,))
    def cost_fn(self, trajs, cps, target):
        """비용 함수 (위치 공간에서 계산)"""
        # 1. Goal Cost
        diff = trajs - target[None, None, :2] # (N, T, 2)
        dist_sq = jnp.sum(diff**2, axis=-1)
        goal_cost = jnp.sum(dist_sq * self.Q, axis=1) # Time summation
        
        # 2. Smoothness Cost (제어점이 너무 튀지 않도록)
        # 2차 차분(Acceleration of CPs)을 최소화
        cp_acc = cps[:, 2:] - 2*cps[:, 1:-1] + cps[:, :-2]
        smooth_cost = jnp.sum(jnp.sum(cp_acc**2, axis=-1), axis=1) * self.R_smooth
        
        # 3. Obstacle Cost (Soft Constraint)
        # 모든 궤적 점에 대해 장애물 거리 검사
        dists = jnp.linalg.norm(trajs[:, :, None, :] - self.obs_pos[None, None, :, :], axis=-1)
        # dists: (Sample, Time, Obs)
        obs_cost = jnp.sum(jnp.sum(jnp.where(dists < self.obs_r, 10000.0, 0.0), axis=-1), axis=1)
        
        return goal_cost + smooth_cost + obs_cost

    @partial(jax.jit, static_argnums=(0,))
    def enforce_constraints(self, samples, mean_cps, fixed_start_cps):
        """
        Dynamically Feasible Projection
        1. Non-holonomic: 초기 방향 고정
        2. Velocity Limit: 인접 제어점 간 거리 제한
        3. Acceleration Limit: 2차 차분 제한 (곡률 및 가속도 제한)
        """
        # 1. Non-holonomic 제약 (초기 2개 점 고정)
        # P0, P1은 로봇의 현재 상태에 의해 결정됨 -> 무조건 준수
        samples = samples.at[:, 0, :].set(fixed_start_cps[0])
        samples = samples.at[:, 1, :].set(fixed_start_cps[1])
        
        # 동역학 파라미터
        V_MAX = 2.0  # m/s
        A_MAX = 1.0  # m/s^2 (가속도 및 구심력 제한)
        
        # B-Spline Knot 간격 (Uniform 가정)
        # 실제로는 (horizon * dt) / (n_cp - degree) 등으로 계산
        dt_knot = (self.spline.horizon * self.spline.dt) / (self.spline.n_cp - self.spline.degree)

        # ---------------------------------------------------------
        # 2. Velocity Constraints (Simple Projection)
        # ||P_i+1 - P_i|| <= V_max * dt_knot
        # 이 부분은 순차적(Sequential) 의존성이 있어서 완벽한 QP는 무겁습니다.
        # MPPI에서는 "Forward Filtering" 방식으로 근사해서 빠르게 풉니다.
        # ---------------------------------------------------------
        
        def vel_constraint_scan(carry, p_next):
            p_prev = carry
            # 목표 거리 벡터
            diff = p_next - p_prev
            dist = jnp.linalg.norm(diff) + 1e-6
            
            # 속도 한계를 넘으면 길이를 줄임 (Scaling)
            scale = jnp.minimum(1.0, (V_MAX * dt_knot) / dist)
            p_corrected = p_prev + diff * scale
            
            return p_corrected, p_corrected

        # 첫 번째 고정점(P0)부터 시작해서 연쇄적으로 거리 제한 적용
        # (Sample 차원은 vmap으로 처리)
        _, samples_vel_constrained = jax.lax.scan(
            lambda c, x: vel_constraint_scan(c, x), 
            samples[:, 0, :], # Init carry (P0)
            samples.transpose(1, 0, 2) # (N_cp, Sample, 2)로 뒤집어서 scan
        )
        # 다시 차원 복구: (Sample, N_cp, 2)
        samples = samples_vel_constrained.transpose(1, 0, 2)
        
        # ---------------------------------------------------------
        # 3. Acceleration Constraints
        # ||P_i+2 - 2P_i+1 + P_i|| <= A_max * dt_knot^2
        # 이 부분은 복잡하므로 보통 Cost Function에 강하게 넣거나,
        # 단순화하여 "곡률이 너무 급한 샘플을 기각(Reject)"하는 방식을 씁니다.
        # 여기서는 간단히 '급격한 꺾임'을 막기 위해 P_i를 평활화(Smoothing)합니다.
        # ---------------------------------------------------------
        
        # 간단한 3-point average smoothing (Low-pass filter 효과 -> 가속도 제한 효과)
        # P_i_new = 0.25*P_i-1 + 0.5*P_i + 0.25*P_i+1
        # 단, 고정점(0, 1)은 건드리지 않음
        
        p_smooth = 0.25 * samples[:, :-2, :] + 0.5 * samples[:, 1:-1, :] + 0.25 * samples[:, 2:, :]
        # 중간 부분만 교체
        samples = samples.at[:, 1:-1, :].set(p_smooth)
        
        # 다시 한 번 고정점 강제 (Smoothing으로 흐트러졌을 수 있으므로)
        samples = samples.at[:, 0, :].set(fixed_start_cps[0])
        samples = samples.at[:, 1, :].set(fixed_start_cps[1])
        
        return samples

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, mean_cps, z_curr, target):
        # 1. 기하학적 초기 조건 계산 (Flatness Constraint)
        # 로봇이 (x, y)에 있고 theta 방향을 보고 v 속도로 달리고 있다면,
        # P0 = (x, y)
        # P1 = (x + v*dt*cos(theta), y + v*dt*sin(theta))
        # 이어야만 부드럽게 출발 가능.
        
        x, y, theta = z_curr
        v_est = 1.0 # 현재 속도 추정치 (또는 이전 명령값). 여기서는 1.0 가정
        # P0, P1 계산
        p0 = jnp.array([x, y])
        p1 = jnp.array([x + v_est * 0.1 * jnp.cos(theta), 
                        y + v_est * 0.1 * jnp.sin(theta)]) # 0.1은 임의의 scale factor
        fixed_start = jnp.stack([p0, p1])
        
        # 2. Output Space Sampling
        # 제어점(CP) 자체에 노이즈를 더함 (위치 공간 샘플링)
        noise = jax.random.normal(key, shape=(self.n_samples, self.spline.n_cp, 2)) * 0.5
        samples = mean_cps + noise
        
        # 3. Projection (Constraints)
        samples = self.enforce_constraints(samples, mean_cps, fixed_start)
        
        # 4. Rollout (Matrix Multiplication only!)
        trajs = self.rollout_geometry(samples)
        
        # 5. Cost & Weighting
        costs = self.cost_fn(trajs, samples, target)
        min_cost = jnp.min(costs)
        weights = jax.nn.softmax(-(costs - min_cost) / self.lam)
        
        # 6. Update Mean
        delta = samples - mean_cps
        mean_cps_new = mean_cps + jnp.sum(weights[:, None, None] * delta, axis=0)
        
        # 평균 궤적도 제약조건 만족해야 함
        mean_cps_new = self.enforce_constraints(mean_cps_new[None], mean_cps, fixed_start)[0]
        
        return mean_cps_new, trajs, weights

# =========================================================
# 3. Main Simulation Loop
# =========================================================

def main():
    # 설정
    N_CP = 10
    HORIZON = 50
    DT = 0.1
    
    bspline = FlatnessBSpline(n_cp=N_CP, horizon=HORIZON, dt=DT)
    controller = FlatnessMPPI(bspline, Q=10.0, R_smooth=5.0)
    
    # 초기화
    z_curr = jnp.array([0.0, 0.0, jnp.pi/4]) # x, y, theta (45도 출발)
    target = jnp.array([8.0, 8.0])
    
    # 초기 제어점 (직선으로 초기화)
    mean_cps = jnp.linspace(z_curr[:2], target, N_CP)
    
    # 시각화 설정
    fig, ax = plt.subplots(figsize=(10, 10))
    key = jax.random.PRNGKey(0)
    
    traj_hist = []
    
    print("=== Simulation Start (Output-Space MPPI) ===")
    
    for t in range(5000):
        key, subkey = jax.random.split(key)
        
        # 1. MPPI Step (최적의 '모양'을 찾음)
        start_time = time.time()
        mean_cps, all_trajs, weights = controller.step(subkey, mean_cps, z_curr, target)
        
        # 2. Flatness Transform (모양 -> 제어 입력 v, w 변환)
        # t=0 시점의 제어 입력을 계산 (다음 스텝 실행용)
        # t_idx=0 은 현재, t_idx=1 은 바로 다음 순간. 
        # 안정성을 위해 t_idx=0의 곡률을 사용하거나 Lookahead를 쓸 수 있음.
        cmd = controller.get_flatness_input(mean_cps, t_idx=0)
        v_cmd, w_cmd = cmd[0], cmd[1]
        
        calc_time = (time.time() - start_time) * 1000
        
        # 3. Robot Dynamics Update (실제 로봇 움직임 시뮬레이션)
        # 실제로는 여기서 로봇에게 v_cmd, w_cmd를 보냄
        # 간단한 Euler 적분
        theta_new = z_curr[2] + w_cmd * DT
        x_new = z_curr[0] + v_cmd * jnp.cos(z_curr[2]) * DT
        y_new = z_curr[1] + v_cmd * jnp.sin(z_curr[2]) * DT
        z_curr = jnp.array([x_new, y_new, theta_new])
        traj_hist.append(z_curr[:2])
        
        # 4. Visualization
        if t % 5 == 0:
            ax.cla()
            # 장애물
            for i in range(len(controller.obs_pos)):
                circle = plt.Circle(controller.obs_pos[i], controller.obs_r, color='r', alpha=0.3)
                ax.add_patch(circle)
                
            # 샘플 궤적 (Top 10)
            top_idx = jnp.argsort(weights)[-10:]
            for idx in top_idx:
                ax.plot(all_trajs[idx, :, 0], all_trajs[idx, :, 1], 'g-', alpha=0.1)
                
            # 평균 궤적 (최적해)
            mean_traj = controller.rollout_geometry(mean_cps[None])[0]
            ax.plot(mean_traj[:, 0], mean_traj[:, 1], 'b-', linewidth=2, label='Planned')
            
            # 제어점 (Control Points)
            ax.plot(mean_cps[:, 0], mean_cps[:, 1], 'ro--', markersize=5, label='Control Points')
            
            # 이동 궤적
            hist_arr = np.array(traj_hist)
            ax.plot(hist_arr[:, 0], hist_arr[:, 1], 'k-', linewidth=3, label='History')
            
            # 로봇 현재 위치 및 방향
            ax.arrow(z_curr[0], z_curr[1], jnp.cos(z_curr[2]), jnp.sin(z_curr[2]), 
                     head_width=0.3, color='k')
            
            ax.plot(target[0], target[1], 'rx', markersize=10, label='Goal')
            
            ax.set_title(f"Step {t} | Calc: {calc_time:.2f}ms | v: {v_cmd:.2f}, w: {w_cmd:.2f}")
            ax.set_xlim(-1, 10)
            ax.set_ylim(-1, 10)
            ax.legend()
            ax.grid(True)
            plt.pause(0.01)
            
        # 목표 도달 확인
        if jnp.linalg.norm(z_curr[:2] - target) < 0.5:
            print("Goal Reached!")
            break
            
        # 제어점 Shift (Receding Horizon)
        # 다음 스텝을 위해 제어점을 앞으로 당기고 끝점은 유지/연장
        mean_cps = jnp.concatenate([mean_cps[1:], mean_cps[-1:]], axis=0)

    plt.show()

if __name__ == "__main__":
    main()