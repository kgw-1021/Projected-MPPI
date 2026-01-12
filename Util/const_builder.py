import jax
import jax.numpy as jnp
from functools import partial

# ==========================================
# 1. Vectorized Projection Primitives
# ==========================================
# JAX의 vmap을 활용하기 위해 단일 연산 함수들을 먼저 정의합니다.

@jax.jit
def _proj_box_single(z, l, u):
    return jnp.clip(z, l, u)

@jax.jit
def _proj_ball_single(z, c, r):
    diff = z - c
    norm = jnp.linalg.norm(diff)
    scale = jnp.minimum(1.0, r / (norm + 1e-6))
    return c + diff * scale

@jax.jit
def _proj_avoid_single(z, c, r):
    """장애물 회피: 구 내부이면 표면으로 밀어내기"""
    diff = z - c
    norm = jnp.linalg.norm(diff)
    # 내부에 있을 때(norm < r)만 밖으로 투영
    scale = jnp.maximum(1.0, r / (norm + 1e-6))
    return c + diff * scale

@jax.jit
def _proj_soc_single(z):
    """Second Order Cone"""
    t, x = z[0], z[1:]
    x_norm = jnp.linalg.norm(x)
    
    # Branchless logic for JAX
    # Case 1: Inside
    res_inside = z
    # Case 2: Below cone -> 0
    res_zero = jnp.zeros_like(z)
    # Case 3: Boundary
    alpha = (t + x_norm) / 2.0
    new_x = x * (alpha / (x_norm + 1e-6))
    res_bound = jnp.concatenate([jnp.array([alpha]), new_x])

    pred = jnp.where(x_norm <= t, 0, jnp.where(x_norm <= -t, 1, 2))
    return jax.lax.switch(pred, [lambda _: res_inside, lambda _: res_zero, lambda _: res_bound], None)

@jax.jit
def _proj_annulus_single(z, c, r_inner, r_outer):
    diff = z - c
    norm = jnp.linalg.norm(diff)
    # Annulus projection logic
    scale = jnp.where(
        norm < r_inner,
        r_inner / (norm + 1e-6),
        jnp.where(
            norm > r_outer,
            r_outer / (norm + 1e-6),
            1.0
        )
    )
    return c + diff * scale


# vmap을 사용하여 "한 방에" 처리하는 함수들 생성
# z: (Batch, Dim), c: (Batch, Dim), r: (Batch, 1)
batch_proj_box   = jax.vmap(_proj_box_single, in_axes=(0, 0, 0))
batch_proj_ball  = jax.vmap(_proj_ball_single, in_axes=(0, 0, 0))
batch_proj_avoid = jax.vmap(_proj_avoid_single, in_axes=(0, 0, 0))
batch_proj_soc   = jax.vmap(_proj_soc_single, in_axes=(0,))
batch_proj_annulus = jax.vmap(_proj_annulus_single, in_axes=(0, 0, 0, 0))

# ==========================================
# 2. Efficient Constraint Builder
# ==========================================

class ConstraintBuilder:
    def __init__(self):
        # 그룹별로 데이터를 모읍니다.
        self.groups = {
            'box':   {'A': [], 'l': [], 'u': [], 'dim': 0},
            'ball':  {'A': [], 'c': [], 'r': [], 'dim': 0},
            'avoid': {'A': [], 'c': [], 'r': [], 'dim': 0},
            'soc':   {'A': [], 'dim': 0},
            'annulus': {'A': [], 'c': [], 'r_inner': [], 'r_outer': [], 'dim': 0}
        }
        self.execution_order = [] # 투영 순서 (그룹 단위)

    def add_box(self, A, l, u):
        """l, u can be scalars or vectors matching A.shape[0]"""
        m = A.shape[0]
        # Broadcast scalar bounds to vector
        if jnp.ndim(l) == 0: l = jnp.full(m, l)
        if jnp.ndim(u) == 0: u = jnp.full(m, u)
        
        self.groups['box']['A'].append(A)
        self.groups['box']['l'].append(l)
        self.groups['box']['u'].append(u)
        self.groups['box']['dim'] = m # Assume consistent dim for simplicity

    def add_ball(self, A, center, radius):
        m = A.shape[0]
        self.groups['ball']['A'].append(A)
        self.groups['ball']['c'].append(center)
        self.groups['ball']['r'].append(radius)
        self.groups['ball']['dim'] = m

    def add_avoid(self, A, center, radius):
        m = A.shape[0]
        self.groups['avoid']['A'].append(A)
        self.groups['avoid']['c'].append(center)
        self.groups['avoid']['r'].append(radius)
        self.groups['avoid']['dim'] = m

    def add_soc(self, A):
        m = A.shape[0]
        self.groups['soc']['A'].append(A)
        self.groups['soc']['dim'] = m

    def add_annulus(self, A, center, r_inner, r_outer):
        m = A.shape[0]
        self.groups['annulus']['A'].append(A)
        self.groups['annulus']['c'].append(center)
        self.groups['annulus']['r_inner'].append(r_inner)
        self.groups['annulus']['r_outer'].append(r_outer)
        self.groups['annulus']['dim'] = m

    def build_system(self):
        """
        제약조건을 그룹별로 병합(Stacking)하여 컴파일 가능한 함수를 생성합니다.
        """
        final_A_list = []
        
        # 각 그룹별로 처리할 데이터(파라미터)를 준비
        # Group Processors
        processors = []
        
        current_idx = 0
        
        # 1. Box Constraints
        if self.groups['box']['A']:
            A_grp = jnp.vstack(self.groups['box']['A'])
            l_grp = jnp.concatenate(self.groups['box']['l'])
            u_grp = jnp.concatenate(self.groups['box']['u'])
            
            # Box는 차원이 1이므로 단순 reshape
            # z_shape: (Total_Rows, ) -> reshape -> (Total_Rows, 1) to match vmap? 
            # Actually box projection is element-wise, so it's simple.
            
            m_total = A_grp.shape[0]
            final_A_list.append(A_grp)
            
            def proc_box(z_flat):
                return batch_proj_box(z_flat.reshape(-1, 1), l_grp.reshape(-1,1), u_grp.reshape(-1,1)).flatten()
            
            processors.append((current_idx, m_total, proc_box))
            current_idx += m_total

        # 2. Avoid Constraints (Obstacles)
        if self.groups['avoid']['A']:
            A_grp = jnp.vstack(self.groups['avoid']['A'])
            c_grp = jnp.vstack(self.groups['avoid']['c']) # (N_obs, 3)
            r_grp = jnp.array(self.groups['avoid']['r'])  # (N_obs,)
            
            dim = self.groups['avoid']['dim'] # e.g., 3
            num_constraints = len(self.groups['avoid']['A'])
            m_total = num_constraints * dim
            
            final_A_list.append(A_grp)
            
            def proc_avoid(z_segment):
                # z_segment는 (N*3, ) 형태
                # 이를 (N, 3)으로 reshape 하여 배치 처리
                z_reshaped = z_segment.reshape(num_constraints, dim)
                z_proj = batch_proj_avoid(z_reshaped, c_grp, r_grp)
                return z_proj.flatten()
            
            processors.append((current_idx, m_total, proc_avoid))
            current_idx += m_total

        # 3. SOC Constraints
        if self.groups['soc']['A']:
            A_grp = jnp.vstack(self.groups['soc']['A'])
            dim = self.groups['soc']['dim']
            num_constraints = len(self.groups['soc']['A'])
            m_total = num_constraints * dim
            
            final_A_list.append(A_grp)
            
            def proc_soc(z_segment):
                z_reshaped = z_segment.reshape(num_constraints, dim)
                z_proj = batch_proj_soc(z_reshaped)
                return z_proj.flatten()
            
            processors.append((current_idx, m_total, proc_soc))
            current_idx += m_total
            
        # 4. Ball Constraints
        if self.groups['ball']['A']:
            A_grp = jnp.vstack(self.groups['ball']['A'])
            c_grp = jnp.vstack(self.groups['ball']['c']) # (N_ball, 3)
            r_grp = jnp.array(self.groups['ball']['r'])  # (N_ball,)
            
            dim = self.groups['ball']['dim'] # e.g., 3
            num_constraints = len(self.groups['ball']['A'])
            m_total = num_constraints * dim
            
            final_A_list.append(A_grp)
            
            def proc_ball(z_segment):
                z_reshaped = z_segment.reshape(num_constraints, dim)
                z_proj = batch_proj_ball(z_reshaped, c_grp, r_grp)
                return z_proj.flatten()
            
            processors.append((current_idx, m_total, proc_ball))
            current_idx += m_total

        # 5. Annulus Constraints
        if self.groups['annulus']['A']:
            A_grp = jnp.vstack(self.groups['annulus']['A'])
            c_grp = jnp.vstack(self.groups['annulus']['c']) # (N_annulus, 3)
            r_inner_grp = jnp.array(self.groups['annulus']['r_inner'])  # (N_annulus,)
            r_outer_grp = jnp.array(self.groups['annulus']['r_outer'])  # (N_annulus,)
            
            dim = self.groups['annulus']['dim'] # e.g., 3
            num_constraints = len(self.groups['annulus']['A'])
            m_total = num_constraints * dim
            
            final_A_list.append(A_grp)
            
            def proc_annulus(z_segment):
                z_reshaped = z_segment.reshape(num_constraints, dim)
                results = []
                for i in range(num_constraints):
                    diff = z_reshaped[i] - c_grp[i]
                    norm = jnp.linalg.norm(diff)
                    # Annulus projection logic
                    scale = jnp.where(
                        norm < r_inner_grp[i],
                        r_inner_grp[i] / (norm + 1e-6),
                        jnp.where(
                            norm > r_outer_grp[i],
                            r_outer_grp[i] / (norm + 1e-6),
                            1.0
                        )
                    )
                    proj_point = c_grp[i] + diff * scale
                    results.append(proj_point)
                return jnp.vstack(results).flatten()
            
            processors.append((current_idx, m_total, proc_annulus))
            current_idx += m_total

        # Final Assembly
        if not final_A_list:
            raise ValueError("No constraints defined.")
            
        A_total = jnp.vstack(final_A_list)
        
        # Closure for the global projection function
        def global_proj_func(z):
            results = []
            # 각 프로세서가 자신이 담당한 z의 구간만 잘라서 처리
            for start, length, proc in processors:
                z_slice = jax.lax.dynamic_slice(z, (start,), (length,))
                results.append(proc(z_slice))
            return jnp.concatenate(results)

        return A_total, global_proj_func