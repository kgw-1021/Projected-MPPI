import jax
import jax.numpy as jnp
from functools import partial

class BatchedADMM:
    """
    Solves: min 1/2 x^T P x + q^T x  s.t.  l <= A x <= u
    Using simple ADMM algorithm optimized for JAX vmap.
    """
    def __init__(self, n_vars, rho=1.0, max_iter=20):
        self.n = n_vars
        self.rho = rho
        self.max_iter = max_iter

    @partial(jax.jit, static_argnums=(0,))
    def solve(self, P, q, A, l, u, init_x=None, init_z=None, init_y=None):
        """
        P: (n, n) - Positive definite matrix (Cost Hessian)
        q: (n,)   - Linear cost
        A: (m, n) - Constraint matrix
        l: (m,)   - Lower bounds
        u: (m,)   - Upper bounds
        """
        m = A.shape[0]
        
        # 1. Pre-factorization
        # K = P + rho * A.T @ A
        K = P + self.rho * (A.T @ A) + jnp.eye(self.n) * 1e-6
        
        # Cholesky Decomposition: K = L @ L.T
        # 역행렬을 직접 구하는 것보다 solve(L, ...)을 쓰는게 더 빠르고 안정적임
        L = jax.scipy.linalg.cholesky(K, lower=True)

        # 2. Initialization
        x = jnp.zeros(self.n) if init_x is None else init_x
        z = jnp.zeros(m)      if init_z is None else init_z
        y = jnp.zeros(m)      if init_y is None else init_y # Dual variable

        # 3. ADMM Loop (Fixed Iterations -> No Branching)
        def body_fn(i, val):
            x, z, y = val
            
            # --- x-update ---
            # Solve: (P + rho*A'A)x = rho*A'(z - u/rho) - q
            # rhs = rho * A.T @ (z - y) - q  (Standard ADMM form scaled y)
            # 여기서는 OSQP paper form: sigma=0, rho 사용
            
            rhs = self.rho * A.T @ (z - y / self.rho) - q
            
            # Forward/Backward Substitution using Cholesky factor L
            # solve Kx = rhs  =>  L L.T x = rhs
            t = jax.scipy.linalg.solve_triangular(L, rhs, lower=True)
            x_new = jax.scipy.linalg.solve_triangular(L.T, t, lower=False)
            
            # --- z-update (Projection) ---
            # z = clip(A*x + y/rho, l, u)
            Ax_hat = A @ x_new
            z_tilde = Ax_hat + y / self.rho
            z_new = jnp.clip(z_tilde, l, u)
            
            # --- y-update (Dual) ---
            y_new = y + self.rho * (Ax_hat - z_new) 
            
            return (x_new, z_new, y_new)

        # jax.lax.fori_loop is faster than python loop inside JIT
        x_final, z_final, y_final = jax.lax.fori_loop(0, self.max_iter, body_fn, (x, z, y))
        
        return x_final, (x_final, z_final, y_final)