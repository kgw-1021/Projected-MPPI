import jax
import jax.numpy as jnp
from functools import partial

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