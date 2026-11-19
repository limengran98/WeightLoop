"""Validated NumPy interface to the recorded functional-MLP numerical rules."""
import numpy as np

def recipes():
    out = []
    for scope in ['output', 'multilayer']:
        for lam in [0.1, 1.0, 10.0]:
            for radius in [0.25, 1.0]:
                out.append({'id': scope + '_lambda' + str(lam) + '_radius' + str(radius), 'scope': scope, 'lambda_rel': lam, 'radius': radius})
    return out


class NativeMLPEditor:
    def __init__(self, preactivation, w, bias, hidden_bias, seed):
        self.z = np.array(preactivation, dtype=np.float64, copy=True)
        self.w = np.array(w, dtype=np.float64, copy=True)
        self.bias = float(bias)
        self.beta = np.array(hidden_bias, dtype=np.float64, copy=True)
        self.d = 128
        if self.z.ndim != 2 or self.z.shape[1] != self.d:
            raise ValueError('preactivation must have shape (observations, 128)')
        if self.w.shape != (self.d,) or self.beta.shape != (self.d,):
            raise ValueError('w and hidden_bias must each have shape (128,)')
        if not all(np.isfinite(v).all() for v in (self.z, self.w, self.beta)) or not np.isfinite(self.bias):
            raise ValueError('Model inputs and parameters must be finite')
        self.base = np.concatenate([self.w, self.beta])

        def scales(v):
            floor = max(float(np.sqrt(np.mean(v * v))), 0.01)
            return np.maximum(np.abs(v), floor)

        self.ws = scales(self.w)
        self.bs = scales(self.beta)
        rng = np.random.default_rng(seed)
        R = rng.choice([-1.0, 1.0], size=(128, 32))
        q, _ = np.linalg.qr(R)
        B = q * np.sqrt(128.0)
        self.D = self.bs[:, None] * B

    def predict(self, vector, rows=None):
        v = np.asarray(vector, dtype=np.float64)
        if v.shape != (2 * self.d,) or not np.isfinite(v).all():
            raise ValueError('vector must be a finite vector with 256 coordinates')
        w_new = v[:self.d]
        beta_new = v[self.d:]
        if rows is None:
            zr = self.z
        else:
            zr = self.z[rows]
        H = np.maximum(zr + (beta_new - self.beta), 0.0)
        return H @ w_new + self.bias

    def _ridge_step(self, J, residual, lam, radius, trust):
        n = J.shape[0]
        G = (J.T @ J) / max(n, 1)
        penalty = lam * max(float(np.trace(G)) / G.shape[0], 1e-12)
        alpha = np.linalg.solve(G + penalty * np.eye(G.shape[0]), (J.T @ residual) / max(n, 1))
        if trust == 'l2':
            norm = float(np.linalg.norm(alpha))
        else:
            norm = float(np.linalg.norm(alpha)) / np.sqrt(len(alpha))
        return alpha * min(1.0, radius / max(norm, 1e-30))

    def propose(self, rows, y, recipe):
        vector = self.base.copy()
        zr = self.z[rows]
        yv = np.asarray(y, dtype=np.float64)
        if zr.ndim != 2 or not len(zr) or yv.shape != (len(zr),) or not np.isfinite(yv).all():
            raise ValueError('Provide one finite outcome per selected feedback row')
        if recipe.get('scope') not in ('output', 'multilayer'):
            raise ValueError('recipe scope must be output or multilayer')
        baseline = np.maximum(zr, 0.0) @ self.w + self.bias
        lam = float(recipe['lambda_rel'])
        radius = float(recipe['radius'])
        if not np.isfinite(lam) or lam <= 0 or not np.isfinite(radius) or radius < 0:
            raise ValueError('Regularization must be positive and radius nonnegative, both finite')

        if recipe['scope'] == 'multilayer':
            eps = 0.1
            n = zr.shape[0]
            J = np.empty((n, 32), dtype=np.float64)
            for j in range(32):
                dcol = self.D[:, j]
                J[:, j] = (np.maximum(zr + eps * dcol, 0.0) @ self.w - np.maximum(zr - eps * dcol, 0.0) @ self.w) / (2.0 * eps)
            alpha = self._ridge_step(J, yv - baseline, lam, radius, 'l2')
            vector[self.d:] = vector[self.d:] + self.D @ alpha

        H = np.maximum(zr + (vector[self.d:] - self.beta), 0.0)
        residual = yv - (H @ self.w + self.bias)
        alpha2 = self._ridge_step(H * self.ws, residual, lam, radius, 'rms')
        vector[:self.d] = vector[:self.d] + self.ws * alpha2
        return vector
