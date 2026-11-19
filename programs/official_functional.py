def recipes():
    out = []
    lambdas = [0.1, 1.0, 10.0]
    bias_budgets = [0.0, 0.5]
    weight_modes = ['uniform', 'magnitude']
    idx = 0
    for lam in lambdas:
        for bb in bias_budgets:
            for wm in weight_modes:
                out.append({
                    'id': 'r%02d' % idx,
                    'lambda': float(lam),
                    'bias_budget': float(bb),
                    'weight_mode': wm
                })
                idx += 1
    return out


class NativeMLPEditor:
    def __init__(self, preactivation, w, bias, hidden_bias, seed):
        self.Z = np.array(preactivation, dtype=np.float64, copy=True)
        self.w = np.array(w, dtype=np.float64, copy=True)
        self.b = float(bias)
        self.beta = np.array(hidden_bias, dtype=np.float64, copy=True)
        self.seed = int(seed)
        self.d = int(self.w.shape[0])
        self.base = np.concatenate([self.w, self.beta])

    def predict(self, vector, rows=None):
        v = np.asarray(vector, dtype=np.float64)
        w_new = v[:self.d]
        beta_new = v[self.d:]
        if rows is None:
            Z = self.Z
        else:
            Z = self.Z[np.asarray(rows, dtype=np.int64)]
        delta = beta_new - self.beta
        H = np.maximum(Z + delta, 0.0)
        return H @ w_new + self.b

    def propose(self, rows, y, recipe):
        rows = np.asarray(rows, dtype=np.int64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        Z = self.Z[rows]
        n = Z.shape[0]
        d = self.d
        lam = float(recipe.get('lambda', 1.0))
        bias_budget = float(recipe.get('bias_budget', 0.0))
        weight_mode = recipe.get('weight_mode', 'uniform')

        H0 = np.maximum(Z, 0.0)
        base_score = H0 @ self.w + self.b
        r = y - base_score

        if weight_mode == 'magnitude':
            scale = np.mean(np.abs(y)) + 1e-12
            sw = np.abs(y) / scale
            sw = np.clip(sw, 0.05, 5.0)
        else:
            sw = np.ones(n, dtype=np.float64)

        A = H0
        AtA = A.T @ (A * sw[:, None])
        Atr = A.T @ (sw * r)
        reg = lam * np.eye(d, dtype=np.float64)
        try:
            dw = np.linalg.solve(AtA + reg, Atr)
        except np.linalg.LinAlgError:
            dw = np.linalg.lstsq(AtA + reg, Atr, rcond=None)[0]

        w_new = self.w + dw

        beta_new = self.beta.copy()
        if bias_budget > 0.0 and n > 0:
            active = (Z > 0.0).astype(np.float64)
            g = active @ w_new
            denom = float(np.sum(sw * g * g)) + lam
            num = float(np.sum(sw * r * g))
            shift = num / denom if denom > 0.0 else 0.0
            zscale = float(np.std(Z)) + 1e-12
            limit = bias_budget * zscale
            shift = float(np.clip(shift, -limit, limit))
            beta_new = self.beta + shift

        vec = np.concatenate([w_new, beta_new]).astype(np.float64)
        if not np.all(np.isfinite(vec)):
            vec = self.base.copy()
        return vec
