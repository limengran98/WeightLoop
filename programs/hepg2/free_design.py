def recipes():
    return [
        {'id': 'out_l1_s05', 'mode': 'output', 'lambda': 1.0, 'step': 0.5, 'max_norm': 1.0},
        {'id': 'out_l01_s1', 'mode': 'output', 'lambda': 0.1, 'step': 1.0, 'max_norm': 2.0},
        {'id': 'out_l10_s1', 'mode': 'output', 'lambda': 10.0, 'step': 1.0, 'max_norm': 0.5},
        {'id': 'out_l1_s1', 'mode': 'output', 'lambda': 1.0, 'step': 1.0, 'max_norm': 1.5},
        {'id': 'joint_pos', 'mode': 'joint', 'lambda': 0.5, 'step': 0.5, 'max_norm': 1.0, 'beta_scale': 0.1, 'beta_dir': 'ones'},
        {'id': 'joint_neg', 'mode': 'joint', 'lambda': 0.5, 'step': 0.5, 'max_norm': 1.0, 'beta_scale': -0.1, 'beta_dir': 'ones'},
        {'id': 'joint_wdir', 'mode': 'joint', 'lambda': 0.5, 'step': 0.5, 'max_norm': 1.0, 'beta_scale': 0.1, 'beta_dir': 'sign_w'},
    ]

class NativeMLPEditor:
    def __init__(self, preactivation, w, bias, hidden_bias, seed):
        self.preactivation = np.array(preactivation, dtype=np.float64, copy=True)
        self.w = np.array(w, dtype=np.float64, copy=True)
        self.bias = float(bias)
        self.hidden_bias = np.array(hidden_bias, dtype=np.float64, copy=True)
        self.seed = int(seed)
        self.base = np.concatenate([self.w, self.hidden_bias])
        self.d = self.w.shape[0]

    def predict(self, vector, rows=None):
        vector = np.asarray(vector, dtype=np.float64)
        w_new = vector[:self.d]
        beta_new = vector[self.d:]
        if rows is None:
            Z = self.preactivation
        else:
            Z = self.preactivation[rows]
        Z_adj = Z + (beta_new - self.hidden_bias)
        H = np.maximum(Z_adj, 0.0)
        return H @ w_new + self.bias

    def propose(self, rows, y, recipe):
        rows = np.asarray(rows, dtype=np.int64)
        y = np.asarray(y, dtype=np.float64)
        Z = self.preactivation[rows]
        n = Z.shape[0]
        d = self.d
        w = self.w
        beta = self.hidden_bias

        mode = recipe.get('mode', 'output')
        lam = float(recipe.get('lambda', 1.0))
        step = float(recipe.get('step', 1.0))
        max_norm = float(recipe.get('max_norm', 1.0))

        if mode == 'output':
            delta_beta = np.zeros(d, dtype=np.float64)
        else:
            beta_scale = float(recipe.get('beta_scale', 0.0))
            beta_dir = recipe.get('beta_dir', 'ones')
            if beta_dir == 'ones':
                direction = np.ones(d, dtype=np.float64)
            elif beta_dir == 'sign_w':
                direction = np.sign(w)
                if np.all(direction == 0):
                    direction = np.ones(d, dtype=np.float64)
            elif beta_dir == 'w':
                nw = np.linalg.norm(w)
                direction = w / nw if nw > 0 else np.ones(d, dtype=np.float64)
            else:
                direction = np.ones(d, dtype=np.float64)
            delta_beta = beta_scale * direction

        Z_adj = Z + delta_beta
        H = np.maximum(Z_adj, 0.0)
        s = H @ w + self.bias

        rhs = np.zeros(d, dtype=np.float64)
        Hmat = np.zeros((d, d), dtype=np.float64)

        for i in range(n):
            mask = y[i] > y
            if not np.any(mask):
                continue
            diff = H[i] - H
            sdiff = s[i] - s
            sdiff_clipped = np.clip(sdiff, -50.0, 50.0)
            p = 1.0 / (1.0 + np.exp(sdiff_clipped))
            p = p * mask
            hweight = p * (1.0 - p)
            rhs += np.sum(p[:, None] * diff, axis=0)
            Hmat += (diff * hweight[:, None]).T @ diff

        A = Hmat + lam * np.eye(d) + 1e-8 * np.eye(d)
        try:
            delta = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(A, rhs, rcond=None)[0]

        delta = step * delta
        norm = np.linalg.norm(delta)
        if norm > max_norm and norm > 0:
            delta = delta * (max_norm / norm)

        w_new = w + delta
        beta_new = beta + delta_beta
        return np.concatenate([w_new, beta_new]).astype(np.float64)
