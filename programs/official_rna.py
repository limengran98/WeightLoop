def _as_float(a):
    return np.asarray(a, dtype=np.float64)


def _ridge_solve(F, r, lam, stab=1e-8):
    # F: (m, d), r: (m,), returns delta (d,)
    # Ridge: minimize ||F delta - r||^2 + (lam*tr + stab) ||delta||^2
    # where tr = trace(F^T F)/d. Choose primal or dual by smaller dimension.
    m, d = F.shape
    tr = float(np.sum(F * F)) / max(d, 1)
    reg = lam * tr + stab
    if m <= d:
        # Dual: delta = F^T (F F^T + reg I_m)^-1 r
        A = F @ F.T
        A = A + reg * np.eye(m)
        try:
            y = np.linalg.solve(A, r)
        except np.linalg.LinAlgError:
            y = np.linalg.lstsq(A, r, rcond=None)[0]
        return F.T @ y
    else:
        # Primal: delta = (F^T F + reg I_d)^-1 F^T r
        A = F.T @ F
        A = A + reg * np.eye(d)
        b = F.T @ r
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]


def propose_linear(features, baseline, target, weights, recipe, shared):
    F = _as_float(features)
    base = _as_float(baseline)
    tgt = _as_float(target)
    W = _as_float(weights)
    lam = float(recipe.get('ridge', 1.0))
    n, G, D = F.shape
    r = (tgt - base).reshape(n, G)
    if shared:
        Fm = F.reshape(n * G, D)
        rm = r.reshape(n * G)
        delta = _ridge_solve(Fm, rm, lam)
        return W + delta
    else:
        newW = W.copy()
        for g in range(G):
            Fg = F[:, g, :]
            rg = r[:, g]
            dg = _ridge_solve(Fg, rg, lam)
            newW[g] = W[g] + dg
        return newW


def _pack_scfoundation(newW):
    G = newW.shape[0]
    return np.concatenate([newW[:, :-1].reshape(-1), newW[:, -1]])


def propose_rna(base_vector, hidden_directions, native_state, target, recipe):
    base_vec = _as_float(base_vector).copy()
    H = _as_float(hidden_directions)
    P, K = H.shape
    lam = float(recipe.get('ridge', 1.0))
    hstep = float(recipe.get('hidden_step', 0.05))
    hlam = float(recipe.get('hidden_ridge', 1.0))
    use_hidden = bool(recipe.get('use_hidden', True))
    max_probe = int(recipe.get('max_probe', 16))
    Kp = min(K, max_probe)
    tgt = _as_float(target)

    def residual_at(vec):
        st = native_state(vec)
        F = _as_float(st['features'])
        b = _as_float(st['baseline'])
        w = _as_float(st['weights'])
        sh = bool(st['shared'])
        # baseline is the complete current native prediction at this vector;
        # with newweights == current weights the prediction equals baseline.
        if sh:
            pred = b + F @ (w - w)
        else:
            pred = b + np.einsum('ngd,gd->ng', F, (w - w))
        return float(np.mean((pred - tgt) ** 2)), st

    best_vec = base_vec
    base_err, base_state = residual_at(base_vec)

    if use_hidden and Kp > 0:
        slopes = np.zeros(Kp)
        for k in range(Kp):
            d = H[:, k]
            vp = base_vec + hstep * d
            vm = base_vec - hstep * d
            ep, _ = residual_at(vp)
            em, _ = residual_at(vm)
            slopes[k] = (ep - em) / (2.0 * hstep)
        A = np.outer(slopes, slopes) + hlam * np.eye(Kp)
        bvec = -slopes * base_err
        try:
            alpha = np.linalg.solve(A, bvec)
        except np.linalg.LinAlgError:
            alpha = np.linalg.lstsq(A, bvec, rcond=None)[0]
        norm = float(np.linalg.norm(alpha))
        max_norm = float(recipe.get('hidden_max_norm', 1.0))
        if norm > max_norm and norm > 0:
            alpha = alpha * (max_norm / norm)
        cand = base_vec.copy()
        for k in range(Kp):
            cand = cand + alpha[k] * H[:, k]
        err_c, st_c = residual_at(cand)
        if np.isfinite(err_c) and err_c < base_err:
            best_vec = cand
            base_state = st_c

    st = base_state
    F = _as_float(st['features'])
    b = _as_float(st['baseline'])
    w = _as_float(st['weights'])
    sh = bool(st['shared'])
    out_w = int(st['output_width'])

    newW = propose_linear(F, b, target, w, recipe, sh)

    if sh:
        new_vec = best_vec.copy()
        new_vec[:out_w] = newW
        return new_vec
    else:
        packed = _pack_scfoundation(newW)
        new_vec = best_vec.copy()
        new_vec[:out_w] = packed
        return new_vec


def recipes(model):
    out = []
    if model == 'scgpt':
        grid = [
            (0.1, False, 0.0, 0.0, 0),
            (1.0, False, 0.0, 0.0, 0),
            (10.0, False, 0.0, 0.0, 0),
            (100.0, False, 0.0, 0.0, 0),
            (1.0, True, 0.05, 1.0, 8),
            (10.0, True, 0.05, 1.0, 8),
            (1.0, True, 0.1, 1.0, 16),
            (10.0, True, 0.1, 1.0, 16),
        ]
        for i, (lam, uh, hs, hl, mp) in enumerate(grid):
            out.append({
                'id': 'scgpt_r%d' % i,
                'model': 'scgpt',
                'ridge': lam,
                'use_hidden': uh,
                'hidden_step': hs,
                'hidden_ridge': hl,
                'hidden_max_norm': 1.0,
                'max_probe': mp,
            })
    elif model == 'scfoundation':
        grid = [
            (0.1, False, 0.0, 0.0, 0),
            (1.0, False, 0.0, 0.0, 0),
            (10.0, False, 0.0, 0.0, 0),
            (100.0, False, 0.0, 0.0, 0),
            (1.0, True, 0.05, 1.0, 8),
            (10.0, True, 0.05, 1.0, 8),
            (1.0, True, 0.1, 1.0, 16),
            (10.0, True, 0.1, 1.0, 16),
        ]
        for i, (lam, uh, hs, hl, mp) in enumerate(grid):
            out.append({
                'id': 'scfoundation_r%d' % i,
                'model': 'scfoundation',
                'ridge': lam,
                'use_hidden': uh,
                'hidden_step': hs,
                'hidden_ridge': hl,
                'hidden_max_norm': 1.0,
                'max_probe': mp,
            })
    return out
