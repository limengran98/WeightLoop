def recipes(model):
    if model == 'ridge':
        return [
            {'id': 'ridge_l1e-2_b1', 'model': 'ridge', 'lam': 1e-2, 'bound': 1.0},
            {'id': 'ridge_l1e-1_b1', 'model': 'ridge', 'lam': 1e-1, 'bound': 1.0},
            {'id': 'ridge_l1_b2', 'model': 'ridge', 'lam': 1.0, 'bound': 2.0},
            {'id': 'ridge_l10_b2', 'model': 'ridge', 'lam': 10.0, 'bound': 2.0},
        ]
    if model == 'xgboost':
        return [
            {'id': 'xgb_k1_c1', 'model': 'xgboost', 'kappa': 1.0, 'clip': 1.0},
            {'id': 'xgb_k5_c1', 'model': 'xgboost', 'kappa': 5.0, 'clip': 1.0},
            {'id': 'xgb_k10_c2', 'model': 'xgboost', 'kappa': 10.0, 'clip': 2.0},
        ]
    return []


def _as_float_array(x):
    return np.array(x, dtype=float, copy=True)


def propose_linear(features, baseline, target, weights, recipe, shared):
    X = _as_float_array(features)
    b = _as_float_array(baseline)
    y = _as_float_array(target)
    w = _as_float_array(weights)
    lam = float(recipe.get('lam', 1.0))
    bound = float(recipe.get('bound', 1.0))
    if lam < 0.0:
        lam = 0.0
    if bound < 0.0:
        bound = 0.0

    if shared:
        n, G, D = X.shape
        Xf = X.reshape(n * G, D)
        rf = (y - b).reshape(n * G)
        XtX = Xf.T @ Xf
        Xtr = Xf.T @ rf
        A = XtX + lam * np.eye(D)
        diag = np.diag(A).copy()
        jitter = 1e-8 * (1.0 + float(np.max(np.abs(diag))))
        A = A + jitter * np.eye(D)
        try:
            delta = np.linalg.solve(A, Xtr)
        except Exception:
            denom = np.diag(XtX) + lam + jitter
            denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
            delta = Xtr / denom
        nrm = float(np.sqrt(np.sum(delta * delta)))
        if nrm > bound and nrm > 0.0:
            delta = delta * (bound / nrm)
        neww = w + delta
        neww = np.where(np.isfinite(neww), neww, w)
        return neww
    else:
        n, G, D = X.shape
        neww = w.copy()
        for g in range(G):
            Xg = X[:, g, :]
            rg = y[:, g] - b[:, g]
            XtX = Xg.T @ Xg
            Xtr = Xg.T @ rg
            A = XtX + lam * np.eye(D)
            diag = np.diag(A).copy()
            jitter = 1e-8 * (1.0 + float(np.max(np.abs(diag))))
            A = A + jitter * np.eye(D)
            try:
                delta = np.linalg.solve(A, Xtr)
            except Exception:
                denom = np.diag(XtX) + lam + jitter
                denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
                delta = Xtr / denom
            nrm = float(np.sqrt(np.sum(delta * delta)))
            if nrm > bound and nrm > 0.0:
                delta = delta * (bound / nrm)
            cand = w[g] + delta
            neww[g] = np.where(np.isfinite(cand), cand, w[g])
        return neww


def propose_leaves(leaf_ids, residual, recipe, tree_count):
    ids = np.array(leaf_ids, dtype=np.int64, copy=True)
    r = _as_float_array(residual)
    kappa = float(recipe.get('kappa', 1.0))
    clip = float(recipe.get('clip', 1.0))
    if kappa < 0.0:
        kappa = 0.0
    if clip < 0.0:
        clip = 0.0
    T = int(tree_count)
    n = ids.shape[0]
    out = []
    for t in range(T):
        sums = {}
        counts = {}
        col = ids[:, t]
        for i in range(n):
            lid = int(col[i])
            val = float(r[i])
            if not np.isfinite(val):
                continue
            if lid in sums:
                sums[lid] += val
                counts[lid] += 1
            else:
                sums[lid] = val
                counts[lid] = 1
        mapping = {}
        for lid in sums:
            c = counts[lid]
            mean = sums[lid] / float(c)
            shrink = float(c) / (float(c) + kappa) if (float(c) + kappa) > 0.0 else 0.0
            inc = shrink * mean
            if inc > clip:
                inc = clip
            elif inc < -clip:
                inc = -clip
            if not np.isfinite(inc):
                inc = 0.0
            mapping[int(lid)] = float(inc)
        out.append(mapping)
    return out
