"""Feedback-only selection with the original tie-aware finite-budget metric."""
from __future__ import annotations
import numpy as np
from .functional import NativeMLPEditor, recipes


def _vector(x, name):
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite vector")
    return a


def functional_utility(scores, outcomes, budget):
    """Random/oracle-normalized discounted utility; exact expectation at ties.

    Higher is better; 0 is random, 1 is oracle, and negative values are retained.
    Return None when all outcomes are equal (no definable oracle/random gap).
    """
    scores, outcomes = _vector(scores, "scores"), _vector(outcomes, "outcomes")
    if len(scores) != len(outcomes) or type(budget) is not int or not 1 <= budget <= len(scores):
        raise ValueError("Matched vectors and an attainable positive integer budget are required")
    weights = 1.0 / np.log2(np.arange(budget) + 2.0)
    weights /= weights.sum()
    order = np.argsort(-scores, kind="stable")
    inclusion = np.zeros(len(scores))
    start = 0
    while start < budget:
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        inclusion[order[start:end]] = weights[start:min(end, budget)].sum() / (end-start)
        start = end
    random = float(outcomes.mean())
    oracle = float(weights @ np.sort(outcomes)[::-1][:budget])
    gap = oracle-random
    if gap <= 1e-12 * max(1., abs(random), abs(oracle)):
        return None
    return float((inclusion @ outcomes-random)/gap)


def pair_loss(scores, outcomes):
    i, j = np.triu_indices(len(outcomes), 1)
    gap = outcomes[i]-outcomes[j]
    weight = np.abs(gap)
    if weight.sum() <= 1e-15:
        return 0.0
    return float(weight @ np.logaddexp(0., -np.sign(gap)*(scores[i]-scores[j]))/weight.sum())


def choose_update(preactivation, outcomes, w, bias, hidden_bias, folds, *, budget=10, seed=2026110917):
    """Choose a native parameter vector using acquired feedback only.

    No query inputs/outcomes are accepted. Each candidate fits all but one
    feedback fold, validates on that fold, then refits on all feedback if chosen.
    The original model is retained when no proposal improves ranking loss while
    preserving mean validation utility.
    """
    z = np.asarray(preactivation, dtype=np.float64)
    y = _vector(outcomes, "outcomes")
    w, hidden_bias = _vector(w, "w"), _vector(hidden_bias, "hidden_bias")
    folds = np.asarray(folds)
    if z.shape != (len(y), 128) or w.shape != (128,) or hidden_bias.shape != (128,):
        raise ValueError("This released interface supports a 128-unit final hidden layer")
    if not np.isfinite(z).all() or not np.isfinite(bias):
        raise ValueError("Model inputs and parameters must be finite")
    if folds.shape != y.shape or folds.dtype.kind not in "iu" or len(set(folds.tolist())) < 2:
        raise ValueError("Provide one integer feedback-fold ID per observation and at least two folds")
    groups = np.unique(folds)
    if type(budget) is not int or budget < 1 or any(np.count_nonzero(folds==f) <= budget for f in groups):
        raise ValueError("Validation budget must be smaller than each held feedback fold")
    editor = NativeMLPEditor(z, w, bias, hidden_bias, seed)
    original = editor.predict(editor.base)

    def metrics(predictions):
        values = [functional_utility(p, y[ix], budget) for ix,p in predictions]
        if any(v is None for v in values):
            return None
        return {"utility": float(np.mean(values)), "pair_loss": float(np.mean([pair_loss(p, y[ix]) for ix,p in predictions]))}

    base = metrics([(np.flatnonzero(folds==f), original[folds==f]) for f in groups])
    if base is None:
        return editor.base.copy(), {"status":"base-kept", "recipe":"unchanged", "reason":"undefined-validation-utility", "changed_parameters":0, "trials":[]}
    best = {"recipe":"unchanged", **base}
    trials = [best.copy()]
    for recipe in recipes():
        predictions = []
        for f in groups:
            train, held = np.flatnonzero(folds!=f), np.flatnonzero(folds==f)
            vector = editor.propose(train, y[train], recipe)
            predictions.append((held, editor.predict(vector, held)))
        value = metrics(predictions)
        row = {"recipe":recipe["id"], **value}
        trials.append(row)
        if row["utility"] >= base["utility"]-1e-12 and row["pair_loss"] < best["pair_loss"]-1e-12:
            best = row
    selected = next((r for r in recipes() if r["id"]==best["recipe"]), None)
    vector = editor.base.copy() if selected is None else editor.propose(np.arange(len(y)), y, selected)
    return vector, {"status":"base-kept" if selected is None else "updated", **best,
                    "changed_parameters":int(np.count_nonzero(vector-editor.base)), "trials":trials,
                    "feedback_count":len(y), "query_outcomes_used":False}
