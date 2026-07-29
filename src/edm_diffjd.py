"""Differentiable, gradient-descent alternative to edm.get_estimation's discrete
neighborhood search.

Everything here is additive: edm.py, distance_metrics.py and enn.py are untouched.
This module only reads from `edm` (get_init_solution, get_coeficient_matrix, the
histogram builders, AE/NKLD, get_distance) -- the discrete search path they support
keeps behaving exactly as it does today.

One deliberate deviation from edm.make_distributions: that function loops over
predicted-class bins but overwrites `est_hist` each iteration instead of accumulating
(`est_hist = train_hist_dict[a]*init_estimate[a]`, no `+=`), so only the last bin
ever contributes. That's harmless for the discrete neighbor search (which still
explores multiple directions structurally), but fatal for gradient descent: only the
last coordinate would ever receive a gradient, and GD could not search at all. The
`make_distributions_torch` below fixes that (proper sum over predicted-class bins)
purely for this differentiable path.
"""

import numpy as np
import torch

import distance_metrics_diff


def make_distributions_torch(train_hists: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """train_hists: (num_classes, bins) stacked histograms indexed by predicted class.
    weights: (num_classes,) count assigned to each predicted class. Returns a
    normalized mixture histogram, summed (not overwritten) across predicted classes.
    """
    est_hist = torch.sum(train_hists * weights.unsqueeze(1), dim=0)
    return est_hist / (est_hist.sum() + 1e-6)


def round_preserve_sum(x: np.ndarray, total: float) -> np.ndarray:
    """Largest-remainder rounding of a nonnegative float vector to integers that sum
    exactly to `total`. Mirrors the fact that every discrete neighbor step in
    edm.get_estimation is sum-zero -- the per-class total assigned during the search
    is invariant, so the differentiable version must preserve the same invariant to
    be a fair comparison.
    """
    x = np.asarray(x, dtype=np.float64)
    total = int(round(float(total)))
    if total <= 0:
        return np.zeros_like(x, dtype=int)

    floor_x = np.floor(x).astype(int)
    fracs = x - floor_x
    remainder = total - int(floor_x.sum())

    if remainder > 0:
        order = np.argsort(-fracs)
        for i in range(remainder):
            floor_x[order[i % len(floor_x)]] += 1
    elif remainder < 0:
        order = np.argsort(fracs)
        need = -remainder
        i = 0
        while need > 0:
            idx = order[i % len(order)]
            if floor_x[idx] > 0:
                floor_x[idx] -= 1
                need -= 1
            i += 1

    return floor_x


def get_estimation_diffjd(train_hist_dictionary, test_hist_dictionary, num_classes,
                          initial_estimate, steps=200, lr=0.05, jd_eps=1e-8,
                          eps_log=1e-6, device='cpu'):
    """Drop-in alternative to edm.get_estimation: same signature shape and same
    (final_estimation, total_distance) return contract, but replaces the discrete
    neighborhood search with Adam gradient descent on a differentiable Jensen
    Divergence, per class p, started from initial_estimate[p].
    """
    final_estimation = []
    total_distance = 0.0

    for p in range(num_classes):
        pred_hist = torch.as_tensor(np.asarray(test_hist_dictionary[p], dtype=np.float64),
                                    device=device)
        train_hists = torch.stack([
            torch.as_tensor(np.asarray(train_hist_dictionary[p][a], dtype=np.float64),
                            device=device)
            for a in range(num_classes)
        ])

        init_vec = np.asarray(initial_estimate[p], dtype=np.float64)
        T_p = init_vec.sum()

        if T_p <= 0:
            final_estimation.append(np.zeros(num_classes, dtype=int))
            continue

        # softmax(log(v)) == v / sum(v), so this reproduces initial_estimate[p]
        # exactly at step 0 -- the search literally starts from the initial solution.
        logits = torch.tensor(np.log(init_vec + eps_log), dtype=torch.float64,
                              device=device, requires_grad=True)
        optimizer = torch.optim.Adam([logits], lr=lr)

        for _ in range(steps):
            optimizer.zero_grad()
            weights = torch.softmax(logits, dim=0) * T_p
            est_hist = make_distributions_torch(train_hists, weights)
            loss = distance_metrics_diff.JD(est_hist, pred_hist, eps=jd_eps)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            weights = torch.softmax(logits, dim=0) * T_p
            est_hist = make_distributions_torch(train_hists, weights)
            final_dist = distance_metrics_diff.JD(est_hist, pred_hist, eps=jd_eps).item()
            best_estimation = round_preserve_sum(weights.cpu().numpy(), T_p)

        final_estimation.append(best_estimation)
        total_distance += final_dist

    return final_estimation, total_distance
