"""Differentiable dissimilarity function(s) for gradient-based EDM search.

Only Jensen Divergence (JD) is implemented here, matching distance_metrics.JD's
formula. Empty bins are handled with a small epsilon floor on both P and Q (and on
the mixture M) instead of the ad-hoc +eps-then-log used in the numpy version, so the
function stays smooth and its gradient stays bounded everywhere the optimizer can
reach -- the same fix already used for NKLD in the paper (a small constant added to
avoid a zero-division / log(0)).
"""

import torch


def JD(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    p = p / p.sum()
    q = q / q.sum()
    m = torch.clamp(0.5 * (p + q), min=eps)
    kl_pm = torch.sum(p * torch.log(p / m))
    kl_qm = torch.sum(q * torch.log(q / m))
    return 0.5 * kl_pm + 0.5 * kl_qm
