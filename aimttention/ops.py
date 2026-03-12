import math
from typing import Dict, List, Tuple

import torch
from torch import Tensor

from aimttention import nbops


def calc_distances(data: Dict[str, Tensor], suffix: str = '', pad_value: float=1.0) -> Tuple[Tensor, Tensor]:
    coord_i, coord_j = nbops.get_ij(data["coord"], data, suffix)
    if f"shifts{suffix}" in data:
        assert "cell" in data, "cell is required if shifts are provided"
        nb_mode = nbops.get_nb_mode(data)
        if nb_mode == 2:
            shifts = torch.einsum('bnmd,bdh->bnmh', data[f"shifts{suffix}"], data["cell"])
        else:
            shifts = data[f"shifts{suffix}"] @ data["cell"]
        coord_j = coord_j + shifts
    r_ij = coord_j - coord_i
    d_ij = torch.norm(r_ij, dim=-1)
    d_ij = nbops.mask_ij_(d_ij, data, mask_value=pad_value, inplace=False, suffix=suffix)
    return d_ij, r_ij


def cosine_cutoff(d_ij: Tensor, rc: float) -> Tensor:
    fc = 0.5 * (torch.cos(d_ij.clamp(min=1e-6, max=rc) * (math.pi / rc)) + 1.0)
    return fc


def exp_cutoff(d: Tensor, rc: Tensor) -> Tensor:
    fc = (
        torch.exp(-1.0 / (1.0 - (d / rc).clamp(0, 1.0 - 1e-6).pow(2)))
        / 0.36787944117144233
    )
    return fc


def exp_expand(d_ij: Tensor, shifts: Tensor, eta: float) -> Tensor:
    # expand on axis -2, e.g. (b, n, m) -> (b, n, shifts, m)
    return torch.exp(-eta * (d_ij.unsqueeze(-2) - shifts.unsqueeze(-1)) ** 2)


def compute_angular_features(r_hat: Tensor, d_ij: Tensor, mask_ij: Tensor,
                              eta: float, n_basis: int) -> Tensor:
    """Compute angular features for all atom pairs by aggregating triplet angles.

    For each pair (i,j), computes: sum_k T_n(cos(angle_kij)) * exp(-eta * d_ik)
    where k runs over all atoms (excluding self and padding).
    Uses Chebyshev polynomials T_0, T_1, ..., T_{n-1} as angular basis.

    Args:
        r_hat: unit displacement vectors
        d_ij: pairwise distances
        mask_ij: boolean mask (True = masked)
        eta: decay rate for distance weighting
        n_basis: number of Chebyshev basis functions
    Returns:
        angular features per pair
    """
    # cos_theta[...,i,j,k] = cos(angle jik at vertex i)
    cos_theta = torch.einsum('...ijd,...ikd->...ijk', r_hat, r_hat)

    # Distance weighting: w[...,i,k] = exp(-eta * d_ik), masked
    w = torch.exp(-eta * d_ij).masked_fill(mask_ij, 0.0)
    w = w.unsqueeze(-2)  # (..., n_i, 1, n_k) for broadcasting

    # Compute Chebyshev polynomials and aggregate over k
    result: List[Tensor] = []
    T_prev = torch.ones_like(cos_theta)
    T_curr = cos_theta

    for i in range(n_basis):
        if i == 0:
            T_i = T_prev
        elif i == 1:
            T_i = T_curr
        else:
            T_i = 2 * cos_theta * T_curr - T_prev
            T_prev = T_curr
            T_curr = T_i
        result.append((T_i * w).sum(-1))

    return torch.stack(result, dim=-1)


def coulomb_potential_dsf(q_j: Tensor, d_ij: Tensor, Rc: float, alpha: float, data: Dict[str, Tensor]) -> Tensor:
    _c1 = (alpha * d_ij).erfc() / d_ij
    _c2 = math.erfc(alpha * Rc) / Rc
    _c3 = _c2 / Rc
    _c4 = 2 * alpha * math.exp(- (alpha * Rc) ** 2) / (Rc * math.pi ** 0.5)
    epot = q_j * (_c1 - _c2 + (d_ij - Rc) * (_c3 + _c4))
    epot = nbops.mask_ij_(epot, data, mask_value=0.0, inplace=True, suffix='_lr')
    epot = epot.sum(-1)
    return epot


def nse(Q: Tensor, q_u: Tensor, f_u: Tensor, data: Dict[str, Tensor], epsilon: float = 1.0e-6) -> Tensor:
    # Q and q_u and f_u must have last dimension size 1 or 2
    F_u = nbops.mol_sum(f_u, data) + epsilon
    Q_u = nbops.mol_sum(q_u, data)
    dQ = Q - Q_u
    # for loss
    data['_dQ'] = dQ

    nb_mode = nbops.get_nb_mode(data)
    if nb_mode in (0, 2):
        F_u = F_u.unsqueeze(-2)
        dQ = dQ.unsqueeze(-2)
    elif nb_mode == 1:
        data['mol_sizes'][-1] += 1
        F_u = torch.repeat_interleave(F_u, data['mol_sizes'], dim=0)
        dQ = torch.repeat_interleave(dQ, data['mol_sizes'], dim=0)
        data['mol_sizes'][-1] -= 1
    else:
        raise ValueError(f"Invalid neighbor mode: {nb_mode}")
    f = f_u / F_u
    q = q_u + f * dQ
    return q
