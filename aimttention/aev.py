from typing import List, Optional, Dict

import torch
from torch import nn, Tensor

from aimttention import ops


class AEVSV(nn.Module):
    """Compute pairwise distances and store radial basis parameters.

    The Gaussian basis parameters (shifts, eta, rc) are stored here and
    shared with GlobalAtomAttention blocks which use them for radial
    gating and bias.

    Parameters:
    -----------
    rmin : float
        Minimum distance for Gaussian basis. Default 0.8.
    rc_s : float
        Cutoff radius. Default 5.0.
    nshifts_s : int
        Number of Gaussian shifts. Default 16.
    eta_s : float, optional
        Gaussian width. Auto-computed if None.
    shifts_s : list of float, optional
        Explicit shift positions. Equidistant if None.
    """

    def __init__(
        self,
        rmin: float = 0.8,
        rc_s: float = 5.0,
        nshifts_s: int = 16,
        eta_s: Optional[float] = None,
        shifts_s: Optional[List[float]] = None,
    ):
        super().__init__()

        self.register_parameter(
            "rc_s",
            nn.Parameter(torch.tensor(rc_s, dtype=torch.float), requires_grad=False),
        )
        if eta_s is None:
            eta_s = (1 / ((rc_s - rmin) / nshifts_s)) ** 2
        self.register_parameter(
            "eta_s",
            nn.Parameter(torch.tensor(eta_s, dtype=torch.float), requires_grad=False),
        )
        if shifts_s is None:
            shifts_s = torch.linspace(rmin, rc_s, nshifts_s + 1)[:nshifts_s]
        else:
            shifts_s = torch.as_tensor(shifts_s, dtype=torch.float)
        self.register_parameter(
            "shifts_s", nn.Parameter(shifts_s, requires_grad=False)
        )

        self.dmat_fill = rc_s

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        d_ij, r_ij = ops.calc_distances(data)
        data["d_ij"] = d_ij
        # Unit displacement vectors for angular/dihedral features
        data["r_hat_ij"] = r_ij / d_ij.unsqueeze(-1).clamp(min=1e-8)
        return data
