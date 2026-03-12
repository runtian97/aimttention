from typing import Any, Dict, List, Optional, Union, Callable
import torch
from torch import Tensor, nn
from aimttention import ops, nbops, constants
from aimttention.config import get_init_module, get_module
import math


def MLP(n_in: int, n_out: int,
        hidden: List[int] = [],
        activation_fn: Union[Callable, str] = 'torch.nn.GELU',
        activation_kwargs: Dict[str, Any] = {},
        weight_init_fn: Union[Callable, str] = 'torch.nn.init.xavier_normal_',
        bias: bool = True, last_linear: bool = True
        ):
    """ Convenience function to build MLP from config
    """
    hidden = [x for x in hidden if x > 0]
    if isinstance(activation_fn, str):
        activation_fn = get_init_module(
            activation_fn, kwargs=activation_kwargs)
    assert callable(activation_fn)
    if isinstance(weight_init_fn, str):
        weight_init_fn = get_module(weight_init_fn)
    assert callable(weight_init_fn)
    sizes = [n_in, *hidden, n_out]
    layers = list()
    for i in range(1, len(sizes)):
        n_in, n_out = sizes[i-1], sizes[i]
        l = nn.Linear(n_in, n_out, bias=bias)
        with torch.no_grad():
            weight_init_fn(l.weight)
            if bias:
                nn.init.zeros_(l.bias)
        layers.append(l)
        if not (last_linear and i == len(sizes) - 1):
            layers.append(activation_fn)
    return nn.Sequential(*layers)


class Embedding(nn.Embedding):
    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)


class GlobalAtomAttention(nn.Module):
    n_angular_basis: torch.jit.Final[int]
    use_dihedral: torch.jit.Final[bool]

    def __init__(self, n_atom_features: int, n_charge_features: int,
                 radial_shifts: Union[Tensor, List[float]], radial_eta: float,
                 hidden: Optional[List[int]] = None,
                 num_heads: int = 4, dropout: float = 0.0,
                 n_angular_basis: int = 0, use_dihedral: bool = False):
        super().__init__()
        if hidden is None:
            hidden = []
        assert n_atom_features > 0
        assert n_charge_features > 0
        radial_shifts = torch.as_tensor(radial_shifts, dtype=torch.float)
        assert radial_shifts.numel() > 0
        assert num_heads > 0
        assert n_atom_features % num_heads == 0, "n_atom_features must be divisible by num_heads"

        self.n_atom_features = n_atom_features
        self.n_charge_features = n_charge_features
        self.n_radial = radial_shifts.numel()
        self.num_heads = num_heads
        self.head_dim = n_atom_features // num_heads
        self.n_angular_basis = n_angular_basis
        self.use_dihedral = use_dihedral
        self.register_buffer('radial_shifts', radial_shifts)
        self.register_buffer('radial_eta', torch.tensor(float(radial_eta)))

        self.q_proj = nn.Linear(n_atom_features + n_charge_features, n_atom_features)
        self.k_proj = nn.Linear(n_atom_features + n_charge_features, n_atom_features)
        self.v_proj = nn.Linear(n_atom_features + n_charge_features, n_atom_features)
        self.radial_gate = nn.Linear(self.n_radial, n_atom_features)
        self.radial_bias = nn.Linear(self.n_radial, num_heads)
        self.out_proj = nn.Linear(2 * n_atom_features + self.n_radial, n_atom_features)
        self.norm1 = nn.LayerNorm(n_atom_features)
        self.norm2 = nn.LayerNorm(n_atom_features)
        self.dropout = nn.Dropout(dropout)
        self.ffn = MLP(
            n_in=n_atom_features,
            n_out=n_atom_features,
            hidden=hidden,
            activation_fn=nn.GELU(),
            last_linear=True,
        )

        # Angular: 3-body features via aggregated triplet angles
        _ang = max(n_angular_basis, 1)  # always create layers for TorchScript
        self.angular_gate = nn.Linear(_ang, n_atom_features)
        self.angular_bias = nn.Linear(_ang, num_heads)

        # Dihedral: 4-body features via product of angular from both sides
        self.dihedral_gate = nn.Linear(_ang, n_atom_features)
        self.dihedral_bias = nn.Linear(_ang, num_heads)

    def forward(self, a: Tensor, charges: Tensor, data: Dict[str, Tensor]) -> Tensor:
        a_i, a_j = nbops.get_ij(a, data)
        q_i, q_j = nbops.get_ij(charges, data)
        center = torch.cat([a_i.squeeze(-2), q_i.squeeze(-2)], dim=-1)
        neigh = torch.cat([a_j, q_j], dim=-1)

        q = self.q_proj(center).unflatten(-1, (self.num_heads, self.head_dim))
        k = self.k_proj(neigh).unflatten(-1, (self.num_heads, self.head_dim))
        v = self.v_proj(neigh).unflatten(-1, (self.num_heads, self.head_dim))

        d_ij = data['d_ij']
        radial = ops.exp_expand(d_ij, self.radial_shifts, self.radial_eta).transpose(-1, -2).contiguous()

        # Radial gate on values
        gate = torch.sigmoid(self.radial_gate(radial))
        # Attention scores with radial bias
        scores = (k * q.unsqueeze(-3)).sum(-1) / math.sqrt(self.head_dim)
        scores = scores + self.radial_bias(radial)

        # Angular: 3-body gate + bias via aggregated triplet angles
        if self.n_angular_basis > 0:
            angular = ops.compute_angular_features(
                data['r_hat_ij'], d_ij, data['mask_ij'],
                float(self.radial_eta.item()), self.n_angular_basis)
            gate = gate * torch.sigmoid(self.angular_gate(angular))
            scores = scores + self.angular_bias(angular)

            # Dihedral: 4-body gate + bias via product of angular from both sides
            if self.use_dihedral:
                dihedral = angular * angular.transpose(-3, -2)
                gate = gate * torch.sigmoid(self.dihedral_gate(dihedral))
                scores = scores + self.dihedral_bias(dihedral)

        scores = scores.transpose(-1, -2)
        mask = data['mask_ij'].unsqueeze(-2)
        scores = scores.masked_fill(mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

        v = v.transpose(-2, -3)
        context = torch.einsum('...hm,...hmd->...hd', attn, v).flatten(-2, -1)
        radial_context = torch.einsum('...hm,...mg->...hg', attn, radial).mean(dim=-2)
        update = self.out_proj(torch.cat([a, context, radial_context], dim=-1))
        a = self.norm1(a + self.dropout(update))
        ff = self.ffn(a)
        a = self.norm2(a + self.dropout(ff))
        return a


class AtomicShift(nn.Module):
    def __init__(self, key_in: str, key_out: str, num_types: int = 64,
                 dtype: torch.dtype = torch.float, requires_grad: bool = True, reduce_sum=False):
        super().__init__()
        shifts = nn.Embedding(num_types, 1, padding_idx=0, dtype=dtype)
        shifts.weight.requires_grad_(requires_grad)
        self.shifts = shifts
        self.key_in = key_in
        self.key_out = key_out
        self.reduce_sum = reduce_sum

    def extra_repr(self) -> str:
        return f'key_in: {self.key_in}, key_out: {self.key_out}'

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        shifts = self.shifts(data['numbers']).squeeze(-1)
        if self.reduce_sum:
            shifts = nbops.mol_sum(shifts, data)
        data[self.key_out] = data[self.key_in] + shifts
        return data


class AtomicSum(nn.Module):
    def __init__(self, key_in: str, key_out: str):
        super().__init__()
        self.key_in = key_in
        self.key_out = key_out

    def extra_repr(self) -> str:
        return f'key_in: {self.key_in}, key_out: {self.key_out}'

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        data[self.key_out] = nbops.mol_sum(data[self.key_in], data)
        return data


class Output(nn.Module):
    def __init__(self, mlp: Union[Dict, nn.Module], n_in: int, n_out: int,
                 key_in: str, key_out: str):
        super().__init__()
        self.key_in = key_in
        self.key_out = key_out
        if not isinstance(mlp, nn.Module):
            mlp = MLP(n_in=n_in, n_out=n_out, **mlp)
        self.mlp = mlp

    def extra_repr(self) -> str:
        return f'key_in: {self.key_in}, key_out: {self.key_out}'

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        v = self.mlp(data[self.key_in]).squeeze(-1)
        if data['_input_padded'].item():
            v = nbops.mask_i_(v, data, mask_value=0.0)
        data[self.key_out] = v
        return data


class Forces(nn.Module):
    def __init__(self, module: nn.Module, x: str = 'coord', y: str = 'energy', key_out: str = 'forces'):
        super().__init__()
        self.module = module
        self.x = x
        self.y = y
        self.key_out = key_out

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        data[self.x].requires_grad_(True)
        data = self.module(data)
        y = data[self.y]
        g = torch.autograd.grad(
            [y.sum()], [data[self.x]], create_graph=self.training)[0]
        assert g is not None
        data[self.key_out] = - g
        torch.set_grad_enabled(prev)
        return data


class LRCoulomb(nn.Module):
    def __init__(self, key_in: str = 'charges', key_out: str = 'energy',
                 rc: float = 4.6, method: str = 'simple',
                 dsf_alpha: float = 0.2, dsf_rc: float = 15.0):
        super().__init__()
        self.key_in = key_in
        self.key_out = key_out
        self._factor = constants.half_Hartree * constants.Bohr
        self.register_buffer('rc', torch.tensor(rc))
        self.dsf_alpha = dsf_alpha
        self.dsf_rc = dsf_rc
        self.method = method

    def _lazy_calc_dij_lr(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if 'd_ij_lr' not in data:
            nb_mode = nbops.get_nb_mode(data)
            if nb_mode == 0:
                data['d_ij_lr'] = data['d_ij']
            else:
                data['d_ij_lr'] = ops.calc_distances(data, suffix='_lr')[0]
        return data

    def coul_simple(self, data: Dict[str, Tensor]) -> Tensor:
        data = self._lazy_calc_dij_lr(data)
        d_ij = data['d_ij_lr']
        q = data[self.key_in]
        q_i, q_j = nbops.get_ij(q, data, suffix='_lr')
        q_ij = q_i * q_j
        fc = 1.0 - ops.exp_cutoff(d_ij, self.rc)
        e_ij = fc * q_ij / d_ij
        e_ij = nbops.mask_ij_(e_ij, data, 0.0, suffix='_lr')
        e = self._factor * nbops.mol_sum(e_ij.sum(-1), data)
        return e

    def coul_simple_sr(self, data: Dict[str, Tensor]) -> Tensor:
        d_ij = data['d_ij']
        q = data[self.key_in]
        q_i, q_j = nbops.get_ij(q, data)
        q_ij = q_i * q_j
        fc = ops.exp_cutoff(d_ij, self.rc)
        e_ij = fc * q_ij / d_ij
        e_ij = nbops.mask_ij_(e_ij, data, 0.0)
        e = self._factor * nbops.mol_sum(e_ij.sum(-1), data)
        return e

    def coul_dsf(self, data: Dict[str, Tensor]) -> Tensor:
        data = self._lazy_calc_dij_lr(data)
        d_ij = data['d_ij_lr']
        q = data[self.key_in]
        q_i, q_j = nbops.get_ij(q, data, suffix='_lr')
        epot = ops.coulomb_potential_dsf(q_j, d_ij, self.dsf_rc, self.dsf_alpha, data)
        q_i = q_i.squeeze(-1)
        e = q_i * epot
        e = self._factor * nbops.mol_sum(e, data)
        e = e - self.coul_simple_sr(data)
        return e

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if self.method == 'simple':
            e = self.coul_simple(data)
        elif self.method == 'dsf':
            e = self.coul_dsf(data)
        else:
            e = self.coul_simple(data)
        if self.key_out in data:
            data[self.key_out] = data[self.key_out] + e
        else:
            data[self.key_out] = e
        return data
