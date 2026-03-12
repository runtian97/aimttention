import torch
from torch import nn, Tensor
from typing import Dict, List, Tuple, Union, Sequence, Mapping
from aimttention import ops, nbops
from aimttention.aev import AEVSV
from aimttention.modules import MLP, Embedding, GlobalAtomAttention
from aimttention.models.base import AIMNet2Base


class AIMNet2(AIMNet2Base):
    def __init__(self, aev: Dict, nfeature: int, d2features: bool, hidden: Tuple[List[int]],
                 aim_size: int, outputs: Union[List[nn.Module], Dict[str, nn.Module]],
                 num_charge_channels: int = 1, attention_heads: int = 4, attention_dropout: float = 0.0,
                 n_angular_basis: int = 0, use_dihedral: bool = False,
                 num_species: int = 64):
        super().__init__()

        assert num_charge_channels in [1, 2], "num_charge_channels must be 1 (closed shell) or 2 (NSE for open-shell)."
        self.num_charge_channels = num_charge_channels

        self.aev = AEVSV(**aev)
        nshifts_s = aev['nshifts_s']
        if d2features:
            nfeature_tot = nshifts_s * nfeature
        else:
            nfeature_tot = nfeature
        self.nfeature = nfeature
        self.nshifts_s = nshifts_s
        self.d2features = d2features

        self.afv = Embedding(num_embeddings=num_species, embedding_dim=nfeature, padding_idx=0)

        with torch.no_grad():
            nn.init.orthogonal_(self.afv.weight[1:])
            if d2features:
                self.afv.weight = nn.Parameter(self.afv.weight.clone().unsqueeze(-1).expand(num_species, nfeature, nshifts_s).flatten(-2, -1))

        self.attention_blocks = nn.ModuleList([
            GlobalAtomAttention(
                n_atom_features=nfeature_tot,
                n_charge_features=num_charge_channels,
                radial_shifts=self.aev.shifts_s.detach().clone(),
                radial_eta=float(self.aev.eta_s.item()),
                hidden=h,
                num_heads=attention_heads,
                dropout=attention_dropout,
                n_angular_basis=n_angular_basis,
                use_dihedral=use_dihedral,
            )
            for h in hidden
        ])

        mlp_param = {'activation_fn': nn.GELU(), 'last_linear': True}
        self.charge_mlps = nn.ModuleList([
            MLP(n_in=nfeature_tot, n_out=2 * num_charge_channels, hidden=h, **mlp_param)
            for h in hidden
        ])
        self.aim_mlp = MLP(n_in=nfeature_tot, n_out=aim_size, hidden=hidden[-1], activation_fn=nn.GELU(), last_linear=True)

        if isinstance(outputs, Sequence):
            self.outputs = nn.ModuleList(outputs)
        elif isinstance(outputs, Mapping):
            self.outputs = nn.ModuleDict(outputs)
        else:
            raise TypeError('`outputs` is not either list or dict')

        # Sync AtomicShift embedding size to match num_species
        for m in self.outputs.modules():
            if hasattr(m, 'shifts') and isinstance(m.shifts, nn.Embedding):
                if m.shifts.num_embeddings != num_species:
                    old = m.shifts
                    m.shifts = nn.Embedding(num_species, old.embedding_dim,
                                            padding_idx=old.padding_idx, dtype=old.weight.dtype)
                    m.shifts.weight.requires_grad_(old.weight.requires_grad)
        
    def _preprocess_spin_polarized_charge(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        assert 'mult' in data, "'mult' key is required for NSE if two channels for charge are not provided"
        _half_spin = 0.5 * (data['mult'] - 1.0)
        _half_q = 0.5 * data['charge']
        data['charge'] = torch.stack([_half_q + _half_spin, _half_q - _half_spin], dim=-1)
        return data

    def _postprocess_spin_polarized_charge(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        data['spin_charges'] = data['charges'][..., 0] - data['charges'][..., 1]
        data['charges'] = data['charges'].sum(dim=-1)
        data['charge'] = data['charge'].sum(dim=-1)
        return data
    
    def _update_q(self, data: Dict[str, Tensor], x: Tensor, delta_q: bool = True) -> Dict[str, Tensor]:
        _q, _f = x.split([self.num_charge_channels, self.num_charge_channels], dim=-1)
        # for loss
        data['_delta_Q'] = data['charge'] - nbops.mol_sum(_q, data)
        if delta_q:
            q = data['charges'] + _q
        else:
            q = _q
        f = _f.pow(2)
        q = ops.nse(data['charge'], q, f, data)
        data['charges'] = q
        return data    


    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        data = self.prepare_input(data)

        # initial features
        a: Tensor = self.afv(data['numbers'])
        data['a'] = a

        # NSE case
        if self.num_charge_channels == 2:
            data = self._preprocess_spin_polarized_charge(data)
        else:
            data['charge'] = data['charge'].unsqueeze(-1)  # make sure that charge has channel dimension
        data['charges'] = data['a'].new_zeros(data['a'].shape[:-1] + (self.num_charge_channels,))

        # AEV
        data = self.aev(data)

        # Attention-based MP iterations
        for ipass, (attn, charge_mlp) in enumerate(zip(self.attention_blocks, self.charge_mlps)):
            data['a'] = attn(data['a'], data['charges'], data)
            if data['_input_padded'].item():
                data['a'] = nbops.mask_i_(data['a'], data, mask_value=0.0)

            _q = charge_mlp(data['a'])
            if data['_input_padded'].item():
                _q = nbops.mask_i_(_q, data, mask_value=0.0)
            data = self._update_q(data, _q, delta_q=(ipass > 0))

        data['aim'] = self.aim_mlp(data['a'])
        if data['_input_padded'].item():
            data['aim'] = nbops.mask_i_(data['aim'], data, mask_value=0.0)

        # squeeze charges
        if self.num_charge_channels == 2:
            data = self._postprocess_spin_polarized_charge(data)
        else:
            data['charges'] = data['charges'].squeeze(-1)
            data['charge'] = data['charge'].squeeze(-1)            

        # readout
        for m in self.outputs.children():
            data = m(data)

        # Convert back to flat format if input was flat (for aimnet2calc compat)
        if data['_was_flat'].item():
            data = self._padded_to_flat(data)

        return data
