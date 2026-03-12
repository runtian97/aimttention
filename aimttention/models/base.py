import torch
from torch import nn, Tensor
from typing import Dict, Final
from aimttention import nbops


class AIMNet2Base(nn.Module):
    _required_keys: Final = ['coord', 'numbers', 'charge']
    _required_keys_dtype: Final = [torch.float32, torch.int64, torch.float32]
    _optional_keys: Final = ['mult', 'mol_idx', 'nbmat', 'nbmat_lr', 'nb_pad_mask', 'nb_pad_mask_lr', 'shifts', 'shifts_lr', 'cell', 'cutoff_lr']
    _optional_keys_dtype: Final = [torch.float32, torch.int64, torch.int64, torch.int64, torch.bool, torch.bool, torch.float32, torch.float32, torch.float32, torch.float32]
    __constants__ = ['_required_keys', '_required_keys_dtype', '_optional_keys', '_optional_keys_dtype']

    def __init__(self):
        super().__init__()

    def _prepare_dtype(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        for k, d in zip(self._required_keys, self._required_keys_dtype):
            assert k in data, f"Key {k} is required"
            data[k] = data[k].to(d)
        for k, d in zip(self._optional_keys, self._optional_keys_dtype):
            if k in data:
                data[k] = data[k].to(d)
        return data

    def _flat_to_padded(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Convert flat input (mode 1 from aimnet2calc) to padded batched (mode 0).

        Uses index_copy to preserve autograd graph on coordinates,
        so forces can be computed via torch.autograd.grad by the caller.
        """
        mol_idx = data['mol_idx']
        mol_sizes = torch.bincount(mol_idx)
        # last atom is padding in mode 1
        mol_sizes[-1] = mol_sizes[-1] - 1
        B = mol_sizes.shape[0]
        max_N = int(mol_sizes.max().item())
        n_real = int(mol_sizes.sum().item())

        coord_flat = data['coord'][:n_real]
        numbers_flat = data['numbers'][:n_real]

        # Build padded index: for each real atom, compute its position in (B*max_N) layout
        # batch_offset[i] = mol_idx[i] * max_N
        # within_mol_idx[i] = cumulative count within its molecule
        batch_offset = mol_idx[:n_real] * max_N
        # Compute within-molecule index using cumcount
        ones = torch.ones(n_real, device=mol_idx.device, dtype=torch.long)
        cumcount = torch.zeros(n_real, device=mol_idx.device, dtype=torch.long)
        # For each molecule, atoms are contiguous, so within_idx = arange - first_idx_of_mol
        mol_starts = torch.zeros(B, device=mol_idx.device, dtype=torch.long)
        mol_starts[1:] = mol_sizes[:-1].cumsum(0)
        within_idx = torch.arange(n_real, device=mol_idx.device) - mol_starts[mol_idx[:n_real]]

        flat_indices = batch_offset + within_idx

        coord_padded = coord_flat.new_zeros(B * max_N, 3)
        numbers_padded = numbers_flat.new_zeros(B * max_N, dtype=numbers_flat.dtype)
        coord_padded = coord_padded.index_copy(0, flat_indices, coord_flat)
        numbers_padded = numbers_padded.index_copy(0, flat_indices, numbers_flat)

        data['coord'] = coord_padded.view(B, max_N, 3)
        data['numbers'] = numbers_padded.view(B, max_N)

        # Override _nb_mode to 0 so the rest of the model uses dense mode.
        data['_nb_mode'] = torch.tensor(0)
        data['_was_flat'] = torch.tensor(True)
        data['_flat_indices'] = flat_indices
        data['_flat_n_atoms'] = torch.tensor(n_real)

        return data

    def _padded_to_flat(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Convert padded outputs back to flat format for aimnet2calc compatibility."""
        flat_indices = data['_flat_indices']
        n_atoms = int(data['_flat_n_atoms'].item())

        # Flatten charges from (B, max_N) -> pick real atoms -> (N,)
        if 'charges' in data:
            charges_flat_all = data['charges'].flatten(0, 1)
            data['charges'] = charges_flat_all[flat_indices]

        if 'spin_charges' in data:
            sc_flat_all = data['spin_charges'].flatten(0, 1)
            data['spin_charges'] = sc_flat_all[flat_indices]

        # Coord back to flat (N+1, 3) with padding atom for calculator
        coord_flat_all = data['coord'].flatten(0, 1)
        coord_real = coord_flat_all[flat_indices]
        # Append padding atom (zeros)
        data['coord'] = torch.cat([coord_real, coord_real.new_zeros(1, 3)], dim=0)

        # Numbers back to flat with padding
        numbers_flat_all = data['numbers'].flatten(0, 1)
        numbers_real = numbers_flat_all[flat_indices]
        data['numbers'] = torch.cat([numbers_real, numbers_real.new_zeros(1, dtype=numbers_real.dtype)], dim=0)

        return data

    def prepare_input(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        data = self._prepare_dtype(data)
        data = nbops.set_nb_mode(data)
        nb_mode = nbops.get_nb_mode(data)

        data['_was_flat'] = torch.tensor(False)

        # Convert flat/sparse input to padded dense for O(N^2) attention
        if nb_mode == 1:
            data = self._flat_to_padded(data)

        data = nbops.calc_masks(data)
        assert data['charge'].ndim == 1, "Charge should be 1D tensor"
        if 'mult' in data:
            assert data['mult'].ndim == 1, "Mult should be 1D tensor"
        return data
