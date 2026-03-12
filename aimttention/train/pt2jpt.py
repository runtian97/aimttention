import torch
from torch import nn
from aimttention.config import build_module, load_yaml
from typing import Optional, List
import click


def set_eval(model: nn.Module) -> torch.nn.Module:
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


def add_cutoff(model: nn.Module, cutoff: Optional[float] = None) -> nn.Module:
    if cutoff is None:
        cutoff = max(v.item() for k, v in model.state_dict().items() if k.endswith('aev.rc_s'))
    model.cutoff = cutoff
    return model


def add_cutoff_lr(model: nn.Module, cutoff_lr: float = float('inf')) -> nn.Module:
    model.cutoff_lr = cutoff_lr
    return model


def add_sae_to_shifts(model: nn.Module, sae_file: str) -> nn.Module:
    sae = load_yaml(sae_file)
    model.outputs.atomic_shift.double()
    for k, v in sae.items():
        model.outputs.atomic_shift.shifts.weight[k] += v
    return model


def mask_not_implemented_species(model: nn.Module, species: List[int]) -> nn.Module:
    weight = model.afv.weight
    for i in range(1, weight.shape[0]):
        if i not in species:
            weight[i, :] = torch.nan
    return model

@click.command(short_help='Compile PyTorch model to TorchScript.')
@click.argument('pt', type=str)
@click.argument('jpt', type=str)
@click.option('--model', type=str, required=True, help='Path to model definition YAML file.')
@click.option('--sae', type=str, default=None, help='Path to the energy shift YAML file.')
@click.option('--species', type=str, default=None, help='Comma-separated list of parametrized atomic numbers.')
def jitcompile(model, pt, jpt, sae, species):
    """ Build model from YAML config, load weight from PT file and write JIT-compiled JPT file.
    Plus some modifications to work with aimnet2calc.
    """
    # Auto-detect num_species from SAE before building model
    model_cfg = load_yaml(model)
    if sae:
        sae_data = load_yaml(sae)
        num_species = max(sae_data.keys()) + 1
        if 'kwargs' not in model_cfg:
            model_cfg['kwargs'] = {}
        model_cfg['kwargs']['num_species'] = num_species
    model: nn.Module = build_module(model_cfg)
    model = set_eval(model)
    model = add_cutoff(model)
    model = add_cutoff_lr(model)
    sd = torch.load(pt, map_location='cpu')
    print(model.load_state_dict(sd, strict=False))
    if sae:
        model = add_sae_to_shifts(model, sae)
    numbers = None
    if species:
        numbers = list(map(int, species.split(',')))
    elif sae:
        numbers = list(load_yaml(sae).keys())
    if numbers:
        model = mask_not_implemented_species(model, numbers)
        model.register_buffer('impemented_species', torch.tensor(numbers, dtype=torch.int64))
    model_jit = torch.jit.script(model)
    model_jit.save(jpt)


if __name__ == '__main__':
    jitcompile()
