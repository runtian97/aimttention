# AIMttention

A global-attention variant of [AIMNet2](https://github.com/isayevlab/AIMNet2) for molecular potential energy prediction.

Trained models are fully compatible with [aimnet2calc](https://github.com/isayevlab/AIMNet2) for inference, geometry optimization, Hessian computation, and thermodynamic property calculation.

## Algorithm

### Overview

AIMttention replaces the original AIMNet2 convolution-style message passing with **dense molecule-wide multi-head attention**. Every atom attends to every other atom in the molecule, modulated by radial, angular, and dihedral geometric features. This gives the model full O(N^2) all-pairs interaction without a fixed cutoff neighborhood.

### Architecture

```
Input: {coord, numbers, charge, mult}
          |
    [Atomic Embedding]        Element lookup -> per-atom feature vectors
          |
    [AEV: Pairwise Distances]  Compute d_ij and unit displacement vectors r_hat_ij
          |
    ┌─────────────────────────────────────────────┐
    │  Attention Block (repeated L times)          │
    │                                              │
    │  1. Multi-Head Attention                     │
    │     Q,K,V projections from atom features     │
    │     + radial gating:  sigmoid(W * G(d_ij))   │
    │     + radial bias:    W * G(d_ij) on scores  │
    │     + angular gating: sigmoid(W * A_ij)      │
    │     + angular bias:   W * A_ij on scores     │
    │     + dihedral gating/bias (optional 4-body) │
    │                                              │
    │  2. Output Projection + LayerNorm + Residual │
    │  3. Feed-Forward Network + LayerNorm + Res.  │
    │  4. Charge MLP -> update partial charges     │
    │     via NSE (normalized softmax equilibrium) │
    └─────────────────────────────────────────────┘
          |
    [AIM MLP]                  Map to AIM representation
          |
    [Energy MLP]               Per-atom energy prediction
          |
    [Atomic Shift]             Learnable per-element energy shifts (+ SAE)
          |
    [Atomic Sum]               Sum atomic energies -> molecular energy
          |
    [LR Coulomb]               Long-range electrostatic correction
          |
Output: {energy, charges, spin_charges}
        Forces computed externally via autograd on energy
```

### Key Components

**Atomic Environment Vector (AEV):** Computes pairwise distances d_ij and unit displacement vectors. Gaussian radial basis functions with learnable shifts and width provide smooth distance encoding.

**Global Atom Attention:** Each attention block computes multi-head attention where:
- **Radial gating/bias:** Gaussian basis expansion G(d_ij) = exp(-eta * (d_ij - mu_k)^2) modulates attention values (gate) and scores (bias), encoding distance information.
- **Angular features (3-body):** For each pair (i,j), aggregates Chebyshev-polynomial-encoded angles from all triplets (k,i,j): sum_k T_n(cos(theta_kij)) * exp(-eta * d_ik). Provides directional sensitivity.
- **Dihedral features (4-body, optional):** Outer product of angular features from both sides of a pair, capturing torsional interactions.
- **Residual connections and LayerNorm** stabilize deep stacking.

**Charge Equilibration (NSE):** After each attention block, a charge MLP predicts per-atom charge updates. The Normalized Softmax Equilibrium ensures total charge conservation: charges are redistributed so their sum matches the input molecular charge. Supports open-shell systems via two charge channels (alpha/beta spin).

**Output Heads:**
- Energy MLP: maps AIM features to per-atom energies
- Atomic Shift: adds learnable per-element energy offsets (self-atomic energies baked in at compile time)
- Atomic Sum: sums per-atom energies to molecular energy
- LR Coulomb: long-range electrostatic correction using predicted partial charges (simple switching function or damped shifted force for periodic systems)

### Model Configuration

The architecture is controlled by the model YAML file (`examples/model_nci.yaml`):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `nfeature` | Radial basis functions per element | 16 |
| `d2features` | Use 2D element x radial features | true |
| `hidden` | Hidden layer sizes per attention block (list of lists) | [[512,380], [512,380], [512,380,380]] |
| `aim_size` | AIM representation dimension | 256 |
| `attention_heads` | Number of attention heads | 8 |
| `n_angular_basis` | Angular (3-body) Chebyshev basis size | 8 |
| `use_dihedral` | Enable 4-body dihedral features | true |
| `num_charge_channels` | 1 = closed-shell, 2 = open-shell (spin) | 2 |
| `aev.rc_s` | Radial cutoff (Angstrom) | 5.0 |
| `aev.nshifts_s` | Number of Gaussian radial shifts | 16 |

The number of attention layers equals `len(hidden)`. Each entry in `hidden` defines the FFN layer sizes for that block. `num_species` (element embedding table size) is auto-detected from the SAE file.

### Speed Considerations

- **O(N^2) scaling:** Dense attention over all atom pairs. Efficient for small-to-medium molecules (<300 atoms). For larger systems, inference time grows quadratically.
- **Attention layers:** Linear cost with number of layers. Reducing from 3 to 2 gives ~25-30% speedup.
- **Dihedral features:** 4-body terms are computationally expensive. Setting `use_dihedral: false` provides a meaningful speedup.
- **Hidden sizes:** Smaller hidden layers (e.g., [256,192] vs [512,380]) reduce both memory and compute.

## Installation

Create conda environment:
```bash
conda create -n aimttention python=3.11
conda activate aimttention
```

Install PyTorch with appropriate CUDA version from [pytorch.org](https://pytorch.org):
```bash
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
```

Install other dependencies:
```bash
conda install -c conda-forge -c pytorch -c nvidia -f requirements.txt
```

Install the package:
```bash
python setup.py install
```

## Training

### Dataset Format

Training data must be an HDF5 file with groups organized by molecule size. Units: Angstrom, electron-volt, electron charge.

```
$ h5ls -r dataset.h5
/028                     Group
/028/charge              Dataset {25768}
/028/charges             Dataset {25768, 28}
/028/coord               Dataset {25768, 28, 3}
/028/energy              Dataset {25768}
/028/forces              Dataset {25768, 28, 3}
/028/numbers             Dataset {25768, 28}
/029                     Group
/029/charge              Dataset {19404}
/029/charges             Dataset {19404, 29}
/029/coord               Dataset {19404, 29, 3}
/029/energy              Dataset {19404}
/029/forces              Dataset {19404, 29, 3}
/029/numbers             Dataset {19404, 29}
```

Optional keys per group: `mult` (spin multiplicity), `spin_charges` (per-atom spin densities).

### Quick Start

The complete training pipeline is provided in `examples/train_nci.sh`:

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_NAME="aimttention_omol25"
PROJECT_NAME="aimttention"
MODEL_FILE="${SCRIPT_DIR}/model_nci.yaml"
CONFIG_FILE="${SCRIPT_DIR}/train_nci.yaml"
DATASET="/path/to/omol25.h5"
OUTPUT_DIR="${ROOT_DIR}/output/${RUN_NAME}"

mkdir -p "${OUTPUT_DIR}"

# Step 1: Compute self-atomic energies (SAE)
python -m aimttention.cli calc_sae "${DATASET}" "${OUTPUT_DIR}/sae.yaml"

# Step 2: Train
python -m aimttention.cli train \
    --config "${CONFIG_FILE}" \
    --model "${MODEL_FILE}" \
    --save "${OUTPUT_DIR}/model.pt" \
    run_name="${RUN_NAME}" \
    project_name="${PROJECT_NAME}" \
    data.train="${DATASET}" \
    data.sae.energy.file="${OUTPUT_DIR}/sae.yaml" \
    checkpoint.dirname="${OUTPUT_DIR}"

# Step 3: Compile to TorchScript for inference
python -m aimttention.cli jitcompile \
    --model "${MODEL_FILE}" \
    --sae "${OUTPUT_DIR}/sae.yaml" \
    "${OUTPUT_DIR}/model.pt" "${OUTPUT_DIR}/model.jpt"
```

**Before running:** Edit `DATASET` to point to your HDF5 file, and log in to W&B:
```bash
wandb login
```

### Training Pipeline

**Step 1 — Compute SAE:** Self-atomic energies (per-element energy offsets) are computed via linear regression on the training set. This is stored as a YAML file mapping atomic numbers to energy offsets, and baked into the model at JIT compile time.

```bash
python -m aimttention.cli calc_sae dataset.h5 sae.yaml
```

**Step 2 — Train:** The training loop uses PyTorch Ignite with:
- **Multi-task loss:** Weighted sum of energy, forces, charges, and spin charges losses
- **Per-parameter learning rates:** Different learning rates for embeddings, attention blocks, output MLPs
- **ReduceLROnPlateau scheduler:** Reduces LR by 0.75x after 10 epochs without improvement
- **Best-model checkpointing:** Saves top-K models ranked by validation loss
- **DDP support:** Automatically uses all available GPUs via DistributedDataParallel
- **W&B logging:** Tracks all training metrics, model gradients, and learning rates

```bash
python -m aimttention.cli train \
    --config train_nci.yaml \
    --model model_nci.yaml \
    --save model.pt \
    run_name=my_run \
    data.train=dataset.h5 \
    data.sae.energy.file=sae.yaml
```

Config values can be overridden on the command line using dot notation (e.g., `optimizer.kwargs.lr=0.001`).

**Step 3 — JIT Compile:** Converts the trained model to TorchScript (`.jpt`) for use with aimnet2calc. This step:
- Removes the Forces wrapper (forces computed externally via autograd)
- Bakes SAE into atomic shift weights
- Sets `model.cutoff` and `model.cutoff_lr` attributes
- Masks unsupported elements with NaN
- Auto-detects `num_species` from the SAE file

```bash
python -m aimttention.cli jitcompile \
    --model model_nci.yaml \
    --sae sae.yaml \
    model.pt model.jpt
```

### Training Configuration

Key settings in `examples/train_nci.yaml`:

| Section | Parameter | Description |
|---------|-----------|-------------|
| `data` | `train` | Path to HDF5 training dataset |
| `data` | `val_fraction` | Fraction of data used for validation (0.02) |
| `data.sae` | `energy.file` | Path to SAE YAML file |
| `data.samplers.train` | `batch_size` | Batch size in atoms (8192) |
| `data.samplers.train` | `batch_mode` | `atoms` = fixed atom count per batch |
| `data.samplers.train` | `batches_per_epoch` | Number of batches per epoch (5000) |
| `loss.kwargs.components` | `energy.weight` | Energy loss weight (0.7) |
| `loss.kwargs.components` | `forces.weight` | Forces loss weight (0.2) |
| `loss.kwargs.components` | `charges.weight` | Charges loss weight (0.05) |
| `optimizer.kwargs` | `lr` | Base learning rate (0.0004) |
| `scheduler.kwargs` | `patience` | Epochs before LR reduction (10) |
| `trainer` | `epochs` | Maximum training epochs (500) |
| `checkpoint` | `n_saved` | Number of best checkpoints to keep (5) |

### Environment Tips

For optimal data loading performance:
```bash
export OMP_NUM_THREADS=1
```

Single GPU training:
```bash
export CUDA_VISIBLE_DEVICES=0
```

CPU-only training is supported but slow (automatically detected when no CUDA GPUs are available).

## Inference with aimnet2calc

The compiled `.jpt` model is compatible with [aimnet2calc](https://github.com/isayevlab/AIMNet2) for:

**Energy, forces, and Hessian:**
```python
import torch
from aimnet2calc import AIMNet2Calculator

calc = AIMNet2Calculator("model.jpt")
# input: dict with coord, numbers, charge, mult
result = calc(input_data)
# result contains: energy, forces, charges, spin_charges, hessian
```

**ASE geometry optimization:**
```python
from ase import Atoms
from aimnet2calc import AIMNet2ASE

calc = AIMNet2ASE("model.jpt")
atoms = Atoms(...)
atoms.calc = calc
# Use any ASE optimizer
from ase.optimize import BFGS
opt = BFGS(atoms)
opt.run(fmax=0.01)
```

**Thermodynamic properties:**
The Hessian output can be used to compute vibrational frequencies, zero-point energy, and thermodynamic corrections via standard ASE thermochemistry tools.

The model accepts three input modes from aimnet2calc:
- **Mode 0:** Padded batched tensors (B, N, 3) — native format
- **Mode 1:** Flat tensors (N, 3) with `mol_idx` and `nbmat` — auto-converted to padded internally, preserving autograd for force computation
- **Mode 2:** Padded batched with explicit neighbor lists

All energy outputs are in **eV**. Forces are in **eV/Angstrom**.
