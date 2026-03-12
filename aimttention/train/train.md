# AIMttention Training Examples

See the main [README.md](../../README.md) for full documentation.

## Quick Start

```bash
cd examples
# Edit train_nci.sh: set DATASET to your HDF5 file path
wandb login
bash train_nci.sh
```

This runs the full pipeline: SAE computation, training, and JIT compilation.

## Manual Steps

### 1. Compute SAE
```bash
python -m aimttention.cli calc_sae dataset.h5 sae.yaml
```

### 2. Train
```bash
python -m aimttention.cli train \
    --config examples/train_nci.yaml \
    --model examples/model_nci.yaml \
    --save output/model.pt \
    run_name=my_run \
    data.train=dataset.h5 \
    data.sae.energy.file=sae.yaml
```

### 3. Compile to TorchScript
```bash
python -m aimttention.cli jitcompile \
    --model examples/model_nci.yaml \
    --sae sae.yaml \
    output/model.pt output/model.jpt
```

### 4. Use with aimnet2calc
```python
from aimnet2calc import AIMNet2Calculator
calc = AIMNet2Calculator("output/model.jpt")
```
