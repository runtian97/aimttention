#!/bin/bash

# Configuration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_NAME="aimttention_omol25"
PROJECT_NAME="aimttention"
MODEL_FILE="${SCRIPT_DIR}/model_nci.yaml"
CONFIG_FILE="${SCRIPT_DIR}/train_nci.yaml"
DATASET="/path/to/omol25.h5"
OUTPUT_DIR="${ROOT_DIR}/output/${RUN_NAME}"

mkdir -p "${OUTPUT_DIR}"

# Step 1: Compute SAE
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

# Step 3: Compile to TorchScript
python -m aimttention.cli jitcompile \
    --model "${MODEL_FILE}" \
    --sae "${OUTPUT_DIR}/sae.yaml" \
    "${OUTPUT_DIR}/model.pt" "${OUTPUT_DIR}/model.jpt"
