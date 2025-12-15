#!/bin/bash
# Hyperparameter testing script for FLUX SAE training
# Trains a SINGLE parameter combination at a time (no loops)
#
# Usage:
#   ./train_small_flux_sae.sh <expansion> <k> <loc> [stream]
#
# Examples:
#   ./train_small_flux_sae.sh 4 20 "transformer_blocks.0.attn" 0
#   ./train_small_flux_sae.sh 2 10 "single_transformer_blocks.0.proj_mlp"
#   ./train_small_flux_sae.sh 8 40 "transformer_blocks.18.ff" 0
#
# Configurations are loaded from YAML files in the config/ directory:
#   - config/fixed_params.yaml: Fixed training parameters
#
# To run multiple combinations:
#   - Use a job scheduler (SLURM, etc.) to submit multiple jobs
#   - Use scripts/run_all_combinations.sh to generate all commands
#   - Run the script multiple times with different parameters manually
#
# To modify hyperparameters, edit the YAML files instead of this script.

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 <expansion> <k> <loc> [stream]

Arguments:
  expansion    Expansion factor (e.g., 0.5, 1, 2, 4, 8, 16)
  k            K value for TopK SAE (e.g., 5, 10, 20, 40, 80, 160)
  loc          Hooking point location (e.g., "transformer_blocks.0.attn")
  stream       Stream index (0 for image stream, 1 for text stream) [default: 0]

Examples:
  $0 4 20 "transformer_blocks.0.attn" 0
  $0 2 10 "single_transformer_blocks.0.proj_mlp" 0
  $0 8 40 "transformer_blocks.18.ff"

EOF
    exit 1
}

# Parse command-line arguments
if [ $# -lt 3 ]; then
    echo "Error: Missing required arguments"
    show_usage
fi

EXPANSION="$1"
K="$2"
LOC="$3"
STREAM="${4:-0}"  # Default to 0 if not provided

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Change to project root for running fluxsae.py
cd "${PROJECT_ROOT}" || exit 1

# Load fixed parameters from YAML file using Python
echo "Loading fixed parameters from config/fixed_params.yaml..."
if command -v python3 &> /dev/null; then
    # Try to load config using Python
    FIXED_PARAMS=$(python3 -c "
import yaml
import sys
from pathlib import Path

try:
    config_path = Path('${PROJECT_ROOT}/config/fixed_params.yaml')
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Output as bash variables
        for key, value in config.items():
            if value is None:
                value = ''
            elif isinstance(value, bool):
                value = 'true' if value else 'false'
            elif isinstance(value, str):
                value = f'\"{value}\"'
            print(f'{key.upper()}={value}')
    else:
        print('# Config file not found, using defaults', file=sys.stderr)
except Exception as e:
    print(f'# Error loading config: {e}', file=sys.stderr)
" 2>/dev/null)
    
    if [ -n "$FIXED_PARAMS" ]; then
        eval "$FIXED_PARAMS"
    fi
fi

# Set defaults if not loaded from config (aligned with groundtruth settings)
ARCH="${ARCH:-topk}"
DATASET="${DATASET:-cc3m}"
BATCH_SIZE="${BATCH_SIZE:-8}"
FEATURES="${FEATURES:-3072}"
LR="${LR:-1e-4}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-128}"
AUXK="${AUXK:-0.03125}"
AUXK_COEF="${AUXK_COEF:-0.03125}"
DEAD_STEPS_THRESHOLD="${DEAD_STEPS_THRESHOLD:-10000000}"
BODYCOUNT="${BODYCOUNT:-8192}"
LMBDA="${LMBDA:-0.01}"
LMBDA_WARMUP_STEPS="${LMBDA_WARMUP_STEPS:-128}"
ITERS="${ITERS:-1000}"
NSAMPLES="${NSAMPLES:-128}"
NORMALISE="${NORMALISE:-true}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BASE_SAVEDIR="${BASE_SAVEDIR:-./checkpoints/hyperparameter_test}"
# Groundtruth-aligned defaults
NUM_STAT_BATCHES="${NUM_STAT_BATCHES:-10}"  # Number of batches for pre-bias initialization
SAMPLE_PERCENTAGE="${SAMPLE_PERCENTAGE:-0.25}"  # Optional: percentage per prompt (e.g., 0.1 for 10%). If set, overrides nsamples

# Automatically determine use-residual based on location
# Entire blocks (transformer_blocks.X without .attn or .ff) should use residual (groundtruth style)
# Submodules (.attn, .ff) default to false unless explicitly set
if [ -z "${USE_RESIDUAL}" ]; then
    # Check if this is an entire transformer block (not a submodule)
    if [[ "${LOC}" == *"transformer_blocks"* ]] && \
       [[ "${LOC}" != *".attn"* ]] && \
       [[ "${LOC}" != *".ff"* ]]; then
        # Entire block: use residual (matches groundtruth)
        USE_RESIDUAL="true"
        echo "Auto-detected: Using --use-residual for entire transformer block (groundtruth style)"
    else
        # Submodule (attention, MLP, etc.): default to false
        USE_RESIDUAL="false"
        echo "Auto-detected: Not using --use-residual for submodule (${LOC})"
    fi
else
    # User explicitly set USE_RESIDUAL, respect their choice
    echo "Using explicit USE_RESIDUAL=${USE_RESIDUAL}"
fi

# Create descriptive name for this run
LOC_SANITIZED=$(echo "${LOC}" | sed 's/[^a-zA-Z0-9]/_/g')
RUN_NAME="exp${EXPANSION}_k${K}_${LOC_SANITIZED}_stream${STREAM}"

# Create log file
mkdir -p "${BASE_SAVEDIR}"
LOG_FILE="${BASE_SAVEDIR}/${RUN_NAME}.log"

echo ""
echo "========================================="
echo "Training SAE with single parameter combination"
echo "Run name: ${RUN_NAME}"
echo "Expansion: ${EXPANSION}"
echo "K: ${K}"
echo "Loc: ${LOC}"
echo "Stream: ${STREAM}"
echo "Use Residual: ${USE_RESIDUAL}"
echo "Sample Percentage: ${SAMPLE_PERCENTAGE}"
echo "Started at $(date)"
echo "========================================="
echo ""

# Run training
python fluxsae.py \
    --name "${RUN_NAME}" \
    --dataset "${DATASET}" \
    --arch "${ARCH}" \
    --batch_size "${BATCH_SIZE}" \
    --features "${FEATURES}" \
    --expansion "${EXPANSION}" \
    --lr "${LR}" \
    --lr_warmup_steps "${LR_WARMUP_STEPS}" \
    --k "${K}" \
    --auxk "${AUXK}" \
    --auxk_coef "${AUXK_COEF}" \
    --dead_steps_threshold "${DEAD_STEPS_THRESHOLD}" \
    --bodycount "${BODYCOUNT}" \
    --savedir "${BASE_SAVEDIR}" \
    --lmbda "${LMBDA}" \
    --lmbda_warmup_steps "${LMBDA_WARMUP_STEPS}" \
    --loc "${LOC}" \
    --stream "${STREAM}" \
    --iters "${ITERS}" \
    --nsamples "${NSAMPLES}" \
    --normalise "${NORMALISE}" \
    --num_workers "${NUM_WORKERS}" \
    --num-stat-batches "${NUM_STAT_BATCHES}" \
    $([ "${USE_RESIDUAL}" = "true" ] && echo "--use-residual" || echo "") \
    $([ -n "${SAMPLE_PERCENTAGE}" ] && echo "--sample-percentage ${SAMPLE_PERCENTAGE}" || echo "") \
    2>&1 | tee "${LOG_FILE}"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================="
if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
    echo "[SUCCESS] Training completed successfully!"
    echo "Checkpoint saved to: ${BASE_SAVEDIR}/${RUN_NAME}/"
else
    echo "[FAILED] Training failed (exit code: ${TRAIN_EXIT_CODE})"
    echo "Check log file: ${LOG_FILE}"
fi
echo "Finished at $(date)"
echo "========================================="

exit ${TRAIN_EXIT_CODE}

# # Entire block - automatically uses residual
# ./scripts/train_small_flux_sae.sh 4 20 "transformer_blocks.18" 0
# # → "Auto-detected: Using --use-residual for entire transformer block (groundtruth style)"

# # Submodule - automatically uses no residual  
# ./scripts/train_small_flux_sae.sh 4 20 "transformer_blocks.18.ff" 0
# # → "Auto-detected: Not using --use-residual for submodule (transformer_blocks.18.ff)"

# # Manual override
# USE_RESIDUAL=true ./scripts/train_small_flux_sae.sh 4 20 "transformer_blocks.18.ff" 0
# # → "Using explicit USE_RESIDUAL=true"