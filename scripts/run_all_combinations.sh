#!/bin/bash
# Helper script to generate all training commands for hyperparameter testing
# You can use this to create a job array or run commands in parallel
#
# This script generates all combinations from the config files and outputs
# commands that can be run individually or submitted as jobs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load combinations from config files
if command -v python3 &> /dev/null; then
    python3 -c "
import yaml
from pathlib import Path

config_dir = Path('${PROJECT_ROOT}/config')

# Load expansions
with open(config_dir / 'expansions.yaml') as f:
    expansions = yaml.safe_load(f)['expansions']

# Load k values
with open(config_dir / 'k_values.yaml') as f:
    k_values = yaml.safe_load(f)['k_values']

# Load hooking points
with open(config_dir / 'hooking_points.yaml') as f:
    hooking_points = yaml.safe_load(f)['hooking_points']

# Generate all combinations
for exp in expansions:
    for k in k_values:
        for hp in hooking_points:
            loc = hp['loc']
            stream = hp['stream']
            print(f\"${SCRIPT_DIR}/train_small_flux_sae.sh {exp} {k} \\\"{loc}\\\" {stream}\")
" > "${PROJECT_ROOT}/all_training_commands.txt"

    echo "Generated all training commands to: all_training_commands.txt"
    echo "Total commands: $(wc -l < "${PROJECT_ROOT}/all_training_commands.txt")"
    echo ""
    echo "To run all commands (sequentially):"
    echo "  bash all_training_commands.txt"
    echo ""
    echo "To run in parallel (e.g., 4 at a time):"
    echo "  parallel -j 4 < all_training_commands.txt"
    echo ""
    echo "To submit as a job array (adjust for your job scheduler):"
    echo "  # Example for SLURM:"
    echo "  sbatch --array=1-$(wc -l < "${PROJECT_ROOT}/all_training_commands.txt")%4 job_array.sh"
else
    echo "Error: python3 not found. Cannot generate commands."
    exit 1
fi

