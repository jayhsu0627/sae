#!/bin/bash
# Sequential batch training script for FLUX SAE
# Trains SAEs by pairing hooking points with expansion rates sequentially
# Each expansion rate is paired with the corresponding hooking point (modulo)
#
# Usage:
#   ./scripts/run_sequential_training.sh [options]
#
# Options:
#   --parallel N          Run N jobs in parallel (default: 1, sequential)
#   --max-jobs N          Maximum number of training jobs to run (default: all)
#   --start-from N        Start from the Nth combination (default: 0)
#   --dry-run             Only print commands, don't execute
#
# Examples:
#   ./scripts/run_sequential_training.sh                    # Sequential, all combinations
#   ./scripts/run_sequential_training.sh --parallel 4       # Run 4 jobs in parallel
#   ./scripts/run_sequential_training.sh --max-jobs 10      # Only run first 10
#   ./scripts/run_sequential_training.sh --dry-run          # Preview commands

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Parse command-line arguments
PARALLEL=1
MAX_JOBS=""
START_FROM=0
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --max-jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        --start-from)
            START_FROM="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            cat << EOF
Usage: $0 [options]

Sequential batch training script that pairs hooking points with expansion rates.

Options:
  --parallel N      Run N jobs in parallel (default: 1)
  --max-jobs N      Maximum number of jobs to run (default: all)
  --start-from N    Start from the Nth combination (default: 0)
  --dry-run         Only print commands, don't execute
  -h, --help        Show this help message

EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Change to project root
cd "${PROJECT_ROOT}" || exit 1

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Required for loading YAML configs."
    exit 1
fi

# Load configurations using Python and generate commands directly
echo "Loading configurations from YAML files..."
python3 -c "
import yaml
from pathlib import Path

config_dir = Path('${PROJECT_ROOT}/config')

# Load expansions
with open(config_dir / 'expansions.yaml') as f:
    expansions = yaml.safe_load(f)['expansions']

# Load hooking points
with open(config_dir / 'hooking_points.yaml') as f:
    hooking_points = yaml.safe_load(f)['hooking_points']

# Load k values (if exists, otherwise use default)
try:
    with open(config_dir / 'k_values.yaml') as f:
        k_values = yaml.safe_load(f)['k_values']
except FileNotFoundError:
    k_values = [20]  # Default k value

# Generate commands: for each hooking point, train with all expansion rates
# Outer loop: hooking points, Inner loop: expansions (and k values)
commands = []
for hp in hooking_points:
    loc = hp['loc']
    stream = hp['stream']
    
    for i, expansion in enumerate(expansions):
        # Cycle through k values for each expansion
        k_idx = i % len(k_values)
        k = k_values[k_idx]
        
        # Generate command
        cmd = f\"${SCRIPT_DIR}/train_small_flux_sae.sh {expansion} {k} \\\"{loc}\\\" {stream}\"
        commands.append(cmd)

# Output commands to file
output_file = Path('${PROJECT_ROOT}/sequential_training_commands.txt')
with open(output_file, 'w') as f:
    for cmd in commands:
        f.write(cmd + '\n')

print(f'Generated {len(commands)} training commands')
print(f'Expansions: {len(expansions)}')
print(f'Hooking points: {len(hooking_points)}')
print(f'K values: {len(k_values)}')
" > /tmp/sequential_config_load.log 2>&1

if [ $? -ne 0 ]; then
    echo "Error loading configurations. Check /tmp/sequential_config_load.log"
    cat /tmp/sequential_config_load.log
    exit 1
fi

# Read the generated commands into an array
COMMANDS_FILE="${PROJECT_ROOT}/sequential_training_commands.txt"
if [ ! -f "${COMMANDS_FILE}" ]; then
    echo "Error: Commands file not generated"
    exit 1
fi

# Count total commands before filtering
INITIAL_TOTAL=$(wc -l < "${COMMANDS_FILE}")

# Display configuration summary
echo ""
echo "========================================="
echo "Sequential Batch Training Configuration"
echo "========================================="
cat /tmp/sequential_config_load.log
echo "Parallel jobs: ${PARALLEL}"
echo "Start from: ${START_FROM}"
[ -n "$MAX_JOBS" ] && echo "Max jobs: ${MAX_JOBS}"
echo "Initial total commands: ${INITIAL_TOTAL}"
echo "========================================="
echo ""

# Apply filters
if [ "${START_FROM}" -gt 0 ] || [ -n "${MAX_JOBS}" ]; then
    TEMP_FILE=$(mktemp)
    sed -n "$((START_FROM + 1)),${MAX_JOBS:-$}"p "${COMMANDS_FILE}" > "${TEMP_FILE}"
    mv "${TEMP_FILE}" "${COMMANDS_FILE}"
    REMAINING=$(wc -l < "${COMMANDS_FILE}")
    echo "After filtering: ${REMAINING} commands to run"
    echo ""
fi

# Count total commands after filtering
TOTAL_COMBS=$(wc -l < "${COMMANDS_FILE}")

# Preview first few commands (only if not dry-run)
if [ "$DRY_RUN" = false ]; then
    echo "First 10 commands to run:"
    head -10 "${COMMANDS_FILE}" | nl
    if [ "${TOTAL_COMBS}" -gt 10 ]; then
        echo "... (${TOTAL_COMBS} total commands)"
    fi
    echo ""
fi

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "========================================="
    echo "DRY RUN MODE - All Commands:"
    echo "========================================="
    cat "${COMMANDS_FILE}" | nl
    echo "========================================="
    echo ""
    echo "Total commands: ${TOTAL_COMBS}"
    echo ""
    echo "To run these commands:"
    echo "  bash ${COMMANDS_FILE}"
    echo ""
    if [ "${PARALLEL}" -gt 1 ]; then
        echo "Or in parallel (${PARALLEL} jobs):"
        if command -v parallel &> /dev/null; then
            echo "  parallel -j ${PARALLEL} < ${COMMANDS_FILE}"
        else
            echo "  (parallel command not found, install GNU parallel for parallel execution)"
            echo "  Falling back to sequential execution"
        fi
    fi
    exit 0
fi

# Execute commands
echo ""
echo "Starting training jobs..."
echo ""

if [ "${PARALLEL}" -eq 1 ]; then
    # Sequential execution
    echo "Running sequentially..."
    bash "${COMMANDS_FILE}"
    EXIT_CODE=$?
else
    # Parallel execution
    if command -v parallel &> /dev/null; then
        echo "Running in parallel (${PARALLEL} jobs)..."
        parallel -j "${PARALLEL}" < "${COMMANDS_FILE}"
        EXIT_CODE=$?
    else
        echo "Error: parallel command not found. Install GNU parallel or use --parallel 1"
        echo "Falling back to sequential execution..."
        bash "${COMMANDS_FILE}"
        EXIT_CODE=$?
    fi
fi

echo ""
echo "========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[SUCCESS] All training jobs completed!"
else
    echo "[FAILED] Some training jobs failed (exit code: ${EXIT_CODE})"
fi
echo "Commands file: ${COMMANDS_FILE}"
echo "========================================="

exit ${EXIT_CODE}

