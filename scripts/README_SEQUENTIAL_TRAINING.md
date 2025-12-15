# Sequential Batch Training Script

The `run_sequential_training.sh` script trains multiple SAEs by sequentially pairing expansion rates with hooking points from the configuration files.

## Overview

This script:
- Loads expansion rates from `config/expansions.yaml`
- Loads hooking points from `config/hooking_points.yaml`
- Loads k values from `config/k_values.yaml` (or uses default k=20)
- Pairs them sequentially: `expansion[i]` with `hooking_point[i % num_hooking_points]` and `k[i % num_k_values]`
- Generates and executes training commands

## Usage

### Basic Sequential Execution

Run all combinations sequentially (one after another):

```bash
./scripts/run_sequential_training.sh
```

### Parallel Execution

Run multiple jobs in parallel (e.g., 4 jobs):

```bash
./scripts/run_sequential_training.sh --parallel 4
```

**Note**: Requires `GNU parallel` to be installed. If not available, falls back to sequential execution.

### Limit Number of Jobs

Run only the first N jobs:

```bash
./scripts/run_sequential_training.sh --max-jobs 10
```

### Start from Specific Job

Start from the Nth combination (useful for resuming):

```bash
./scripts/run_sequential_training.sh --start-from 5
```

### Dry Run (Preview)

Preview what commands would be executed without running them:

```bash
./scripts/run_sequential_training.sh --dry-run
```

## Examples

### Example 1: Preview All Commands

```bash
./scripts/run_sequential_training.sh --dry-run
```

Output:
```
Loading configurations from YAML files...

=========================================
Sequential Batch Training Configuration
=========================================
Generated 6 training commands
Expansions: 6
Hooking points: 3
K values: 6
Parallel jobs: 1
Start from: 0
Total commands: 6
=========================================

First 5 commands to run:
     1	./scripts/train_small_flux_sae.sh 0.5 5 "transformer_blocks.18" 0
     2	./scripts/train_small_flux_sae.sh 1 10 "transformer_blocks.18.ff" 0
     3	./scripts/train_small_flux_sae.sh 2 20 "transformer_blocks.18.attn" 0
     ...
```

### Example 2: Run First 3 Jobs Sequentially

```bash
./scripts/run_sequential_training.sh --max-jobs 3
```

### Example 3: Run 4 Jobs in Parallel

```bash
./scripts/run_sequential_training.sh --parallel 4
```

### Example 4: Resume from Job 5

```bash
./scripts/run_sequential_training.sh --start-from 5
```

## Pairing Logic

The script pairs configurations sequentially with modulo cycling:

- **Expansion[0]** → HookingPoint[0], K[0]
- **Expansion[1]** → HookingPoint[1], K[1]
- **Expansion[2]** → HookingPoint[2], K[2]
- **Expansion[3]** → HookingPoint[0], K[3]  (cycles back)
- ...

**Example with 6 expansions, 3 hooking points, 6 k values:**

| Expansion | Hooking Point | K |
|-----------|---------------|---|
| 0.5       | transformer_blocks.18      | 5 |
| 1         | transformer_blocks.18.ff   | 10 |
| 2         | transformer_blocks.18.attn | 20 |
| 4         | transformer_blocks.18      | 40 |
| 8         | transformer_blocks.18.ff   | 80 |
| 16        | transformer_blocks.18.attn | 160 |

## Output

### Generated Files

1. **`sequential_training_commands.txt`**: List of all generated commands
   - Located in project root
   - Can be used for manual execution or job schedulers

2. **Logs**: Each training job creates its own log file
   - Location: `{base_savedir}/{run_name}.log`
   - As specified in `config/fixed_params.yaml`

3. **Checkpoints**: Each training job saves its checkpoint
   - Location: `{base_savedir}/{run_name}/`
   - As specified in `config/fixed_params.yaml`

### Command Output

The script prints:
- Configuration summary
- First 5 commands to run
- Progress during execution
- Final success/failure status

## Configuration Files

The script reads from:

1. **`config/expansions.yaml`**: Expansion factors to test
   ```yaml
   expansions:
     - 0.5
     - 1
     - 2
     - 4
     - 8
     - 16
   ```

2. **`config/hooking_points.yaml`**: Layer locations to hook
   ```yaml
   hooking_points:
     - loc: "transformer_blocks.18"
       stream: 0
       description: "Transformer block 18"
   ```

3. **`config/k_values.yaml`**: K values for TopK SAE
   ```yaml
   k_values:
     - 5
     - 10
     - 20
     - 40
     - 80
     - 160
   ```

4. **`config/fixed_params.yaml`**: Fixed training parameters (batch size, learning rate, etc.)

## Auto-Detection Features

The training script (`train_small_flux_sae.sh`) automatically:
- **Detects `--use-residual`**: Enabled for entire blocks (`transformer_blocks.18`), disabled for submodules
- **Uses sample percentage**: From `SAMPLE_PERCENTAGE` environment variable (default: 0.25)

## Troubleshooting

### Error: "parallel command not found"

Install GNU parallel:
```bash
# Ubuntu/Debian
sudo apt-get install parallel

# macOS
brew install parallel
```

Or use sequential execution:
```bash
./scripts/run_sequential_training.sh --parallel 1
```

### Error: "python3 not found"

The script requires Python 3 with PyYAML:
```bash
pip install pyyaml
```

### Resuming After Interruption

Use `--start-from` to resume:
```bash
# Find the last completed job number (check logs)
./scripts/run_sequential_training.sh --start-from 5
```

## Advanced Usage

### Using with Job Schedulers

For SLURM:
```bash
# Generate commands
./scripts/run_sequential_training.sh --dry-run

# Submit as job array
sbatch --array=1-$(wc -l < sequential_training_commands.txt)%4 job_array.sh
```

### Manual Execution

Generate commands and run manually:
```bash
# Generate commands
./scripts/run_sequential_training.sh --dry-run

# Run specific command
bash sequential_training_commands.txt | head -1 | bash

# Run all manually
bash sequential_training_commands.txt
```

## Summary

This script provides a convenient way to run systematic hyperparameter sweeps by pairing expansion rates with hooking points sequentially, with support for parallel execution and resuming.

