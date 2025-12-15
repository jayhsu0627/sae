# Training Script Usage Guide

## Quick Start

The `train_small_flux_sae.sh` script trains a single SAE with specific hyperparameters. It loads fixed parameters from `config/fixed_params.yaml` and accepts variable parameters as command-line arguments.

## Basic Usage

```bash
./scripts/train_small_flux_sae.sh <expansion> <k> <loc> [stream]
```

### Required Arguments

1. **expansion** - Expansion factor (multiplies features to get hidden dimension)
   - Examples: `0.5`, `1`, `2`, `4`, `8`, `16`
   - Formula: `pages = expansion × features` (e.g., 4 × 3072 = 12,288)

2. **k** - Top-K sparsity value (number of active features)
   - Examples: `5`, `10`, `20`, `40`, `80`, `160`
   - Groundtruth uses: `20`

3. **loc** - Layer location to hook activations from
   - Examples: 
     - `"transformer_blocks.0.attn"` - Layer 0 attention
     - `"transformer_blocks.18.ff"` - Layer 18 feedforward (groundtruth)
     - `"single_transformer_blocks.0.proj_mlp"` - Layer 0 MLP projection

4. **stream** (optional) - Stream index
   - `0` - Image stream (queries) - **groundtruth default**
   - `1` - Text stream (keys)
   - Default: `0` if not specified

## Examples

### Example 1: Groundtruth Settings (Layer 18, Image Stream)

```bash
./scripts/train_small_flux_sae.sh 4 20 "transformer_blocks.18.ff" 0
```

This trains:
- Expansion: 4 (12,288 hidden features)
- K: 20 (groundtruth value)
- Layer: 18 feedforward
- Stream: 0 (image stream)

### Example 2: Layer 0 Attention

```bash
./scripts/train_small_flux_sae.sh 4 20 "transformer_blocks.0.attn" 0
```

### Example 3: Smaller Model

```bash
./scripts/train_small_flux_sae.sh 2 10 "transformer_blocks.0.attn" 0
```

### Example 4: Text Stream

```bash
./scripts/train_small_flux_sae.sh 4 20 "transformer_blocks.0.attn" 1
```

## Advanced Configuration

### Using Residual Activations

**Auto-Detection** (Recommended - Default Behavior):
The script **automatically determines** whether to use `--use-residual` based on the location:

- **Entire blocks** (`transformer_blocks.18`) → Automatically uses `--use-residual` ✓
  - Matches groundtruth behavior (trains on `output - input` = `Attn(x) + MLP(x)`)
  
- **Submodules** (`transformer_blocks.18.attn`, `transformer_blocks.18.ff`) → Defaults to no residual
  - Trains on direct output

**Examples**:
```bash
# Entire block - automatically uses residual (groundtruth style)
./scripts/train_small_flux_sae.sh 4 20 "transformer_blocks.18" 0
# Output: "Auto-detected: Using --use-residual for entire transformer block (groundtruth style)"

# Submodule - automatically uses no residual
./scripts/train_small_flux_sae.sh 4 20 "transformer_blocks.18.ff" 0
# Output: "Auto-detected: Not using --use-residual for submodule (transformer_blocks.18.ff)"
```

**Manual Override**:
To explicitly control residual usage (overrides auto-detection):

```bash
# Force use residual
USE_RESIDUAL=true ./scripts/train_small_flux_sae.sh 4 20 "transformer_blocks.18.ff" 0

# Force no residual
USE_RESIDUAL=false ./scripts/train_small_flux_sae.sh 4 20 "transformer_blocks.18" 0
```

### Changing Number of Stat Batches

To collect more/fewer batches for initialization (default: 10):

```bash
export NUM_STAT_BATCHES=20
./scripts/train_small_flux_sae.sh 4 20 "transformer_blocks.18.ff" 0
```

### Modifying Fixed Parameters

Edit `config/fixed_params.yaml` to change parameters that remain constant:

```yaml
batch_size: 8        # Batch size
iters: 2000          # Number of training iterations
nsamples: 128        # Number of activations to sample per batch
lr: 1e-4            # Learning rate
normalise: true     # Decoder normalization
```

These are loaded automatically - you don't need to pass them as arguments.

## Output Locations

- **Checkpoints**: `./checkpoints/hyperparameter_test/<run_name>/`
- **Logs**: `./checkpoints/hyperparameter_test/<run_name>.log`

Run names follow the format:
```
exp<expansion>_k<k>_<location>_stream<stream>
```

Example: `exp4_k20_transformer_blocks_18_ff_stream0`

## Common Layer Locations

### Transformer Blocks
- `"transformer_blocks.0.attn"` - Layer 0 attention
- `"transformer_blocks.0.ff"` - Layer 0 feedforward
- `"transformer_blocks.18.attn"` - Layer 18 attention (groundtruth layer)
- `"transformer_blocks.18.ff"` - Layer 18 feedforward (groundtruth)

### Single Transformer Blocks
- `"single_transformer_blocks.0.proj_mlp"` - Layer 0 MLP projection
- `"single_transformer_blocks.37.proj_mlp"` - Layer 37 MLP projection

## Recommended Settings

### For Groundtruth Alignment

```bash
# Use these settings to match groundtruth paper:
USE_RESIDUAL=true \
./scripts/train_small_flux_sae.sh 4 20 "transformer_blocks.18.ff" 0
```

### For Faster Experimentation

```bash
# Smaller model, fewer iterations (edit config/fixed_params.yaml):
# Set iters: 500, nsamples: 64

./scripts/train_small_flux_sae.sh 2 10 "transformer_blocks.0.attn" 0
```

## Troubleshooting

### Script Not Executable
```bash
chmod +x scripts/train_small_flux_sae.sh
```

### Missing Python Dependencies
```bash
pip install pyyaml  # Required for loading config
```

### CUDA Out of Memory
- Reduce `batch_size` in `config/fixed_params.yaml`
- Reduce `nsamples` in `config/fixed_params.yaml`
- Use smaller expansion factor

### Check Logs
```bash
tail -f ./checkpoints/hyperparameter_test/exp4_k20_*.log
```

## Running Multiple Experiments

### Sequential (One at a Time)
```bash
./scripts/train_small_flux_sae.sh 4 20 "transformer_blocks.18.ff" 0
./scripts/train_small_flux_sae.sh 4 20 "transformer_blocks.0.attn" 0
./scripts/train_small_flux_sae.sh 2 10 "transformer_blocks.18.ff" 0
```

### Parallel (Background Jobs)
```bash
./scripts/train_small_flux_sae.sh 4 20 "transformer_blocks.18.ff" 0 &
./scripts/train_small_flux_sae.sh 4 20 "transformer_blocks.0.attn" 0 &
wait  # Wait for all jobs to complete
```

### With Job Scheduler (SLURM)
```bash
#!/bin/bash
#SBATCH --job-name=sae_train
#SBATCH --gres=gpu:1

./scripts/train_small_flux_sae.sh 4 20 "transformer_blocks.18.ff" 0
```

## Quick Reference

| Parameter | Description | Examples |
|-----------|-------------|----------|
| expansion | Hidden dim multiplier | 2, 4, 8 |
| k | Active features | 10, 20, 40 |
| loc | Layer location | `"transformer_blocks.18.ff"` |
| stream | Image (0) or Text (1) | 0, 1 |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_RESIDUAL` | `false` | Use residual activations |
| `NUM_STAT_BATCHES` | `10` | Batches for initialization |

