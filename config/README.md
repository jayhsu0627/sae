# Hyperparameter Configuration Files

This directory contains YAML configuration files for hyperparameter testing of FLUX SAE training.

## Requirements

The configuration loader requires PyYAML. Install it with:
```bash
pip install PyYAML
```

## Configuration Files

### `expansions.yaml`
Defines the expansion factors to test. The expansion factor determines the number of SAE features:
```
pages = expansion × features
```

Example values:
- `0.5` - Compression (fewer features than input dimension)
- `1` - Same size as input
- `2`, `4`, `8`, `16` - Increasing expansion factors

### `k_values.yaml`
Defines the K values (top-k sparsity) to test for each expansion factor. K determines how many features are active per sample in the TopK SAE architecture.

### `hooking_points.yaml`
Defines the activation hooking points in the FLUX transformer to test. Each entry specifies:
- `loc`: The module path to hook (e.g., `transformer_blocks.0.attn`)
- `stream`: Stream index (0 for image stream/query, 1 for text stream/key)
- `description`: Human-readable description

### `fixed_params.yaml`
Contains fixed training parameters that remain constant across all hyperparameter combinations:
- Architecture settings
- Dataset configuration
- Learning rate and optimization settings
- Training duration (iters, nsamples)
- Output directory

## Usage

The hyperparameter testing script (`scripts/train_small_flux_sae.sh`) automatically loads these configuration files. To modify the hyperparameters, simply edit the corresponding YAML file.

### Example: Adding a new expansion factor

Edit `expansions.yaml`:
```yaml
expansions:
  - 0.5
  - 1
  - 2
  - 4
  - 8
  - 16
  - 32  # Add new value here
```

### Example: Adding a new hooking point

Edit `hooking_points.yaml`:
```yaml
hooking_points:
  # ... existing entries ...
  - loc: "transformer_blocks.9.attn"
    stream: 0
    description: "Transformer block 9 attention (image stream)"
```

### Example: Modifying fixed parameters

Edit `fixed_params.yaml`:
```yaml
# Change training iterations
iters: 2000  # was 1000

# Change batch size
batch_size: 16  # was 8
```

## Total Combinations

The total number of training runs is:
```
num_expansions × num_k_values × num_hooking_points
```

With default values:
- 6 expansions × 6 k values × 8 hooking points = **288 runs**

Each run trains for `iters` steps (default: 1000).

## Manual Loading

To manually load and inspect configurations:

```bash
# Load all configs as bash variables
eval "$(python3 scripts/load_hyperparameter_config.py)"

# Or inspect individual YAML files
cat config/expansions.yaml
cat config/hooking_points.yaml
```

