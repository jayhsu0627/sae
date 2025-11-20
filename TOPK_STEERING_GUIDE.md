# TopK SAE Steering Guide: Finding Active Features & Gytis-Style Steering

## Understanding TopK SAE (k=16)

With `k=16`, your TopK SAE:
- **Dynamically selects** the top 16 features for each input
- Different inputs will have **different sets of 16 active features**
- The 16 features are selected based on highest activation values after ReLU

## Question 1: How to Find Active Features?

### Method 1: Use `get_active_features()` (New Method)

```python
# For a single input
x = activations_flat  # Shape: (batch*seq, features)
active_indices, active_values = sae.get_active_features(x)

# active_indices shape: (batch*seq, 16) - the 16 feature indices
# active_values shape: (batch*seq, 16) - their activation values

# Example: Check which features are active for first token
print(f"Active features for first token: {active_indices[0]}")
print(f"Activation values: {active_values[0]}")
```

### Method 2: Use `encode(return_topk=True)`

```python
encoded, active_values, active_indices = sae.encode(x, return_topk=True)
# Same as get_active_features(), but returns encoded too
```

### Method 3: Check Non-Zero Features in Encoded Output

```python
encoded = sae.encode(x)  # Shape: (batch*seq, pages)
# Only top-k features are non-zero

# Find which features are active (non-zero)
active_mask = encoded > 0  # Shape: (batch*seq, pages)
active_indices = torch.nonzero(active_mask, as_tuple=False)
```

### Method 4: Aggregate Across Multiple Inputs

```python
# Find features that activate frequently across a dataset
all_activations = []
for batch in dataloader:
    encoded = sae.encode(batch)
    all_activations.append(encoded)

# Stack and find frequently active features
all_encoded = torch.cat(all_activations, dim=0)  # Shape: (total_samples, pages)
feature_frequencies = (all_encoded > 0).float().mean(dim=0)  # Shape: (pages,)

# Top 100 most frequently active features
top_frequent_features = torch.topk(feature_frequencies, k=100).indices
```

## Question 2: Can We Only Steer the 16 Active Features?

**Short answer: No, but you should prefer active features.**

### Current Behavior:
- You can steer **any feature** (0 to pages-1)
- If you steer a feature that's **not in top-k**, it creates a new activation
- This is more disruptive and causes reconstruction error

### Best Practice:
1. **First, find active features** for your input
2. **Steer only active features** (Gytis-style)
3. This boosts existing features rather than creating new ones

## Question 3: Gytis-Style Steering with TopK

### The Problem with Current `surgery()`:

```python
def surgery(self, x, k, strength):
    encoded = self.encode(x)  # Only top-k features non-zero
    offset[..., k] = strength  # Adds to feature k
    return self.decode(encoded + offset)
```

**Issue:** If feature `k` is not in top-k, `encoded[..., k] = 0`, so you're forcing activation of a new feature.

### Solution: New `surgery_gytis_style()` Method

This method:
1. **Checks if feature k is active** (in top-k)
2. **If active**: Boosts it (Gytis-style)
3. **If not active**: Uses additive method or skips it

```python
def surgery_gytis_style(self, x, k, strength, fallback_to_additive=True):
    # Get active features
    encoded, active_values, active_indices = self.encode(x, return_topk=True)
    
    # Check if feature k is in active set
    is_active = (active_indices == k).any(dim=-1)
    
    if is_active.all():
        # Feature is active - boost it (Gytis-style)
        offset = torch.zeros_like(encoded)
        offset[..., k] = strength
        return self.decode(encoded + offset)
    elif fallback_to_additive:
        # Feature not active - use additive method
        feature_direction = self.decoder.weight[:, k]
        return x + strength * feature_direction.unsqueeze(0)
    # ... handles mixed cases too
```

## Usage Examples

### Example 1: Find and Steer Active Features

```python
# 1. Get activations for your prompt
activations_flat = rearrange(activation_hook_output, "b ... d -> (b ...) d")

# 2. Find active features
active_indices, active_values = sae.get_active_features(activations_flat)

# 3. Get unique active features across all tokens
unique_active = torch.unique(active_indices.flatten())

print(f"Found {len(unique_active)} unique active features")
print(f"Active feature indices: {unique_active[:10]}")  # First 10

# 4. Steer only active features (Gytis-style)
for feat_idx in unique_active[:5]:  # Test first 5 active features
    steered = sae.surgery_gytis_style(activations_flat, k=feat_idx, strength=1.0)
    # Use steered activations...
```

### Example 2: Steer Top-Activating Features

```python
# 1. Encode to get activations
encoded = sae.encode(activations_flat)

# 2. Find features with highest total activation
feature_totals = encoded.abs().sum(dim=0)  # Sum across all tokens
top_features = torch.topk(feature_totals, k=10).indices

# 3. Steer top features (they're likely active)
for feat_idx in top_features:
    steered = sae.surgery_gytis_style(activations_flat, k=feat_idx, strength=0.5)
```

### Example 3: Update Your `toy.ipynb` Hook

```python
def create_surgery_hook(sae, feature_idx, strength, stream=0, method="gytis_style"):
    """
    Create a hook with Gytis-style steering.
    """
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            query, key = output
            target = query if stream == 0 else key
            other = key if stream == 0 else query
            
            original_shape = target.shape
            target_flat = rearrange(target, "b ... d -> (b ...) d")
            
            # Ensure dtype/device match
            sae_dtype = next(sae.parameters()).dtype
            sae_device = next(sae.parameters()).device
            target_flat = target_flat.to(dtype=sae_dtype, device=sae_device)
            
            # Use Gytis-style surgery
            with torch.no_grad():
                if method == "gytis_style":
                    target_steered = sae.surgery_gytis_style(
                        target_flat, 
                        k=feature_idx, 
                        strength=strength,
                        fallback_to_additive=True
                    )
                elif method == "additive":
                    feature_direction = sae.decoder.weight[:, feature_idx]
                    target_steered = target_flat + strength * feature_direction.unsqueeze(0)
                else:  # original surgery
                    target_steered = sae.surgery(target_flat, k=feature_idx, strength=strength)
            
            # Convert back
            target_steered = target_steered.to(dtype=target.dtype, device=target.device)
            target_reshaped = target_steered.reshape(original_shape)
            
            return (target_reshaped, other) if stream == 0 else (other, target_reshaped)
        else:
            # Handle non-tuple output similarly
            ...
    
    return hook_fn
```

## Comparison: Three Steering Methods

### 1. Original `surgery()` (Current)
- **Always adds** strength to feature k
- **Problem**: If k not in top-k, forces new activation
- **Reconstruction error**: High if k not active

### 2. `surgery_gytis_style()` (New)
- **Only boosts** if feature k is active
- **Falls back** to additive if not active
- **Reconstruction error**: Lower, more stable

### 3. `additive` (Direct)
- **Always adds** feature direction
- **No encode/decode** cycle
- **Reconstruction error**: Lowest, but less "sparse"

## Recommendations

1. **Use `surgery_gytis_style()`** for steering
   - More stable than original surgery
   - Only modifies active features (Gytis-style)
   - Falls back gracefully if feature not active

2. **Find active features first**
   ```python
   active_indices, _ = sae.get_active_features(activations)
   # Steer only features in active_indices
   ```

3. **Use lower strengths**
   - For Gytis-style: `strength = 0.5 - 2.0`
   - Original surgery needed higher strengths because it was less effective

4. **Test with top-activating features**
   - Features that activate strongly are more interpretable
   - They're also more likely to be in top-k

## Summary

- **k=16 means**: Top 16 features selected per input (dynamically)
- **Active features**: Use `get_active_features()` or `encode(return_topk=True)`
- **Steering**: Prefer active features, use `surgery_gytis_style()`
- **Gytis-style**: Only boosts existing features, more stable

