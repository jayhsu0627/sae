# CLIP SAE Steering vs FLUX SAE Steering: Key Differences

## Overview

**Gytis's Approach (CLIP SAE → Kandinsky 2.2):**
- Modifies **CLIP embeddings** (conditioning inputs)
- CLIP embeddings are fed into Kandinsky 2.2 as text/image conditioning
- This is **conditioning-level steering** - modifies what the model receives as input

**Your Approach (FLUX SAE → FLUX):**
- Modifies **internal transformer activations** directly
- Hooks into transformer blocks during forward pass
- This is **activation-level steering** - modifies internal representations

## Key Differences & Potential Issues

### 1. **Steering Location (CRITICAL DIFFERENCE)**

**Gytis:**
- Modifies CLIP embeddings **before** they enter the generation model
- CLIP embeddings are the conditioning signal
- Changes propagate naturally through the model

**Your Implementation:**
- Modifies activations **inside** the transformer blocks
- Hooks into `single_transformer_blocks.37.attn` (late layer)
- This is modifying activations mid-forward-pass

**Issue:** Late-layer steering (layer 37) might be too late - the model has already processed most information. Early layers (like `transformer_blocks.0`) might be more effective.

### 2. **Steering Strength (MAJOR ISSUE)**

**Your Code:**
```python
test_strengths = list(range(2, 100, 10))  # [2, 12, 22, 32, 42, 52, 62, 72, 82, 92]
```

**Problem:** These strengths are **extremely high** for activation steering!

- For CLIP embedding steering: typical strengths are 0.1-5.0
- For activation steering: should be even smaller (0.01-2.0)
- Your strengths (2-92) are likely **destroying** the activations

**Recommendation:** Try strengths like `[0.1, 0.5, 1.0, 2.0, 5.0]` instead

### 3. **Steering Method**

**Your Implementation has two methods:**

**a) Additive Method:**
```python
feature_direction = sae.decoder.weight[:, feature_idx]
target_steered = target_flat + strength * feature_direction.unsqueeze(0)
```
- Adds feature direction directly
- Less reconstruction error
- **This is likely better for steering**

**b) Surgery Method:**
```python
target_steered = sae.surgery(target_flat, k=feature_idx, strength=strength)
```
- Does: encode → modify → decode
- Introduces reconstruction error
- The `surgery` method adds `strength` to the encoded feature, then decodes
- **This might introduce artifacts**

**Issue:** You're using `surgery` method by default, which might be causing issues.

### 4. **Hook Placement Timing**

**Your Code:**
```python
hook_location = f"single_transformer_blocks.{block_num}.attn"
```

**Problem:** You're hooking into the **attention output**, but:
- For `single_transformer_blocks`, this might be the final output
- The hook modifies the output, but subsequent layers might not use it properly
- Consider hooking into **input** of the next layer instead

**Gytis's Approach:**
- Modifies CLIP embeddings which are used as **input** to the model
- No hooking needed - just modify the conditioning

### 5. **Stream Selection**

**Your Code:**
```python
stream = 0  # Default fallback to query stream
```

**Issue:** You're defaulting to stream 0 (query), but:
- For `transformer_blocks.0.attn`, stream 1 (key) might be more effective
- The blog author mentioned using stream 1 for `transformer_blocks.0`
- Your autodetection might be wrong for `single_transformer_blocks`

### 6. **Feature Selection**

**Your Code:**
```python
test_features = random.sample(range(49152), 3)  # Random features
```

**Problem:** Random features are unlikely to have interpretable effects!

**Better Approach:**
- First, analyze which features activate for your prompt
- Then steer with top-activating features (you do this later, but should do it first)
- Use features that are known to be interpretable

### 7. **Dtype and Device Handling**

**Your Code:**
```python
# Ensure input dtype matches SAE weights
sae_dtype = next(sae.parameters()).dtype
original_dtype = target_flat.dtype
if target_flat.dtype != sae_dtype:
    target_flat = target_flat.to(sae_dtype)
```

**Good:** You're handling dtype mismatches, but:
- FLUX uses `bfloat16`, SAE might be `float32`
- This conversion might cause precision issues
- Consider keeping everything in the same dtype throughout

### 8. **Reconstruction Error**

**Surgery Method Issue:**
The `surgery` method does:
1. `encode(x)` - gets sparse representation
2. Adds `strength` to feature `k`
3. `decode(encoded + offset)` - reconstructs

**Problem:** This introduces reconstruction error because:
- The modified encoded vector might not be a valid sparse representation
- Decoding a modified sparse code can produce artifacts

**Additive Method is Better:**
- Directly adds the feature direction
- No encode/decode cycle
- Less reconstruction error

## Recommendations

### 1. **Use Additive Method, Not Surgery**
```python
steering_method = "additive"  # Change from "surgery"
```

### 2. **Reduce Steering Strengths Dramatically**
```python
test_strengths = [0.1, 0.5, 1.0, 2.0, 5.0]  # Much smaller!
```

### 3. **Try Earlier Layers**
```python
# Instead of single_transformer_blocks.37
hook_location = "transformer_blocks.0.attn"  # Early layer
stream = 1  # Key stream
```

### 4. **Use Top-Activating Features**
```python
# Run activation analysis FIRST
# Then use top_features_to_test instead of random
```

### 5. **Consider CLIP SAE Steering Instead**
If you want to replicate Gytis's approach:
- Train/load CLIP SAE
- Modify CLIP embeddings before feeding to FLUX
- This might work better than internal activation steering

## Why Gytis's Approach Works Better

1. **Conditioning-level steering** is more stable than activation-level
2. **CLIP embeddings** are designed to be semantic and interpretable
3. **No reconstruction error** - just modify the input conditioning
4. **Lower strengths needed** - conditioning is more sensitive
5. **Works across models** - CLIP embeddings are model-agnostic

## Why Your Approach Might Not Work

1. **FLUX is a flow matching model** - less sensitive to activation steering
2. **Late-layer steering** - too late in the process
3. **Too high strengths** - destroying activation structure
4. **Reconstruction error** from surgery method
5. **Internal activations** are less interpretable than CLIP embeddings

