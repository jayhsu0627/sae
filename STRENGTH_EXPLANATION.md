# Why Gytis Uses strength=40 and You Should Use Lower Strengths

## Gytis's Example

```python
features = model.encode(embed)
features[22420] = 40  # Set feature 22420 to 40
generate_one(model.decode(features) + get_error(embed))
```

**Key point:** Gytis is working with **CLIP embeddings**, not internal model activations.

## Why strength=40 Works for CLIP Embeddings

### 1. **Different Scale/Scale**

**CLIP Embeddings:**
- CLIP text/image encoders output normalized embeddings
- Feature activations in CLIP SAE are typically in range [0, 10-50]
- The SAE decoder outputs are in the same scale as CLIP embeddings
- Setting a feature to 40 is **within the normal range** of CLIP feature activations

**FLUX Internal Activations:**
- FLUX transformer activations are in a different scale
- **Actual scale needs to be measured** (see `check_activation_scale.py`)
- The scale likely differs from CLIP embeddings
- Using strength=40 would likely be **too high** and break the activations
- **Recommendation**: Measure actual activation values first, then use strengths that are small relative to typical values

### 2. **Different Context**

**Gytis's Approach:**
- Modifies **CLIP embeddings** (conditioning input)
- CLIP embeddings are semantic, interpretable features
- They're designed to be manipulated
- The model receives these as input, so it's more tolerant

**Your Approach:**
- Modifies **internal FLUX activations** (mid-forward-pass)
- Internal activations are more sensitive
- They're part of the computation graph
- Large modifications can break the forward pass

### 3. **Different SAE Architecture**

**Gytis's SAE (Standard):**
- Standard SAE with ReLU
- Keeps all positive features
- Feature activations can naturally be higher
- Setting a feature to 40 might just boost an existing feature

**Your SAE (TopK):**
- TopK SAE with k=16
- Only top-k features are kept
- Feature activations are more constrained
- Setting a feature to 40 when it's not in top-k forces a new activation

## Recommended Strengths

### For CLIP Embeddings (Gytis's case):
- **strength = 10-50** is reasonable
- CLIP embeddings are normalized and can handle larger modifications
- The model is designed to work with CLIP embeddings as input

### For FLUX Internal Activations (Your case):
- **strength = 0.1-5.0** is recommended
- Start with 0.5-2.0 for testing
- Internal activations are more sensitive
- Use Gytis-style steering to only boost active features

## Why Lower Strengths Work Better

1. **Less Reconstruction Error**
   - Lower strengths cause less disruption
   - The modified activations stay closer to the original distribution
   - Less likely to break the forward pass

2. **More Stable Steering**
   - Small changes propagate more naturally
   - The model can still generate coherent images
   - Large changes can cause artifacts or broken generations

3. **Gytis-Style Helps**
   - Only boosting active features (in top-k) is more stable
   - You can use lower strengths because you're not forcing new activations
   - This is why Gytis-style steering works better

## Example Comparison

### Gytis's Code (CLIP):
```python
features = clip_sae.encode(clip_embedding)  # CLIP embedding
features[22420] = 40  # Set to 40 - this is in normal range for CLIP
steered_embedding = clip_sae.decode(features)
# Feed to Kandinsky 2.2 as conditioning
```

### Your Code (FLUX):
```python
# Inside FLUX forward pass
activations = transformer_block.attn(...)  # Internal activations
# Use Gytis-style to boost active features
steered = sae.surgery_gytis_style(activations, k=feature_idx, strength=0.5)
# Continue forward pass with steered activations
```

## Summary

- **Gytis uses strength=40** because CLIP embeddings are in a different scale and context
- **You should use strength=0.5-5.0** for FLUX internal activations
- **Gytis-style steering** allows lower strengths because it only boosts active features
- **Start with 0.5-2.0** and adjust based on results

The key insight: **Different scales, different contexts, different appropriate strengths!**

