# Surgery Method Comparison: Gytis's CLIP SAE vs Your TopK SAE

## Gytis's CLIP SAE (Standard SAE)

From [gytdau/clip-sae-128](https://huggingface.co/gytdau/clip-sae-128/blob/main/model.py):

```python
def encode(self, x):
    return F.relu(self.encoder(x) + self.enc_bias)

def decode(self, x):
    return self.decoder(x) + self.dec_bias
```

**Key characteristics:**
- **Standard SAE**: Uses ReLU activation, keeps ALL non-negative features
- **No top-k selection**: All features with positive activations are kept
- **Sparse but not top-k**: Sparsity comes from ReLU (many features are zero naturally)

## Your TopK SAE

From your `autoencoder.py`:

```python
def encode(self, x, return_topk: bool = False):
    post_relu_feat_acts_BF = nn.functional.relu(self.encoder(x - self.bias))
    post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)
    
    tops_acts_BK = post_topk.values
    top_indices_BK = post_topk.indices
    
    buffer_BF = torch.zeros_like(post_relu_feat_acts_BF)
    encoded_acts_BF = buffer_BF.scatter_(dim=-1, index=top_indices_BK, src=tops_acts_BK)
    
    return encoded_acts_BF

def decode(self, x):
    return self.decoder(x) + self.bias

def surgery(self, x, k:int, strength:float = 1):
    encoded = self.encode(x)
    
    # increase the kth feature by strength
    offset = torch.zeros_like(encoded)
    offset[..., k] = strength
    
    return self.decode(encoded + offset)
```

**Key characteristics:**
- **TopK SAE**: Uses ReLU + top-k selection, keeps only top-k features
- **Hard sparsity**: Only k features are non-zero (others are explicitly zeroed)
- **Top-k selection**: After ReLU, selects top-k and zeros out the rest

## Critical Difference in Surgery Method

### Gytis's Approach (if he had a surgery method):
```python
def surgery(self, x, k, strength):
    encoded = self.encode(x)  # ALL non-negative features kept
    offset = torch.zeros_like(encoded)
    offset[..., k] = strength
    return self.decode(encoded + offset)
```

**Behavior:**
- Feature `k` might already be non-zero (if it was in the natural top activations)
- Adding `strength` **boosts** an existing feature
- The feature was already "active" in the encoding

### Your Approach:
```python
def surgery(self, x, k, strength):
    encoded = self.encode(x)  # Only top-k features kept, rest are ZERO
    offset = torch.zeros_like(encoded)
    offset[..., k] = strength
    return self.decode(encoded + offset)
```

**Behavior:**
- Feature `k` is likely **ZERO** (if it wasn't in top-k)
- Adding `strength` **activates** a previously inactive feature
- This creates a NEW feature activation that wasn't in the original encoding

## The Problem

### Issue 1: Feature k Might Not Be in Top-K

When you do:
```python
encoded = self.encode(x)  # Only top-k features are non-zero
offset[..., k] = strength  # Adding to feature k
```

If feature `k` was **not** in the top-k features for this input:
- `encoded[..., k] = 0` (it was zeroed out)
- Adding `strength` to it creates: `encoded[..., k] = strength`
- This is **forcing activation** of a feature that wasn't naturally selected

### Issue 2: Reconstruction Error

The surgery method does:
1. Encode: `x → encoded` (top-k sparse representation)
2. Modify: `encoded + offset` (add strength to feature k)
3. Decode: `decode(encoded + offset) → x_steered`

**Problem:** The modified sparse code `encoded + offset` might not be a valid sparse representation:
- If feature k wasn't in top-k, you're adding a feature that wasn't selected
- The decoder expects sparse codes from the encoder's distribution
- Decoding an "unnatural" sparse code can produce artifacts

### Issue 3: Comparison with Gytis

**Gytis's method (if implemented):**
- Feature k is likely already active (standard SAE keeps all positive features)
- Adding strength boosts an existing feature
- Less reconstruction error because the feature was already part of the encoding

**Your method:**
- Feature k is likely zero (top-k only keeps k features)
- Adding strength activates a new feature
- More reconstruction error because you're adding a feature that wasn't selected

## Why This Matters for Steering

### Gytis's Approach (CLIP embeddings):
- CLIP embeddings are semantic and interpretable
- Modifying CLIP embeddings affects the conditioning signal
- Standard SAE surgery: boosts existing features (more stable)

### Your Approach (FLUX activations):
- FLUX activations are internal representations
- Modifying activations mid-forward-pass is more disruptive
- TopK SAE surgery: activates new features (more disruptive)

## Recommendations

### Option 1: Use Additive Method Instead
```python
# Instead of surgery, directly add feature direction
feature_direction = sae.decoder.weight[:, feature_idx]
target_steered = target_flat + strength * feature_direction.unsqueeze(0)
```
- No encode/decode cycle
- No reconstruction error
- Directly adds the feature direction

### Option 2: Fix Surgery to Only Boost Existing Features
```python
def surgery(self, x, k, strength):
    encoded = self.encode(x)
    
    # Only boost if feature k is already active
    if encoded[..., k].sum() > 0:  # Feature is active
        offset = torch.zeros_like(encoded)
        offset[..., k] = strength
        return self.decode(encoded + offset)
    else:
        # Feature not in top-k, use additive method instead
        feature_dir = self.decoder.weight[:, k]
        return x + strength * feature_dir
```

### Option 3: Use Standard SAE Instead of TopK
- Standard SAE keeps all positive features
- Surgery would work more like Gytis's approach
- But you'd lose the explicit top-k control

## Conclusion

**No, they are NOT the same!**

The key difference:
- **Gytis's SAE (standard)**: Surgery boosts existing features
- **Your SAE (top-k)**: Surgery activates new features (if not in top-k)

This is why your surgery method might be causing more reconstruction error and artifacts. The additive method is likely better for your use case.

