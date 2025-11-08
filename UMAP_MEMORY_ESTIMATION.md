# UMAP Memory Estimation for SAE Feature Visualization

## Overview

This document estimates GPU memory requirements for applying UMAP to SAE feature matrices to avoid out-of-memory (OOM) errors.

## Dataset and Model Configurations

Based on the codebase analysis:

### CLIP SAEs
- **Base Features**: 768 (CLIP ViT-Large) or 512 (CLIP ViT-Base)
- **Expansion Factor**: 4x (from available checkpoints)
- **SAE Pages (Features)**: 768 × 4 = **3,072 features**
- **Dataset**: CC3M (Conceptual Captions 3M) ≈ **3,000,000 samples**

### FLUX SAEs  
- **Base Features**: 3,072 (from `fluxsae.py` default)
- **Expansion Factor**: 4x (default)
- **SAE Pages (Features)**: 3,072 × 4 = **12,288 features**
- **Dataset**: Variable, but potentially large-scale

## Memory Calculation

### Feature Matrix Memory

The feature matrix for UMAP has shape: `(num_samples, num_pages)`

#### CLIP SAE Feature Matrix
- **Shape**: (3,000,000 samples × 3,072 features) = 9,216,000,000 elements
- **FP32**: 9,216,000,000 × 4 bytes = **36.9 GB**
- **FP16/BF16**: 9,216,000,000 × 2 bytes = **18.4 GB**

#### FLUX SAE Feature Matrix
- **Shape**: (N samples × 12,288 features)
- For 3M samples: 3,000,000 × 12,288 = 36,864,000,000 elements
- **FP32**: 36,864,000,000 × 4 bytes = **147.5 GB**
- **FP16/BF16**: 36,864,000,000 × 2 bytes = **73.7 GB**

### UMAP Algorithm Overhead

UMAP's internal computations require additional memory:
- **Nearest neighbor graph construction**: ~1-2× input size
- **Embedding optimization**: ~0.5-1× input size
- **Temporary buffers**: ~0.5-1× input size

**Total UMAP overhead**: Approximately **2-3× the input matrix size**

### Total Memory Requirements

#### CLIP SAE (3M samples, 3,072 features)

| Data Type | Feature Matrix | UMAP Overhead (2×) | UMAP Overhead (3×) | **Total (2×)** | **Total (3×)** |
|-----------|---------------|-------------------|-------------------|---------------|---------------|
| FP32      | 36.9 GB       | 73.8 GB          | 110.7 GB         | **110.7 GB**  | **147.6 GB**  |
| FP16/BF16 | 18.4 GB       | 36.8 GB          | 55.2 GB          | **55.2 GB**   | **73.6 GB**   |

#### FLUX SAE (3M samples, 12,288 features)

| Data Type | Feature Matrix | UMAP Overhead (2×) | UMAP Overhead (3×) | **Total (2×)** | **Total (3×)** |
|-----------|---------------|-------------------|-------------------|---------------|---------------|
| FP32      | 147.5 GB      | 295.0 GB         | 442.5 GB         | **442.5 GB**  | **590.0 GB**  |
| FP16/BF16 | 73.7 GB       | 147.4 GB         | 221.1 GB         | **147.4 GB**  | **221.1 GB**  |

## Upper Limit GPU Memory Recommendations

### Minimum Requirements (Conservative Estimate)

| Scenario | GPU Memory Needed |
|----------|------------------|
| **CLIP SAE (FP16)** | **80 GB** (A100 80GB) |
| **CLIP SAE (FP32)** | **160 GB** (2× A100 80GB) |
| **FLUX SAE (FP16)** | **160 GB** (2× A100 80GB) |
| **FLUX SAE (FP32)** | **640 GB** (8× A100 80GB) |

### Recommended GPU Memory (Safe Buffer)

| Scenario | GPU Memory Needed |
|----------|------------------|
| **CLIP SAE (FP16)** | **80 GB** (A100 80GB) ✅ |
| **CLIP SAE (FP32)** | **160 GB** (2× A100 80GB) |
| **FLUX SAE (FP16)** | **240 GB** (3× A100 80GB) |
| **FLUX SAE (FP32)** | **640 GB+** (8× A100 80GB) |

## Memory Optimization Strategies

### 1. Use Lower Precision (FP16/BF16)
- **Reduction**: 2× memory savings
- **Trade-off**: Minimal accuracy loss for visualization
- **Recommendation**: ✅ Use FP16/BF16 for UMAP

### 2. Subsample Dataset
- **Strategy**: Process subset of samples (e.g., 10-50% of dataset)
- **Example**: 300K samples instead of 3M
  - CLIP: 36.9 GB → **3.7 GB** (FP32) or **1.8 GB** (FP16)
  - FLUX: 147.5 GB → **14.8 GB** (FP32) or **7.4 GB** (FP16)
- **Recommendation**: ✅ Use stratified sampling to maintain diversity

### 3. Batch Processing
- **Strategy**: Process UMAP in batches/chunks
- **Implementation**: 
  - Compute embeddings in batches
  - Use incremental UMAP or divide dataset
- **Limitation**: May reduce UMAP quality for very large batches
- **Recommendation**: ✅ For datasets > 1M samples

### 4. Dimensionality Reduction Before UMAP
- **Strategy**: Apply PCA/ICA before UMAP
  - Reduce from 3,072 → 512 or 12,288 → 1024
- **Memory Reduction**: 
  - CLIP: 36.9 GB → **6.1 GB** (FP32, 512 dims)
  - FLUX: 147.5 GB → **24.6 GB** (FP32, 1024 dims)
- **Trade-off**: Some information loss
- **Recommendation**: ✅ For very large feature spaces

### 5. Use GPU-Accelerated UMAP (RAPIDS cuML)
- **Strategy**: Use `cuml.UMAP` instead of `umap.UMAP`
- **Benefits**:
  - Better GPU memory utilization
  - Faster computation
  - Optimized for large-scale data
- **Requirement**: NVIDIA GPU with CUDA support
- **Recommendation**: ✅ Preferred for GPU workflows

### 6. Process on CPU with Large RAM
- **Strategy**: Use CPU-based UMAP with system RAM
- **Requirement**: 128-256 GB system RAM
- **Trade-off**: Slower but avoids GPU memory limits
- **Recommendation**: ⚠️ Fallback option

## Practical Recommendations

### For CLIP SAEs (3,072 features, 3M samples)

**Recommended Setup**:
- ✅ **80 GB GPU** (A100 80GB) with **FP16/BF16**
- ✅ Use **RAPIDS cuML UMAP** for GPU acceleration
- ✅ Consider **subsampling to 500K-1M samples** if memory is tight

**Memory Breakdown**:
- Feature matrix (FP16): 18.4 GB
- UMAP overhead: ~40-60 GB
- Buffer: ~10 GB
- **Total: ~70-80 GB** ✅

### For FLUX SAEs (12,288 features, 3M samples)

**Recommended Setup**:
- ✅ **240 GB GPU memory** (3× A100 80GB) with **FP16/BF16**
- ✅ **Pre-apply PCA** to reduce to 1024-2048 dimensions first
- ✅ **Subsample to 1M samples** if needed
- ✅ Use **RAPIDS cuML UMAP**

**Alternative (More Practical)**:
- ✅ Reduce to **1M samples** + **2048 dims** via PCA
- ✅ Requires: **40-60 GB GPU** (single A100 80GB sufficient)

**Memory Breakdown (Optimized)**:
- Feature matrix (1M × 2048, FP16): 4.1 GB
- UMAP overhead: ~10-15 GB
- PCA intermediate: ~5 GB
- Buffer: ~5 GB
- **Total: ~25-30 GB** ✅ (fits on A100 80GB)

## Code Example for Memory-Efficient UMAP

```python
import torch
from cuml.manifold import UMAP as cuUMAP
from sklearn.decomposition import PCA
import numpy as np

# Load SAE features (assume already encoded)
# features shape: (num_samples, num_pages)
features = load_sae_features()  # e.g., (3_000_000, 3072)

# Strategy 1: Use FP16 and subsample
features = features.half()  # Convert to FP16
indices = torch.randperm(len(features))[:500_000]  # Subsample
features_subset = features[indices].cpu().numpy()

# Strategy 2: Reduce dimensions with PCA first
if features_subset.shape[1] > 1024:
    pca = PCA(n_components=1024)
    features_reduced = pca.fit_transform(features_subset)
else:
    features_reduced = features_subset

# Strategy 3: Use GPU-accelerated UMAP
umap_model = cuUMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)
embeddings = umap_model.fit_transform(features_reduced)
```

## Summary Table

| Configuration | Samples | Features | Data Type | Min GPU Memory | Recommended GPU |
|--------------|---------|----------|-----------|----------------|-----------------|
| CLIP SAE (full) | 3M | 3,072 | FP16 | 80 GB | A100 80GB |
| CLIP SAE (full) | 3M | 3,072 | FP32 | 160 GB | 2× A100 80GB |
| CLIP SAE (subsampled) | 500K | 3,072 | FP16 | 20 GB | A100 40GB / 80GB |
| FLUX SAE (full) | 3M | 12,288 | FP16 | 240 GB | 3× A100 80GB |
| FLUX SAE (optimized) | 1M | 2,048* | FP16 | 30 GB | A100 80GB |

*After PCA dimensionality reduction

## Conclusion

**Upper limit GPU memory to avoid OOM errors**:

1. **Minimum for CLIP SAEs**: **80 GB** (A100 80GB) with FP16
2. **Minimum for FLUX SAEs**: **240 GB** (3× A100 80GB) with FP16, or **30-40 GB** with optimization (subsampling + PCA)

**Best practice**: Always use FP16/BF16, consider subsampling or PCA for large feature spaces, and use RAPIDS cuML for GPU-accelerated UMAP.


