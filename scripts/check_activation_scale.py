"""
Script to check the actual scale of FLUX activations.
This will help us understand what strength values are appropriate.
"""
import sys
from pathlib import Path

# Add parent directory to path to allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from einops import rearrange

from fluxsae import FluxActivationSampler

# Test prompts
test_prompts = [
    "a cat sitting on a mat",
    "a beautiful landscape with mountains",
    "a woman with pink hair standing in a forest",
]

# Sample activations from different locations
locations = [
    "transformer_blocks.0.attn",
    "single_transformer_blocks.9.attn",
    "single_transformer_blocks.37.attn",
]

print("=" * 60)
print("FLUX Activation Scale Analysis")
print("=" * 60)

for loc in locations:
    print(f"\n📍 Location: {loc}")
    sampler = FluxActivationSampler("black-forest-labs/FLUX.1-schnell", loc=loc)
    sampler.pipe = sampler.pipe.to("cuda")
    
    all_activations = []
    
    with torch.no_grad():
        for prompt in test_prompts:
            with sampler as s:
                outputs = s(
                    prompt,
                    height=256,
                    width=256,
                    guidance_scale=0.0,
                    max_sequence_length=256,
                    num_inference_steps=1,
                )
                activations = outputs["activations"]
                
                # Handle tuple output (query, key)
                if isinstance(activations, tuple):
                    # Use key stream (stream=1)
                    _, x = activations
                else:
                    x = activations
                
                # Flatten
                x_flat = rearrange(x, "b ... d -> (b ...) d")
                all_activations.append(x_flat.cpu())
    
    # Concatenate all activations
    all_activations = torch.cat(all_activations, dim=0)
    
    # Convert to float32 for quantile computation (quantile requires float or double)
    all_activations_float = all_activations.float()
    
    # Compute statistics
    mean_val = all_activations_float.mean().item()
    std_val = all_activations_float.std().item()
    min_val = all_activations_float.min().item()
    max_val = all_activations_float.max().item()
    
    # Compute percentiles
    p25 = torch.quantile(all_activations_float, 0.25).item()
    p50 = torch.quantile(all_activations_float, 0.50).item()
    p75 = torch.quantile(all_activations_float, 0.75).item()
    p95 = torch.quantile(all_activations_float, 0.95).item()
    p99 = torch.quantile(all_activations_float, 0.99).item()
    
    print(f"  Shape: {all_activations_float.shape}")
    print(f"  Mean: {mean_val:.4f}")
    print(f"  Std:  {std_val:.4f}")
    print(f"  Min:  {min_val:.4f}")
    print(f"  Max:  {max_val:.4f}")
    print(f"  Percentiles:")
    print(f"    P25: {p25:.4f}")
    print(f"    P50: {p50:.4f}")
    print(f"    P75: {p75:.4f}")
    print(f"    P95: {p95:.4f}")
    print(f"    P99: {p99:.4f}")
    
    # Check if activations are mostly positive or can be negative
    negative_ratio = (all_activations_float < 0).float().mean().item() * 100
    print(f"  Negative values: {negative_ratio:.2f}%")
    
    # Compute L2 norm per activation vector
    norms = all_activations_float.norm(dim=-1)
    print(f"  L2 norm per vector:")
    print(f"    Mean: {norms.mean().item():.4f}")
    print(f"    Std:  {norms.std().item():.4f}")
    print(f"    Max:  {norms.max().item():.4f}")

print("\n" + "=" * 60)
print("Recommendations:")
print("=" * 60)
print("Based on the statistics above, you can determine:")
print("1. Typical activation magnitude range")
print("2. Appropriate strength values for steering")
print("3. Whether activations are normalized or not")
print("\nFor steering, strength should be:")
print("  - Small relative to typical activation values")
print("  - Start with 0.1-1.0 and adjust based on results")

