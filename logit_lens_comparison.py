#!/usr/bin/env python3
"""
Logit Lens style comparison: Original vs SAE-reconstructed activations per layer.

1. Hook multiple layers to capture activations
2. Compare original activations vs SAE-reconstructed (where transformer_blocks.0.attn is replaced)
3. Visualize layer-by-layer differences
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from PIL import Image
from diffusers import FluxPipeline
from einops import rearrange
from autoencoder import TopkSparseAutoencoder
from accelerate import Accelerator
try:
    from scipy.ndimage import zoom
except ImportError:
    zoom = None

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

class MultiLayerHook:
    """Hook to capture activations from multiple layers."""
    def __init__(self, transformer, layer_locations):
        self.transformer = transformer
        self.layer_locations = layer_locations
        self.handles = []
        self.activations = {}
        
    def __enter__(self):
        def make_hook(loc):
            def hook_fn(module, input, output):
                # Store activation for this layer
                if isinstance(output, tuple):
                    # For attention, store both streams
                    self.activations[loc] = output
                else:
                    self.activations[loc] = output
            return hook_fn
        
        # Register hooks for all layers
        for loc in self.layer_locations:
            module = self.transformer.get_submodule(loc)
            handle = module.register_forward_hook(make_hook(loc))
            self.handles.append((loc, handle))
        
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        # Remove all hooks
        for loc, handle in self.handles:
            handle.remove()
        self.activations = {}
        self.handles = []

class SAEReplacementHook:
    """Hook that replaces transformer_blocks.0.attn output with SAE reconstruction."""
    def __init__(self, transformer, sae_model, loc="transformer_blocks.0.attn", stream=1):
        self.transformer = transformer
        self.sae = sae_model
        self.loc = loc
        self.stream = stream
        self.handle = None
        self.original_output = None
        
    def __enter__(self):
        def hook_fn(module, input, output):
            # Store original output
            self.original_output = output
            
            # Extract the stream we're interested in
            if isinstance(output, tuple):
                if self.stream == 0:
                    x, _ = output
                else:
                    _, x = output
            else:
                x = output
            
            # Flatten spatial dimensions
            original_shape = x.shape
            x_flat = rearrange(x, "b ... d -> (b ...) d")
            
            # Encode and decode through SAE
            with torch.no_grad():
                encoded = self.sae.encode(x_flat.to(self.sae.encoder.weight.device))
                reconstructed = self.sae.decode(encoded)
            
            # Reshape back to original shape
            # reconstructed shape is (batch*seq, features)
            # Need to reshape to match original_shape
            if len(original_shape) == 3:  # (batch, seq, features)
                batch_size, seq_len, feat_dim = original_shape
                reconstructed = reconstructed.view(batch_size, seq_len, feat_dim)
            elif len(original_shape) == 2:  # (batch, features)
                batch_size, feat_dim = original_shape
                reconstructed = reconstructed.view(batch_size, feat_dim)
            else:
                # Fallback: try to reshape using original shape
                reconstructed = reconstructed.view(*original_shape)
            
            # CRITICAL: Match the original dtype (bfloat16) to avoid dtype mismatch errors
            original_dtype = x.dtype
            reconstructed = reconstructed.to(dtype=original_dtype)
            # Also ensure it's on the same device
            reconstructed = reconstructed.to(device=x.device)
            
            # Replace the stream in the tuple
            if isinstance(output, tuple):
                if self.stream == 0:
                    return (reconstructed, output[1])
                else:
                    return (output[0], reconstructed)
            else:
                return reconstructed
        
        module = self.transformer.get_submodule(self.loc)
        self.handle = module.register_forward_hook(hook_fn)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle:
            self.handle.remove()
        self.handle = None
        self.original_output = None

def load_model(checkpoint_path):
    """Load the trained SAE model."""
    config_path = Path(checkpoint_path) / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    model = TopkSparseAutoencoder.from_pretrained(checkpoint_path)
    print(f"Loaded SAE model: {checkpoint_path}")
    print(f"  Features: {config['features']}, Pages: {config['pages']}, k: {config['k']}")
    return model, config

def run_logit_lens_comparison(prompt, sae_model, sae_config, output_dir, 
                              num_layers=19, loc="transformer_blocks.0.attn", stream=1):
    """Run logit lens comparison: original vs SAE-reconstructed."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    accelerator = Accelerator()
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe.vae = torch.compile(pipe.vae)
    pipe.text_encoder = torch.compile(pipe.text_encoder)
    pipe.text_encoder_2 = torch.compile(pipe.text_encoder_2)
    
    pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2 = accelerator.prepare(
        pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2
    )
    sae_model = accelerator.prepare(sae_model)
    sae_model.eval()
    
    transformer = pipe.transformer
    
    # Define layers to hook (early to mid layers)
    layer_locations = [f"transformer_blocks.{i}.attn" for i in range(min(num_layers, 19))]
    
    print(f"\nRunning logit lens comparison for prompt: '{prompt}'")
    print(f"Hooking {len(layer_locations)} layers: {layer_locations[:3]}...{layer_locations[-3:]}")
    
    # 1. Original forward pass
    print("\n1. Running original forward pass...")
    with MultiLayerHook(transformer, layer_locations) as hook:
        with torch.no_grad():
            _ = pipe(
                prompt,
                height=256,
                width=256,
                guidance_scale=0.0,
                max_sequence_length=256,
                num_inference_steps=1,
            )
        original_activations = {k: v for k, v in hook.activations.items()}
    
    # 2. SAE-reconstructed forward pass
    print("2. Running SAE-reconstructed forward pass...")
    with SAEReplacementHook(transformer, sae_model, loc=loc, stream=stream) as sae_hook:
        with MultiLayerHook(transformer, layer_locations) as hook:
            with torch.no_grad():
                _ = pipe(
                    prompt,
                    height=256,
                    width=256,
                    guidance_scale=0.0,
                    max_sequence_length=256,
                    num_inference_steps=1,
                )
            sae_activations = {k: v for k, v in hook.activations.items()}
    
    # Process and compare activations
    print("\n3. Processing and comparing activations...")
    
    # Extract and compare activations at each layer
    layer_comparisons = []
    
    for layer_loc in layer_locations:
        orig = original_activations[layer_loc]
        sae = sae_activations[layer_loc]
        
        # Handle tuple outputs (attention)
        if isinstance(orig, tuple):
            orig_act = orig[stream] if stream < len(orig) else orig[0]
            sae_act = sae[stream] if stream < len(sae) else sae[0]
        else:
            orig_act = orig
            sae_act = sae
        
        # Flatten for comparison
        orig_flat = rearrange(orig_act, "b ... d -> (b ...) d").float()
        sae_flat = rearrange(sae_act, "b ... d -> (b ...) d").float()
        
        # Compute metrics
        mse = ((orig_flat - sae_flat) ** 2).mean().item()
        cosine_sim = torch.nn.functional.cosine_similarity(
            orig_flat.mean(dim=0, keepdim=True),
            sae_flat.mean(dim=0, keepdim=True)
        ).item()
        
        # L2 norm difference
        orig_norm = orig_flat.norm(dim=-1).mean().item()
        sae_norm = sae_flat.norm(dim=-1).mean().item()
        norm_diff = abs(orig_norm - sae_norm) / orig_norm * 100 if orig_norm > 0 else 0
        
        layer_comparisons.append({
            'layer': layer_loc,
            'mse': mse,
            'cosine_sim': cosine_sim,
            'norm_diff_percent': norm_diff,
            'orig_norm': orig_norm,
            'sae_norm': sae_norm,
        })
    
    # Visualize - save as three separate images
    print("4. Generating visualizations...")
    
    layers = [comp['layer'] for comp in layer_comparisons]
    layer_indices = [int(l.split('.')[1]) for l in layers]
    mse_values = [comp['mse'] for comp in layer_comparisons]
    cos_sim_values = [comp['cosine_sim'] for comp in layer_comparisons]
    norm_diff_values = [comp['norm_diff_percent'] for comp in layer_comparisons]
    
    # 1. MSE per layer
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    ax1.plot(layer_indices, mse_values, 'o-', color='red', linewidth=2, markersize=6)
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('Mean Squared Error', fontsize=12)
    ax1.set_title(f'Reconstruction Error per Layer\nPrompt: "{prompt[:50]}..." | SAE: {loc} (stream {stream})', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(0, color='green', linestyle='--', linewidth=2, label='SAE Replacement Layer')
    ax1.legend(fontsize=10)
    # Add summary text
    summary_text1 = f'Mean MSE: {np.mean(mse_values):.4f} | Layer 0: {mse_values[0]:.4f} | Layer {layer_indices[-1]}: {mse_values[-1]:.4f}'
    ax1.text(0.02, 0.98, summary_text1, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(output_dir / 'logit_lens_mse.png', bbox_inches='tight', dpi=300)
    print(f"Saved: {output_dir / 'logit_lens_mse.png'}")
    plt.close()
    
    # 2. Cosine Similarity per layer
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    ax2.plot(layer_indices, cos_sim_values, 'o-', color='blue', linewidth=2, markersize=6)
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('Cosine Similarity', fontsize=12)
    ax2.set_title(f'Activation Similarity per Layer\nPrompt: "{prompt[:50]}..." | SAE: {loc} (stream {stream})', 
                  fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)
    ax2.axvline(0, color='green', linestyle='--', linewidth=2, label='SAE Replacement Layer')
    ax2.legend(fontsize=10)
    # Add summary text
    summary_text2 = f'Mean Cosine Sim: {np.mean(cos_sim_values):.4f} | Layer 0: {cos_sim_values[0]:.4f} | Layer {layer_indices[-1]}: {cos_sim_values[-1]:.4f}'
    ax2.text(0.02, 0.98, summary_text2, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    plt.tight_layout()
    plt.savefig(output_dir / 'logit_lens_cosine_similarity.png', bbox_inches='tight', dpi=300)
    print(f"Saved: {output_dir / 'logit_lens_cosine_similarity.png'}")
    plt.close()
    
    # 3. Norm difference per layer
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
    ax3.plot(layer_indices, norm_diff_values, 'o-', color='orange', linewidth=2, markersize=6)
    ax3.set_xlabel('Layer Index', fontsize=12)
    ax3.set_ylabel('Norm Difference (%)', fontsize=12)
    ax3.set_title(f'Activation Magnitude Difference per Layer\nPrompt: "{prompt[:50]}..." | SAE: {loc} (stream {stream})', 
                  fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(0, color='green', linestyle='--', linewidth=2, label='SAE Replacement Layer')
    ax3.legend(fontsize=10)
    # Add summary text
    summary_text3 = f'Mean Norm Diff: {np.mean(norm_diff_values):.2f}% | Layer 0: {norm_diff_values[0]:.2f}% | Layer {layer_indices[-1]}: {norm_diff_values[-1]:.2f}%'
    ax3.text(0.02, 0.98, summary_text3, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    plt.tight_layout()
    plt.savefig(output_dir / 'logit_lens_norm_difference.png', bbox_inches='tight', dpi=300)
    print(f"Saved: {output_dir / 'logit_lens_norm_difference.png'}")
    plt.close()
    
    # Also save the combined version for reference
    fig_combined, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig_combined.suptitle(f'Logit Lens Comparison: Original vs SAE-Reconstructed\nPrompt: "{prompt[:60]}..."', 
                 fontsize=14, fontweight='bold')
    
    axes[0, 0].plot(layer_indices, mse_values, 'o-', color='red', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Layer Index')
    axes[0, 0].set_ylabel('Mean Squared Error')
    axes[0, 0].set_title('Reconstruction Error per Layer')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(0, color='green', linestyle='--', linewidth=2, label='SAE Replacement Layer')
    axes[0, 0].legend()
    
    axes[0, 1].plot(layer_indices, cos_sim_values, 'o-', color='blue', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Layer Index')
    axes[0, 1].set_ylabel('Cosine Similarity')
    axes[0, 1].set_title('Activation Similarity per Layer')
    axes[0, 1].set_ylim([0, 1.05])
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(0, color='green', linestyle='--', linewidth=2, label='SAE Replacement Layer')
    axes[0, 1].legend()
    
    axes[1, 0].plot(layer_indices, norm_diff_values, 'o-', color='orange', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Layer Index')
    axes[1, 0].set_ylabel('Norm Difference (%)')
    axes[1, 0].set_title('Activation Magnitude Difference per Layer')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(0, color='green', linestyle='--', linewidth=2, label='SAE Replacement Layer')
    axes[1, 0].legend()
    
    axes[1, 1].axis('off')
    summary_text = f"""
    LOGIT LENS COMPARISON SUMMARY
    
    Prompt: {prompt[:50]}...
    
    SAE Replacement:
    • Layer: {loc}
    • Stream: {stream}
    
    METRICS:
    • Mean MSE: {np.mean(mse_values):.6f}
    • Mean Cosine Sim: {np.mean(cos_sim_values):.4f}
    • Mean Norm Diff: {np.mean(norm_diff_values):.2f}%
    
    LAYER 0 (SAE replaced):
    • MSE: {mse_values[0]:.6f}
    • Cosine Sim: {cos_sim_values[0]:.4f}
    • Norm Diff: {norm_diff_values[0]:.2f}%
    
    LAYER {layer_indices[-1]} (furthest):
    • MSE: {mse_values[-1]:.6f}
    • Cosine Sim: {cos_sim_values[-1]:.4f}
    • Norm Diff: {norm_diff_values[-1]:.2f}%
    
    INTERPRETATION:
    Lower MSE and higher cosine similarity
    indicate SAE reconstruction preserves
    activation patterns. Differences propagate
    through deeper layers.
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'logit_lens_comparison.png', bbox_inches='tight', dpi=300)
    print(f"Saved: {output_dir / 'logit_lens_comparison.png'}")
    plt.close()
    
    # Save statistics
    stats = {
        'prompt': prompt,
        'sae_location': loc,
        'sae_stream': stream,
        'num_layers': len(layer_comparisons),
        'layer_comparisons': layer_comparisons,
        'mean_mse': float(np.mean(mse_values)),
        'mean_cosine_sim': float(np.mean(cos_sim_values)),
        'mean_norm_diff_percent': float(np.mean(norm_diff_values)),
    }
    
    with open(output_dir / 'logit_lens_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats

def visualize_sae_heatmap_on_image(image_path, sae_model, sae_config, output_dir,
                                   loc="transformer_blocks.0.attn", stream=1, prompt=None):
    """Visualize SAE feature activations as heatmap overlaid on image."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image
    image = Image.open(image_path)
    print(f"\nLoaded image: {image_path}")
    print(f"  Size: {image.size}")
    
    # Use image caption as prompt if not provided
    if prompt is None:
        # Try to extract caption from image filename or use default
        if "cat" in str(image_path).lower():
            prompt = "a cat"
        else:
            prompt = "an image"
    
    accelerator = Accelerator()
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe.vae = torch.compile(pipe.vae)
    pipe.text_encoder = torch.compile(pipe.text_encoder)
    pipe.text_encoder_2 = torch.compile(pipe.text_encoder_2)
    
    pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2 = accelerator.prepare(
        pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2
    )
    sae_model = accelerator.prepare(sae_model)
    sae_model.eval()
    
    # Sample activations for this prompt
    print(f"\nSampling activations for prompt: '{prompt}'")
    from fluxsae import FluxActivationSampler
    
    sampler = FluxActivationSampler("black-forest-labs/FLUX.1-schnell", loc=loc)
    sampler.pipe.transformer, sampler.pipe.vae, sampler.pipe.text_encoder, sampler.pipe.text_encoder_2 = accelerator.prepare(
        sampler.pipe.transformer, sampler.pipe.vae, sampler.pipe.text_encoder, sampler.pipe.text_encoder_2
    )
    
    with torch.no_grad():
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
            
            # Handle tuple output
            if isinstance(activations, tuple):
                if stream == 0:
                    x, _ = activations
                else:
                    _, x = activations
            else:
                x = activations
            
            # Store original shape for spatial mapping
            original_shape = x.shape
            # Flatten
            x_flat = rearrange(x, "b ... d -> (b ...) d")
    
    # Encode through SAE
    print("Encoding through SAE...")
    with torch.no_grad():
        encoded = sae_model.encode(x_flat.to(sae_model.encoder.weight.device))
        encoded = encoded.cpu()
    
    # Store raw activations for spatial mapping
    raw_activations = x_flat.cpu()
    
    # Get top-k features (average across sequence)
    k = sae_config['k']
    avg_encoded = encoded.mean(dim=0)  # Average across sequence positions
    top_k_values, top_k_indices = torch.topk(avg_encoded, k=k)
    
    # Since activations come from sequence positions, we need to map back to spatial locations
    # For now, create a summary visualization
    print("Creating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SAE Feature Activations on Image', fontsize=14, fontweight='bold')
    
    # 1. Original image (top-left)
    ax1 = axes[0, 0]
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    
    # 2. Feature activation bar chart (top-right)
    ax2 = axes[0, 1]
    # Use top-k features (already computed)
    top_k_avg = top_k_values
    top_k_idx_avg = top_k_indices
    
    colors = plt.cm.viridis(top_k_avg.float().numpy() / top_k_avg.max())
    bars = ax2.barh(range(len(top_k_avg)), top_k_avg.float().numpy(), color=colors, edgecolor='black')
    ax2.set_yticks(range(len(top_k_avg)))
    ax2.set_yticklabels([f"Feature {idx.item()}" for idx in top_k_idx_avg])
    ax2.set_xlabel('Average Activation Magnitude')
    ax2.set_title(f'Top-{len(top_k_avg)} Activated Features', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Feature activation heatmap (bottom-left)
    ax3 = axes[1, 0]
    # Show activation pattern across all features
    all_features = torch.zeros(sae_config['pages'])
    all_features[top_k_idx_avg] = top_k_avg.float()
    
    num_show = min(200, sae_config['pages'])
    feature_heatmap = all_features[:num_show].unsqueeze(0).numpy()
    sns.heatmap(feature_heatmap, ax=ax3, cmap='YlOrRd', cbar_kws={'label': 'Activation'},
                xticklabels=50, yticklabels=False)
    ax3.set_xlabel('Feature Index')
    ax3.set_title(f'Feature Activation Pattern\n(Showing first {num_show} features)', 
                  fontsize=12, fontweight='bold')
    
    # 4. Image with feature overlay (bottom-right)
    ax4 = axes[1, 1]
    ax4.imshow(image)
    ax4.axis('off')
    
    # Create feature-based overlay
    # Map feature activations to image using gradient-based approach
    # For now, create a heatmap based on feature strength
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Create heatmap from raw activations (which have spatial structure)
    # Use top SAE feature's decoder direction to project spatial activations
    if len(raw_activations.shape) == 2:  # (seq_len, features)
        seq_len = raw_activations.shape[0]
        sqrt_seq = int(np.sqrt(seq_len))
        if sqrt_seq * sqrt_seq == seq_len:
            # Perfect square - can reshape to spatial grid
            spatial_raw = raw_activations.view(sqrt_seq, sqrt_seq, -1).float()
            # Get top SAE feature's decoder weight
            decoder_weights = sae_model.decoder.weight.data.cpu().T  # (pages, features)
            top_feat_idx = top_k_idx_avg[0].item()
            if top_feat_idx < decoder_weights.shape[0]:
                top_feat_decoder = decoder_weights[top_feat_idx]  # (features,)
                # Project spatial activations onto this feature's decoder direction
                spatial_importance = (spatial_raw * top_feat_decoder.unsqueeze(0).unsqueeze(0)).sum(dim=-1).numpy()
            else:
                spatial_importance = spatial_raw.norm(dim=-1).numpy()
            
            # Resize to image size
            if zoom is not None:
                zoom_factor = (h / sqrt_seq, w / sqrt_seq)
                heatmap = zoom(spatial_importance, zoom_factor, order=1)
            else:
                import torch.nn.functional as F
                spatial_tensor = torch.from_numpy(spatial_importance).unsqueeze(0).unsqueeze(0)
                heatmap_tensor = F.interpolate(spatial_tensor, size=(h, w), mode='bilinear', align_corners=False)
                heatmap = heatmap_tensor.squeeze().numpy()
        else:
            # Not a perfect square - create uniform heatmap based on top feature
            heatmap = np.ones((h, w)) * (top_k_avg[0].item() / top_k_avg.max().item())
    else:
        # Fallback: uniform heatmap
        heatmap = np.ones((h, w)) * (top_k_avg[0].item() / top_k_avg.max().item())
    
    # Normalize heatmap
    if heatmap.max() > 0:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Overlay heatmap on image
    ax4.imshow(heatmap, alpha=0.5, cmap='hot', interpolation='bilinear')
    ax4.set_title(f'SAE Feature Heatmap Overlay\nTop Feature: {top_k_idx_avg[0].item()}', 
                  fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sae_heatmap_on_image.png', bbox_inches='tight', dpi=300)
    print(f"Saved: {output_dir / 'sae_heatmap_on_image.png'}")
    plt.close()
    
    # Also create a more detailed overlay using attention-like visualization
    # Map activations to image regions (conceptual - using attention weights if available)
    # Pass both encoded (SAE features) and raw_activations for spatial mapping
    create_detailed_overlay(image, encoded, raw_activations, sae_model, sae_config, output_dir, k=k)
    
    return top_k_idx_avg, top_k_avg

def create_detailed_overlay(image, encoded, raw_activations, sae_model, sae_config, output_dir, k=16):
    """Create a more detailed overlay showing feature activations mapped to image."""
    # Get top features
    avg_encoded = encoded.mean(dim=0)
    top_k_values, top_k_indices = torch.topk(avg_encoded, k=k)
    
    # Create visualization with feature importance mapped to image
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SAE Feature Activation Analysis on Image', fontsize=14, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].axis('off')
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    
    # Top features
    axes[0, 1].barh(range(k), top_k_values.float().numpy(), edgecolor='black')
    axes[0, 1].set_yticks(range(k))
    axes[0, 1].set_yticklabels([f"Feat {idx.item()}" for idx in top_k_indices])
    axes[0, 1].set_xlabel('Activation Magnitude')
    axes[0, 1].set_title(f'Top-{k} Features', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # Feature distribution
    axes[0, 2].hist(top_k_values.float().numpy(), bins=20, edgecolor='black', alpha=0.7)
    axes[0, 2].set_xlabel('Activation Value')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Feature Activation Distribution', fontsize=12, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Image with heatmap overlay
    axes[1, 0].imshow(image)
    axes[1, 0].axis('off')
    
    # Create spatial heatmap from activations
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Try to create spatial heatmap from raw activations (which have spatial structure)
    # Use raw activations to get spatial mapping, then overlay SAE feature importance
    if len(raw_activations.shape) == 2:
        seq_len = raw_activations.shape[0]
        sqrt_seq = int(np.sqrt(seq_len))
        if sqrt_seq * sqrt_seq == seq_len:
            # Reshape to spatial grid
            spatial_raw = raw_activations.view(sqrt_seq, sqrt_seq, -1).float()
            # Get top SAE feature's decoder weight to see what it represents
            decoder_weights = sae_model.decoder.weight.data.cpu().T  # (pages, features)
            top_feat_idx = top_k_indices[0].item()
            if top_feat_idx < decoder_weights.shape[0]:
                top_feat_decoder = decoder_weights[top_feat_idx]  # (features,)
                # Project spatial activations onto this feature's decoder direction
                # This gives us spatial importance for this feature
                spatial_importance = (spatial_raw * top_feat_decoder.unsqueeze(0).unsqueeze(0)).sum(dim=-1).numpy()
            else:
                # Use average activation magnitude
                spatial_importance = spatial_raw.norm(dim=-1).numpy()
            
            # Resize to image
            if zoom is not None:
                zoom_factor = (h / sqrt_seq, w / sqrt_seq)
                heatmap = zoom(spatial_importance, zoom_factor, order=1)
            else:
                import torch.nn.functional as F
                spatial_tensor = torch.from_numpy(spatial_importance).unsqueeze(0).unsqueeze(0)
                heatmap_tensor = F.interpolate(spatial_tensor, size=(h, w), mode='bilinear', align_corners=False)
                heatmap = heatmap_tensor.squeeze().numpy()
            
            if heatmap.max() > 0:
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            axes[1, 0].imshow(heatmap, alpha=0.5, cmap='hot', interpolation='bilinear')
        else:
            # Fallback
            heatmap = np.ones((h, w)) * (top_k_values[0].item() / top_k_values.max().item())
            axes[1, 0].imshow(heatmap, alpha=0.4, cmap='hot')
    else:
        heatmap = np.ones((h, w)) * (top_k_values[0].item() / top_k_values.max().item())
        axes[1, 0].imshow(heatmap, alpha=0.4, cmap='hot')
    
    axes[1, 0].set_title(f'Feature Heatmap Overlay\nTop Feature: {top_k_indices[0].item()}', 
                         fontsize=12, fontweight='bold')
    
    # Feature importance summary
    axes[1, 1].axis('off')
    summary = f"""
    SAE FEATURE ANALYSIS
    
    Image: cat.jpg
    Total Features: {sae_config['pages']}
    Activated: {k}
    
    TOP 5 FEATURES:
    """
    for i in range(min(5, k)):
        summary += f"  {i+1}. Feature {top_k_indices[i].item()}: {top_k_values[i].item():.2f}\n"
    
    summary += f"""
    ACTIVATION STATS:
    • Mean: {top_k_values.mean().item():.2f}
    • Max: {top_k_values.max().item():.2f}
    • Min: {top_k_values.min().item():.2f}
    """
    
    axes[1, 1].text(0.1, 0.5, summary, fontsize=10, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Feature activation pattern
    all_features = torch.zeros(sae_config['pages'])
    all_features[top_k_indices] = top_k_values.float()
    feature_map = all_features[:100].unsqueeze(0).numpy()
    sns.heatmap(feature_map, ax=axes[1, 2], cmap='YlOrRd', cbar_kws={'label': 'Activation'},
                xticklabels=20, yticklabels=False)
    axes[1, 2].set_xlabel('Feature Index')
    axes[1, 2].set_title('Activation Pattern\n(First 100 features)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sae_detailed_overlay.png', bbox_inches='tight', dpi=300)
    print(f"Saved: {output_dir / 'sae_detailed_overlay.png'}")
    plt.close()

def main():
    """Main execution."""
    checkpoint_path = "./checkpoints/small-flux-sae-test"
    output_dir = "./checkpoints/small-flux-sae-test/visualizations"
    image_path = "/mnt/drive_a/Projects/sae/cat.jpg"
    loc = "transformer_blocks.0.attn"
    stream = 1
    
    # Test prompt
    test_prompt = "a cat sitting on a windowsill"
    
    print("="*80)
    print("Logit Lens Comparison & SAE Feature Visualization")
    print("="*80)
    
    # Load SAE model
    sae_model, sae_config = load_model(checkpoint_path)
    
    # 1. Logit lens comparison
    print("\n" + "="*80)
    print("PART 1: Logit Lens Comparison")
    print("="*80)
    logit_stats = run_logit_lens_comparison(
        test_prompt, sae_model, sae_config, output_dir,
        num_layers=19, loc=loc, stream=stream
    )
    
    # 2. SAE heatmap on image
    print("\n" + "="*80)
    print("PART 2: SAE Feature Visualization on Image")
    print("="*80)
    top_features, top_values = visualize_sae_heatmap_on_image(
        image_path, sae_model, sae_config, output_dir,
        loc=loc, stream=stream, prompt=test_prompt
    )
    
    print("\n✅ Analysis complete!")
    print(f"\nFiles saved to: {output_dir}")
    print("  - logit_lens_mse.png: MSE per layer (separate)")
    print("  - logit_lens_cosine_similarity.png: Cosine similarity per layer (separate)")
    print("  - logit_lens_norm_difference.png: Norm difference per layer (separate)")
    print("  - logit_lens_comparison.png: Combined comparison (all metrics)")
    print("  - logit_lens_stats.json: Comparison statistics")
    print("  - sae_heatmap_on_image.png: SAE features on image")
    print("  - sae_detailed_overlay.png: Detailed feature analysis")

if __name__ == "__main__":
    main()

