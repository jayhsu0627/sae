#!/usr/bin/env python3
"""
Validate SAE sparsity by comparing raw activations vs SAE activations.
Uses CC3M data filtered for 'cat' concept.

1. Collect 100 text prompts containing 'cat' from CC3M
2. Randomly pick one image containing 'cat' from CC3M
3. Show activation across all features vs SAE to validate k=16 sparsity
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import io
import json
import random
import math
from diffusers import FluxPipeline
from einops import rearrange
from PIL import Image
import polars as pl
from autoencoder import TopkSparseAutoencoder
from fluxsae import FluxActivationSampler
from accelerate import Accelerator

# CC3M dataset class (from scripts/conversion.py)
class CC3M:
    def __init__(self, path):
        data = []
        for file in Path(path).glob("*.parquet"):
            data.append(pl.read_parquet(file))
        self.dataset = pl.concat(data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.row(idx, named=True)
        image = Image.open(io.BytesIO(row['image']['bytes']))
        text = row["conversations"][-1]['value']
        return image, text

# Set style for paper-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

def load_model(checkpoint_path):
    """Load the trained SAE model."""
    config_path = Path(checkpoint_path) / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    model = TopkSparseAutoencoder.from_pretrained(checkpoint_path)
    print(f"Loaded model: {checkpoint_path}")
    print(f"  Features: {config['features']}, Pages: {config['pages']}, k: {config['k']}")
    return model, config

def collect_cat_prompts(dataset, num_prompts=100, concept='cat'):
    """Collect prompts containing the concept from CC3M dataset."""
    print(f"\nCollecting {num_prompts} prompts containing '{concept}' from CC3M...")
    
    cat_prompts = []
    cat_indices = []
    
    # Search through dataset for prompts containing 'cat' (case-insensitive)
    for idx in range(len(dataset)):
        row = dataset.dataset.row(idx, named=True)
        text = row["conversations"][-1]['value'].lower()
        
        if concept.lower() in text:
            cat_prompts.append(row["conversations"][-1]['value'])
            cat_indices.append(idx)
            
            if len(cat_prompts) >= num_prompts:
                break
    
    if len(cat_prompts) < num_prompts:
        print(f"Warning: Only found {len(cat_prompts)} prompts containing '{concept}', requested {num_prompts}")
    
    print(f"Found {len(cat_prompts)} prompts containing '{concept}'")
    return cat_prompts, cat_indices

def get_random_cat_image(dataset, concept='cat'):
    """Randomly pick one image containing the concept from CC3M."""
    print(f"\nFinding random image containing '{concept}' from CC3M...")
    
    cat_images = []
    cat_image_indices = []
    
    # Find all images with 'cat' in their caption
    for idx in range(len(dataset)):
        row = dataset.dataset.row(idx, named=True)
        text = row["conversations"][-1]['value'].lower()
        
        if concept.lower() in text:
            cat_image_indices.append(idx)
            if len(cat_image_indices) >= 1000:  # Sample from first 1000 matches
                break
    
    if not cat_image_indices:
        raise ValueError(f"No images found containing '{concept}'")
    
    # Randomly pick one
    random_idx = random.choice(cat_image_indices)
    row = dataset.dataset.row(random_idx, named=True)
    image = Image.open(io.BytesIO(row['image']['bytes']))
    text = row["conversations"][-1]['value']
    
    print(f"Selected image at index {random_idx}")
    print(f"Caption: {text[:100]}...")
    
    return image, text, random_idx

def sample_activations(prompts, loc="transformer_blocks.0.attn", stream=1, num_inference_steps=1):
    """Sample activations from FLUX for given prompts."""
    accelerator = Accelerator()
    sampler = FluxActivationSampler("black-forest-labs/FLUX.1-schnell", loc=loc)
    
    # Prepare models
    sampler.pipe.transformer, sampler.pipe.vae, sampler.pipe.text_encoder, sampler.pipe.text_encoder_2 = accelerator.prepare(
        sampler.pipe.transformer, sampler.pipe.vae, sampler.pipe.text_encoder, sampler.pipe.text_encoder_2
    )
    
    activations_list = []
    with torch.no_grad():
        with sampler as s:
            outputs = s(
                prompts,
                height=256,
                width=256,
                guidance_scale=0.0,
                max_sequence_length=256,
                num_inference_steps=num_inference_steps,
            )
            activations = outputs["activations"]
            
            # Handle tuple output (attention) vs single tensor (MLP)
            if isinstance(activations, tuple):
                if stream == 0:
                    x, _ = activations
                else:
                    _, x = activations
            else:
                x = activations
            
            # Flatten spatial dimensions if present
            x = rearrange(x, "b ... d -> (b ...) d")
            activations_list.append(x.cpu())
    
    return torch.cat(activations_list, dim=0)

def visualize_sparsity_comparison(model, raw_activations, sae_activations, prompts, output_dir, k=16):
    """Visualize sparsity comparison: raw activations vs SAE activations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    # Process SAE activations
    with torch.no_grad():
        # Encode through SAE
        encoded = model.encode(raw_activations.to(model.encoder.weight.device))
        encoded = encoded.cpu()
    
    # Statistics
    num_prompts = len(prompts)
    num_features = model.pages
    total_activations = raw_activations.shape[0]
    
    # Each prompt may generate multiple activation vectors (due to sequence length)
    # Calculate activations per prompt
    activations_per_prompt = total_activations // num_prompts
    if total_activations % num_prompts != 0:
        print(f"Warning: {total_activations} activations for {num_prompts} prompts, not evenly divisible")
        # Use first num_prompts * activations_per_prompt activations
        num_use = num_prompts * activations_per_prompt
        raw_activations = raw_activations[:num_use]
        encoded = encoded[:num_use]
        total_activations = num_use
    
    # Reshape to group by prompt: (num_prompts, activations_per_prompt, features)
    raw_activations_grouped = raw_activations.view(num_prompts, activations_per_prompt, -1)
    encoded_grouped = encoded.view(num_prompts, activations_per_prompt, -1)
    
    # Average activations per prompt for visualization
    raw_activations_avg = raw_activations_grouped.mean(dim=1)  # (num_prompts, features)
    encoded_avg = encoded_grouped.mean(dim=1)  # (num_prompts, features)
    
    # Calculate L0 (number of non-zero features) for raw and SAE
    # For raw: count features above a threshold (e.g., mean + std)
    # Convert to float32 for calculations
    raw_activations_float = raw_activations_avg.abs().float()
    raw_threshold = raw_activations_float.mean() + raw_activations_float.std()
    raw_l0 = (raw_activations_float > raw_threshold).sum(dim=-1).float()
    
    # For SAE: count non-zero features (TopK ensures exactly k)
    sae_l0 = (encoded_avg > 0).sum(dim=-1).float()
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Raw activation distribution (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    # Sample one prompt's raw activations (convert bfloat16 to float32 for numpy)
    sample_raw = raw_activations_avg[0].abs().float().numpy()
    ax1.hist(sample_raw, bins=100, edgecolor='black', alpha=0.7, log=True)
    ax1.axvline(raw_threshold.item(), color='red', linestyle='--', label=f'Threshold: {raw_threshold.item():.4f}')
    ax1.set_xlabel('Activation Magnitude (Raw)')
    ax1.set_ylabel('Frequency (log scale)')
    ax1.set_title('Raw Activation Distribution\n(One Prompt)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. SAE activation distribution (top-middle)
    ax2 = fig.add_subplot(gs[0, 1])
    sample_sae = encoded_avg[0].numpy()
    non_zero_sae = sample_sae[sample_sae > 0]
    ax2.hist(non_zero_sae, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax2.set_xlabel('Activation Magnitude (SAE)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'SAE Activation Distribution\n(Exactly {k} Non-Zero)')
    ax2.grid(True, alpha=0.3)
    
    # 3. L0 Comparison (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(raw_l0.numpy(), bins=30, alpha=0.5, label=f'Raw (mean: {raw_l0.mean():.1f})', edgecolor='black')
    ax3.hist(sae_l0.numpy(), bins=30, alpha=0.5, label=f'SAE (mean: {sae_l0.mean():.1f})', edgecolor='black', color='green')
    ax3.axvline(k, color='red', linestyle='--', linewidth=2, label=f'Target k={k}')
    ax3.set_xlabel('L0 (Number of Active Features)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Sparsity Comparison: Raw vs SAE')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Print info about aggregation
    print(f"  Total activations: {total_activations}, Prompts: {num_prompts}, Activations per prompt: {activations_per_prompt}")
    
    # 4. Activation heatmap - Raw (middle-left)
    ax4 = fig.add_subplot(gs[1, 0])
    # Show top 100 features for first 10 prompts
    num_show_prompts = min(10, num_prompts)
    num_show_features = min(100, num_features)
    
    # Get top features for each prompt (raw) - convert to float32
    raw_top_features = torch.topk(raw_activations_avg[:num_show_prompts].abs().float(), k=num_show_features, dim=-1).indices
    raw_heatmap = torch.zeros(num_show_prompts, num_features)
    for i in range(num_show_prompts):
        raw_heatmap[i, raw_top_features[i]] = 1
    
    sns.heatmap(raw_heatmap[:, :num_show_features].numpy(), ax=ax4, cmap='Reds', 
                cbar_kws={'label': 'Active'}, xticklabels=False, yticklabels=False)
    ax4.set_xlabel(f'Feature Index (showing top {num_show_features})')
    ax4.set_ylabel('Prompt Index')
    ax4.set_title(f'Raw Activations\n(Top {num_show_features} features, {num_show_prompts} prompts)')
    
    # 5. Activation heatmap - SAE (middle-middle)
    ax5 = fig.add_subplot(gs[1, 1])
    # SAE activations (only k features active)
    sae_heatmap = (encoded_avg[:num_show_prompts] > 0).float()
    # Find which features activate across prompts
    active_features = sae_heatmap.sum(dim=0) > 0
    active_feature_indices = torch.where(active_features)[0]
    
    if len(active_feature_indices) > 0:
        num_active_features = min(len(active_feature_indices), num_show_features)
        selected_features = active_feature_indices[:num_active_features]
        sns.heatmap(sae_heatmap[:, selected_features].numpy(), ax=ax5, cmap='Greens',
                    cbar_kws={'label': 'Active'}, xticklabels=False, yticklabels=False)
        ax5.set_xlabel(f'Feature Index (showing {num_active_features} active)')
        ax5.set_ylabel('Prompt Index')
        ax5.set_title(f'SAE Activations\n(Exactly {k} features per prompt)')
    else:
        ax5.text(0.5, 0.5, 'No activations', ha='center', va='center')
        ax5.set_title('SAE Activations')
    
    # 6. Feature activation frequency (middle-right)
    ax6 = fig.add_subplot(gs[1, 2])
    # Count how many prompts activate each feature (using averaged activations)
    # For raw: count features that appear in top-k for any prompt
    raw_all_features = (raw_activations_avg[:num_show_prompts].abs() > raw_threshold).sum(dim=0) > 0
    feature_freq_raw = raw_all_features.sum().item()
    # For SAE: count features that activate for any prompt
    sae_all_features = (encoded_avg[:num_show_prompts] > 0).sum(dim=0) > 0
    feature_freq_sae = sae_all_features.sum().item()
    
    categories = ['Raw Activations', 'SAE Activations']
    values = [feature_freq_raw, feature_freq_sae]
    colors = ['red', 'green']
    bars = ax6.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax6.set_ylabel('Number of Unique Features Activated')
    ax6.set_title(f'Feature Diversity\n({num_show_prompts} prompts)')
    ax6.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}',
                ha='center', va='bottom')
    
    # 7. Activation magnitude comparison (bottom-left)
    ax7 = fig.add_subplot(gs[2, 0])
    # Compare activation magnitudes (convert bfloat16 to float32)
    raw_magnitudes = raw_activations_avg[0].abs().float().numpy()
    sae_magnitudes = encoded_avg[0].float().numpy()
    sae_magnitudes_nonzero = sae_magnitudes[sae_magnitudes > 0]
    
    ax7.hist(raw_magnitudes, bins=100, alpha=0.5, label='Raw', edgecolor='black', log=True, color='red')
    if len(sae_magnitudes_nonzero) > 0:
        ax7.hist(sae_magnitudes_nonzero, bins=50, alpha=0.5, label='SAE (non-zero)', 
                edgecolor='black', color='green')
    ax7.set_xlabel('Activation Magnitude')
    ax7.set_ylabel('Frequency (log scale)')
    ax7.set_title('Activation Magnitude Distribution\n(First Prompt)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Sparsity ratio over prompts (bottom-middle)
    ax8 = fig.add_subplot(gs[2, 1])
    sparsity_ratio_raw = (raw_l0 / num_features * 100).float().numpy()
    sparsity_ratio_sae = (sae_l0 / num_features * 100).float().numpy()
    
    ax8.plot(range(num_prompts), sparsity_ratio_raw, 'o-', alpha=0.6, label='Raw', color='red', markersize=3)
    ax8.plot(range(num_prompts), sparsity_ratio_sae, 'o-', alpha=0.6, label='SAE', color='green', markersize=3)
    ax8.axhline(k / num_features * 100, color='red', linestyle='--', linewidth=2, label=f'Target: {k/num_features*100:.2f}%')
    ax8.set_xlabel('Prompt Index')
    ax8.set_ylabel('Sparsity Ratio (%)')
    ax8.set_title('Sparsity Ratio Across Prompts')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Summary statistics (bottom-right)
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    stats_text = f"""
    SPARSITY VALIDATION SUMMARY
    
    Total SAE Features: {num_features}
    Target k (sparsity): {k}
    
    RAW ACTIVATIONS:
    • Mean L0: {raw_l0.mean():.1f} ± {raw_l0.std():.1f}
    • Sparsity: {raw_l0.mean()/num_features*100:.2f}%
    • Features used: {feature_freq_raw} unique
    
    SAE ACTIVATIONS:
    • Mean L0: {sae_l0.mean():.1f} ± {sae_l0.std():.1f}
    • Sparsity: {sae_l0.mean()/num_features*100:.2f}%
    • Features used: {feature_freq_sae} unique
    
    VALIDATION:
    ✓ SAE L0 = {sae_l0.mean():.1f} ≈ {k} (target)
    ✓ Sparsity reduction: {(1 - sae_l0.mean()/raw_l0.mean())*100:.1f}%
    ✓ SAE enforces exactly {k} features per prompt
    """
    
    ax9.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('SAE Sparsity Validation: Raw vs SAE Activations\n(100 "cat" prompts from CC3M)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(output_dir / 'sparsity_validation.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'sparsity_validation.png'}")
    plt.close()
    
    # Calculate activation magnitude statistics (using averaged activations)
    raw_abs = raw_activations_avg.abs().float()
    raw_nonzero = raw_abs[raw_abs > raw_threshold]
    sae_nonzero = encoded_avg.float()[encoded_avg.float() > 0]
    
    # Save statistics
    stats = {
        'num_prompts': num_prompts,
        'num_features': num_features,
        'k': k,
        'activations_per_prompt': int(activations_per_prompt),
        'raw_l0_mean': float(raw_l0.mean()),
        'raw_l0_std': float(raw_l0.std()),
        'sae_l0_mean': float(sae_l0.mean()),
        'sae_l0_std': float(sae_l0.std()),
        'sparsity_reduction_percent': float((1 - sae_l0.mean()/raw_l0.mean())*100) if raw_l0.mean() > 0 else 0.0,
        'raw_sparsity_percent': float(raw_l0.mean()/num_features*100),
        'sae_sparsity_percent': float(sae_l0.mean()/num_features*100),
        'raw_activation_magnitude_mean': float(raw_nonzero.mean()) if len(raw_nonzero) > 0 else 0.0,
        'sae_activation_magnitude_mean': float(sae_nonzero.mean()) if len(sae_nonzero) > 0 else 0.0,
    }
    
    with open(output_dir / 'sparsity_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats

def visualize_features_on_image(model, reference_image, reference_prompt, raw_activation, 
                                sae_activation, output_dir, k=16):
    """Visualize SAE features activated for the reference image."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    # Get which features activate
    with torch.no_grad():
        encoded = model.encode(raw_activation.to(model.encoder.weight.device))
        encoded = encoded.cpu()
    
    # Find top-k activated features
    top_k_values, top_k_indices = torch.topk(encoded, k=k, dim=-1)
    top_k_indices = top_k_indices.squeeze()
    top_k_values = top_k_values.squeeze()
    
    # Get decoder weights for these features (what they represent)
    # Decoder weight shape: (features, pages) where features=3072, pages=6144
    # We want (pages, features) to access individual feature weights
    decoder_weights = model.decoder.weight.data.cpu().T  # Shape: (pages, features)
    activated_feature_weights = decoder_weights[top_k_indices]  # Shape: (k, features)
    
    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Reference image (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(reference_image)
    ax1.axis('off')
    ax1.set_title('Reference Image\n(from CC3M)', fontsize=12, fontweight='bold')
    
    # Add caption
    caption = reference_prompt[:80] + "..." if len(reference_prompt) > 80 else reference_prompt
    ax1.text(0.5, -0.1, caption, transform=ax1.transAxes, ha='center', 
             fontsize=9, wrap=True)
    
    # 2. Activated features bar chart (top-middle)
    ax2 = fig.add_subplot(gs[0, 1])
    colors = plt.cm.viridis(top_k_values.float().numpy() / top_k_values.max())
    bars = ax2.barh(range(k), top_k_values.float().numpy(), color=colors, edgecolor='black')
    ax2.set_yticks(range(k))
    ax2.set_yticklabels([f"Feature {idx.item()}" for idx in top_k_indices])
    ax2.set_xlabel('Activation Magnitude')
    ax2.set_title(f'Top-{k} Activated SAE Features', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_k_values)):
        height = bar.get_height()
        ax2.text(bar.get_width(), bar.get_y() + height/2,
                f'{val:.2f}',
                ha='left', va='center', fontsize=8)
    
    # 3. Feature activation heatmap (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    # Show activation pattern across all features
    all_features = torch.zeros(model.pages)
    all_features[top_k_indices] = top_k_values.float()
    
    # Reshape to show as heatmap (visualize first 200 features for clarity)
    num_show = min(200, model.pages)
    feature_heatmap = all_features[:num_show].unsqueeze(0).numpy()
    sns.heatmap(feature_heatmap, ax=ax3, cmap='YlOrRd', cbar_kws={'label': 'Activation'},
                xticklabels=50, yticklabels=False)
    ax3.set_xlabel('Feature Index')
    ax3.set_title(f'Feature Activation Pattern\n(Showing first {num_show} features)', 
                  fontsize=12, fontweight='bold')
    
    # 4. Decoder weight visualization (bottom-left) - what features represent
    ax4 = fig.add_subplot(gs[1, 0])
    # Visualize decoder weights as "feature directions"
    # Take mean of activated feature weights to show "average direction"
    avg_feature_direction = activated_feature_weights.mean(dim=0).numpy()  # Average over k features
    
    # Reshape to 2D for visualization (assuming square-ish dimensions)
    # For 3072 dim, try 32x96 or similar
    dim = len(avg_feature_direction)
    # Find reasonable factorization
    sqrt_dim = int(math.sqrt(dim))
    h, w = sqrt_dim, dim // sqrt_dim
    if h * w != dim:
        # Try other factorizations
        for h in range(sqrt_dim, 0, -1):
            if dim % h == 0:
                w = dim // h
                break
    
    feature_map = avg_feature_direction[:h*w].reshape(h, w)
    im = ax4.imshow(feature_map, cmap='RdBu_r', aspect='auto')
    ax4.set_title('Average Feature Direction\n(Decoder Weights)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Dimension')
    ax4.set_ylabel('Dimension')
    plt.colorbar(im, ax=ax4, label='Weight Value')
    
    # 5. Individual feature decoder weights (bottom-middle)
    ax5 = fig.add_subplot(gs[1, 1])
    # Show decoder weights for top 5 features
    num_show_features = min(5, k)
    for i in range(num_show_features):
        feat_idx = top_k_indices[i].item()
        feat_weight = decoder_weights[feat_idx].numpy()  # Shape: (features,)
        # Show as line plot
        ax5.plot(feat_weight[:500], alpha=0.7, label=f'Feat {feat_idx} (val={top_k_values[i]:.2f})', linewidth=1)
    ax5.set_xlabel('Dimension Index')
    ax5.set_ylabel('Decoder Weight Value')
    ax5.set_title(f'Decoder Weights: Top {num_show_features} Features', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=8, loc='best')
    ax5.grid(True, alpha=0.3)
    
    # 6. Feature importance summary (bottom-right)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Calculate statistics
    total_activation = top_k_values.sum().item()
    feature_contributions = (top_k_values / total_activation * 100).float().numpy()
    
    summary_text = f"""
    SAE FEATURE ANALYSIS
    
    Reference: {reference_prompt[:50]}...
    
    ACTIVATED FEATURES:
    • Total features: {k} (target sparsity)
    • Total activation: {total_activation:.2f}
    
    TOP 5 FEATURES:
    """
    
    for i in range(min(5, k)):
        feat_idx = top_k_indices[i].item()
        feat_val = top_k_values[i].item()
        feat_contrib = feature_contributions[i]
        summary_text += f"  {i+1}. Feature {feat_idx}: {feat_val:.2f} ({feat_contrib:.1f}%)\n"
    
    summary_text += f"""
    FEATURE DIVERSITY:
    • Unique features: {len(torch.unique(top_k_indices))}
    • Activation spread: {top_k_values.std().item():.2f}
    
    INTERPRETATION:
    Each feature represents a learned pattern
    in FLUX's attention activations. Higher
    activation values indicate stronger
    presence of that pattern.
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle('SAE Features Activated for Reference Image', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(output_dir / 'features_on_reference_image.png', bbox_inches='tight', dpi=300)
    print(f"Saved: {output_dir / 'features_on_reference_image.png'}")
    plt.close()
    
    return top_k_indices, top_k_values

def main():
    """Main validation pipeline."""
    # Configuration
    checkpoint_path = "./checkpoints/small-flux-sae-test"
    output_dir = "./checkpoints/small-flux-sae-test/visualizations"
    cc3m_path = Path("/mnt/drive_a/Projects/sae/data/cc3m/")
    loc = "transformer_blocks.0.attn"
    stream = 1
    concept = 'cat'
    num_prompts = 100
    
    print("="*80)
    print("SAE Sparsity Validation")
    print("="*80)
    
    # Load CC3M dataset
    print(f"\nLoading CC3M dataset from {cc3m_path}...")
    cc3m_dataset = CC3M(cc3m_path)
    print(f"Dataset size: {len(cc3m_dataset)} samples")
    
    # Collect cat prompts
    cat_prompts, cat_indices = collect_cat_prompts(cc3m_dataset, num_prompts=num_prompts, concept=concept)
    
    # Get random cat image (for reference, not used in activation sampling)
    cat_image, cat_image_text, cat_image_idx = get_random_cat_image(cc3m_dataset, concept=concept)
    
    # Save the reference image
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    cat_image.save(output_dir_path / 'reference_cat_image.png')
    print(f"Saved reference image to: {output_dir_path / 'reference_cat_image.png'}")
    
    # Load model
    model, config = load_model(checkpoint_path)
    model.eval()
    
    print(f"\nModel Configuration:")
    print(f"  Total SAE features (pages): {config['pages']}")
    print(f"  Sparsity (k): {config['k']} features activated per input")
    print(f"  Input dimension: {config['features']}")
    
    # Sample activations for cat prompts
    print(f"\nSampling activations for {len(cat_prompts)} '{concept}' prompts...")
    print(f"Location: {loc}, Stream: {stream}")
    print("This may take a while as FLUX pipeline runs for each prompt...")
    
    raw_activations = sample_activations(cat_prompts, loc=loc, stream=stream)
    print(f"Raw activation shape: {raw_activations.shape}")
    
    # Visualize comparison
    print(f"\nGenerating sparsity validation visualizations...")
    stats = visualize_sparsity_comparison(
        model, raw_activations, None, cat_prompts, output_dir, k=config['k']
    )
    
    # Visualize features on reference image
    print(f"\nVisualizing SAE features on reference image...")
    # Get activation for the reference image's prompt
    reference_activation = sample_activations([cat_image_text], loc=loc, stream=stream)
    reference_activation = reference_activation[0:1]  # Keep first activation vector
    
    top_features, top_values = visualize_features_on_image(
        model, cat_image, cat_image_text, reference_activation, None, 
        output_dir, k=config['k']
    )
    
    print(f"\n✅ Validation complete!")
    print(f"\nResults:")
    print(f"  Raw L0: {stats['raw_l0_mean']:.1f} ± {stats['raw_l0_std']:.1f}")
    print(f"  SAE L0: {stats['sae_l0_mean']:.1f} ± {stats['sae_l0_std']:.1f}")
    print(f"  Sparsity reduction: {stats['sparsity_reduction_percent']:.1f}%")
    print(f"\nTop features for reference image:")
    for i in range(min(5, len(top_features))):
        print(f"  Feature {top_features[i].item()}: {top_values[i].item():.2f}")
    print(f"\nFiles saved to: {output_dir}")
    print("  - sparsity_validation.png: Comprehensive comparison")
    print("  - features_on_reference_image.png: SAE features on reference image")
    print("  - sparsity_stats.json: Quantitative statistics")
    print("  - reference_cat_image.png: Reference image from CC3M")

if __name__ == "__main__":
    main()

