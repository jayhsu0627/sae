#!/usr/bin/env python3
"""
Visualize SAE activations for the small-flux-sae-test model.
This script generates visualizations suitable for paper/report inclusion.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from diffusers import FluxPipeline
from einops import rearrange
from autoencoder import TopkSparseAutoencoder
from fluxsae import FluxActivationSampler, CC3MPromptDataset
from accelerate import Accelerator
import json

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

def visualize_activation_patterns(model, activations, prompts, output_dir, k=16):
    """Visualize which SAE features activate for different prompts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        # Encode activations through SAE
        encoded = model.encode(activations.to(model.encoder.weight.device))
        encoded = encoded.cpu()
    
    # 1. Activation heatmap for multiple prompts
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SAE Activation Patterns', fontsize=14, fontweight='bold')
    
    # Top-left: Activation heatmap (prompts x features)
    # Show top-k activated features for each prompt
    num_prompts = min(len(prompts), 10)
    top_features_per_prompt = []
    
    for i in range(num_prompts):
        prompt_activations = encoded[i]
        top_k_indices = torch.topk(prompt_activations, k=k, dim=-1).indices
        top_features_per_prompt.append(top_k_indices.numpy())
    
    # Create binary matrix: 1 if feature is in top-k for that prompt
    feature_matrix = np.zeros((num_prompts, model.pages))
    for i, top_k in enumerate(top_features_per_prompt):
        feature_matrix[i, top_k] = 1
    
    sns.heatmap(
        feature_matrix,
        ax=axes[0, 0],
        cmap='YlOrRd',
        cbar_kws={'label': 'Activated (Top-k)'},
        xticklabels=False,
        yticklabels=[f"Prompt {i+1}" for i in range(num_prompts)]
    )
    axes[0, 0].set_title(f'Top-{k} Activated Features per Prompt')
    axes[0, 0].set_xlabel('SAE Feature Index')
    axes[0, 0].set_ylabel('Prompt')
    
    # Top-right: Feature activation distribution
    # Show how many features activate per prompt (L0 sparsity)
    l0_per_prompt = (encoded > 0).sum(dim=-1).numpy()
    axes[0, 1].hist(l0_per_prompt, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(l0_per_prompt.mean(), color='red', linestyle='--', 
                        label=f'Mean: {l0_per_prompt.mean():.1f}')
    axes[0, 1].set_xlabel('Number of Activated Features (L0)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Sparsity Distribution (L0)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bottom-left: Feature activation strength distribution
    # Show distribution of activation magnitudes
    non_zero_activations = encoded[encoded > 0].numpy()
    axes[1, 0].hist(non_zero_activations, bins=50, edgecolor='black', alpha=0.7, log=True)
    axes[1, 0].set_xlabel('Activation Magnitude')
    axes[1, 0].set_ylabel('Frequency (log scale)')
    axes[1, 0].set_title('Activation Magnitude Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Bottom-right: Most frequently activated features
    # Count how many prompts activate each feature
    feature_activation_counts = (encoded > 0).sum(dim=0).numpy()
    top_activated_features = np.argsort(feature_activation_counts)[-20:][::-1]
    
    axes[1, 1].barh(range(len(top_activated_features)), 
                    feature_activation_counts[top_activated_features],
                    edgecolor='black', alpha=0.7)
    axes[1, 1].set_yticks(range(len(top_activated_features)))
    axes[1, 1].set_yticklabels([f"Feature {idx}" for idx in top_activated_features])
    axes[1, 1].set_xlabel('Number of Prompts Activating Feature')
    axes[1, 1].set_title('Top 20 Most Frequently Activated Features')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'activation_patterns.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'activation_patterns.png'}")
    plt.close()
    
    # 2. Individual prompt activation visualization
    fig, axes = plt.subplots(min(len(prompts), 5), 1, figsize=(12, 2*min(len(prompts), 5)))
    if len(prompts) == 1:
        axes = [axes]
    
    fig.suptitle('Top-K Activated Features per Prompt', fontsize=14, fontweight='bold')
    
    for i, (prompt, ax) in enumerate(zip(prompts[:5], axes)):
        prompt_activations = encoded[i]
        top_k_values, top_k_indices = torch.topk(prompt_activations, k=k, dim=-1)
        
        # Create bar plot
        colors = plt.cm.viridis(top_k_values.numpy() / top_k_values.max())
        ax.barh(range(k), top_k_values.numpy(), color=colors, edgecolor='black')
        ax.set_yticks(range(k))
        ax.set_yticklabels([f"Feature {idx.item()}" for idx in top_k_indices])
        ax.set_xlabel('Activation Magnitude')
        ax.set_title(f'Prompt {i+1}: "{prompt[:60]}..."', fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prompt_activations.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'prompt_activations.png'}")
    plt.close()
    
    # 3. Sparsity statistics
    stats = {
        'mean_l0': float(l0_per_prompt.mean()),
        'std_l0': float(l0_per_prompt.std()),
        'min_l0': int(l0_per_prompt.min()),
        'max_l0': int(l0_per_prompt.max()),
        'mean_activation_magnitude': float(non_zero_activations.mean()),
        'std_activation_magnitude': float(non_zero_activations.std()),
        'total_features': int(model.pages),
        'k': int(k),
        'num_prompts': len(prompts)
    }
    
    with open(output_dir / 'activation_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\nActivation Statistics:")
    print(f"  Mean L0 (activated features per prompt): {stats['mean_l0']:.1f} ± {stats['std_l0']:.1f}")
    print(f"  L0 range: {stats['min_l0']} - {stats['max_l0']}")
    print(f"  Mean activation magnitude: {stats['mean_activation_magnitude']:.4f}")
    print(f"  Total SAE features: {stats['total_features']}")
    
    return stats

def main():
    """Main visualization pipeline."""
    # Configuration
    checkpoint_path = "./checkpoints/small-flux-sae-test"
    output_dir = "./checkpoints/small-flux-sae-test/visualizations"
    loc = "transformer_blocks.0.attn"
    stream = 1
    
    # Test prompts for visualization
    # NOTE: These are hardcoded test prompts for demonstration.
    # To use actual training prompts from CC3M, uncomment the dataset loading below.
    use_training_prompts = False  # Set to True to use CC3M dataset prompts
    
    if use_training_prompts:
        # Load prompts from training dataset
        dataset = CC3MPromptDataset(folder="/mnt/drive_a/Projects/sae/data/cc3m/")
        test_prompts = [dataset[i] for i in range(8)]  # Get first 8 prompts
        print(f"Using {len(test_prompts)} prompts from CC3M training dataset")
    else:
        # Hardcoded test prompts for visualization
        test_prompts = [
            "a cat sitting on a windowsill",
            "a red car driving on a highway",
            "a beautiful sunset over the ocean",
            "a person reading a book in a library",
            "a dog playing in a park",
            "a city skyline at night",
            "a bowl of fruit on a table",
            "a mountain landscape with trees",
        ]
        print(f"Using {len(test_prompts)} hardcoded test prompts")
    
    print("="*80)
    print("SAE Activation Visualization")
    print("="*80)
    
    # Load model
    model, config = load_model(checkpoint_path)
    model.eval()
    
    # Print model info
    print(f"\nModel Configuration:")
    print(f"  Total SAE features (pages): {config['pages']}")
    print(f"  Sparsity (k): {config['k']} features activated per input")
    print(f"  Input dimension: {config['features']}")
    print(f"  Note: k={config['k']} means only top-{config['k']} features activate per prompt,")
    print(f"        but the SAE learned {config['pages']} total features in its dictionary.")
    
    # Sample activations
    print(f"\nSampling activations for {len(test_prompts)} test prompts...")
    print(f"Location: {loc}, Stream: {stream}")
    print(f"Note: transformer_blocks.0.attn processes multimodal (image+text) representations")
    activations = sample_activations(test_prompts, loc=loc, stream=stream)
    print(f"Activation shape: {activations.shape}")
    
    # Visualize
    print(f"\nGenerating visualizations...")
    stats = visualize_activation_patterns(
        model, activations, test_prompts, output_dir, k=config['k']
    )
    
    print(f"\n✅ Visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - activation_patterns.png: Overview of activation patterns")
    print("  - prompt_activations.png: Individual prompt activations")
    print("  - activation_stats.json: Quantitative statistics")

if __name__ == "__main__":
    main()

