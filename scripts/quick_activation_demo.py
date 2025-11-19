#!/usr/bin/env python3
"""
Quick demo script to show SAE activations for a few prompts.
Simpler version for quick testing.
"""

import torch
from pathlib import Path
from autoencoder import TopkSparseAutoencoder
from fluxsae import FluxActivationSampler
from einops import rearrange
from accelerate import Accelerator
import json

def main():
    # Load model
    checkpoint_path = "./checkpoints/small-flux-sae-test"
    model = TopkSparseAutoencoder.from_pretrained(checkpoint_path)
    
    with open(Path(checkpoint_path) / "config.json") as f:
        config = json.load(f)
    
    print(f"Model: {checkpoint_path}")
    print(f"  Features: {config['features']}, Pages: {config['pages']}, k: {config['k']}")
    
    # Test prompts
    test_prompts = [
        "a cat sitting on a windowsill",
        "a red car driving on a highway",
        "a beautiful sunset over the ocean",
    ]
    
    # Setup FLUX sampler
    accelerator = Accelerator()
    sampler = FluxActivationSampler("black-forest-labs/FLUX.1-schnell", loc="transformer_blocks.0.attn")
    sampler.pipe.transformer, sampler.pipe.vae, sampler.pipe.text_encoder, sampler.pipe.text_encoder_2 = accelerator.prepare(
        sampler.pipe.transformer, sampler.pipe.vae, sampler.pipe.text_encoder, sampler.pipe.text_encoder_2
    )
    
    model = accelerator.prepare(model)
    model.eval()
    
    print(f"\nSampling activations for {len(test_prompts)} prompts...")
    
    with torch.no_grad():
        with sampler as s:
            outputs = s(
                test_prompts,
                height=256,
                width=256,
                guidance_scale=0.0,
                max_sequence_length=256,
                num_inference_steps=1,
            )
            activations = outputs["activations"]
            
            # Handle tuple (attention) vs single tensor
            if isinstance(activations, tuple):
                _, x = activations  # stream 1 (key)
            else:
                x = activations
            
            x = rearrange(x, "b ... d -> (b ...) d")
    
    # Encode through SAE
    print(f"Activation shape: {x.shape}")
    encoded = model.encode(x.to(model.encoder.weight.device))
    encoded = encoded.cpu()
    
    # Show statistics
    print("\n" + "="*60)
    print("Activation Statistics")
    print("="*60)
    
    for i, prompt in enumerate(test_prompts):
        prompt_encoded = encoded[i]
        l0 = (prompt_encoded > 0).sum().item()
        top_k_values, top_k_indices = torch.topk(prompt_encoded, k=config['k'], dim=-1)
        
        print(f"\nPrompt {i+1}: \"{prompt}\"")
        print(f"  L0 (activated features): {l0}")
        print(f"  Top-{config['k']} features: {top_k_indices.tolist()}")
        print(f"  Top-{config['k']} values: {top_k_values.tolist()[:5]}...")  # Show first 5
    
    # Overall statistics
    l0_all = (encoded > 0).sum(dim=-1)
    print(f"\nOverall Statistics:")
    print(f"  Mean L0: {l0_all.float().mean():.1f}")
    print(f"  Std L0: {l0_all.float().std():.1f}")
    print(f"  Min L0: {l0_all.min().item()}")
    print(f"  Max L0: {l0_all.max().item()}")
    
    print("\n✅ Demo complete! Run visualize_sae_activations.py for full visualizations.")

if __name__ == "__main__":
    main()

