#!/usr/bin/env python3
"""
Test script to verify what activations are being captured from different locations.
This helps verify if the SAE is correctly attached to attention vs MLP/FF layers.
"""

import torch
from diffusers import FluxPipeline

def test_activation_hook():
    """Test what different locations return."""
    
    print("Loading FLUX pipeline...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", 
        torch_dtype=torch.bfloat16
    )
    
    transformer = pipe.transformer
    
    # Test locations
    test_locations = [
        ("transformer_blocks.0.attn", "Attention module"),
        ("transformer_blocks.0.ff", "Feed-forward/MLP module"),
        ("single_transformer_blocks.0", "Single transformer block output"),
    ]
    
    print("\n" + "="*80)
    print("TESTING ACTIVATION HOOK OUTPUTS")
    print("="*80)
    
    # Simple test prompt
    test_prompt = "a cat"
    
    for loc_name, description in test_locations:
        print(f"\n{'='*80}")
        print(f"Location: {loc_name}")
        print(f"Description: {description}")
        print(f"{'='*80}")
        
        try:
            module = transformer.get_submodule(loc_name)
            print(f"✅ Module found: {type(module).__name__}")
            
            # Register hook
            captured_output = None
            
            def hook_fn(module, input, output):
                nonlocal captured_output
                captured_output = output
                return output
            
            handle = module.register_forward_hook(hook_fn)
            
            # Run a forward pass
            with torch.no_grad():
                _ = pipe(
                    test_prompt,
                    height=256,
                    width=256,
                    guidance_scale=0.0,
                    max_sequence_length=256,
                    num_inference_steps=1,
                )
            
            # Analyze output
            if captured_output is not None:
                print(f"\n📊 Output Analysis:")
                print(f"   Type: {type(captured_output)}")
                
                if isinstance(captured_output, tuple):
                    print(f"   ✅ Returns tuple with {len(captured_output)} elements")
                    for i, elem in enumerate(captured_output):
                        if isinstance(elem, torch.Tensor):
                            print(f"      Element {i}: shape={elem.shape}, dtype={elem.dtype}")
                        else:
                            print(f"      Element {i}: type={type(elem)}")
                    
                    # Check if it's (query, key) or similar
                    if len(captured_output) == 2:
                        q, k = captured_output[0], captured_output[1]
                        if isinstance(q, torch.Tensor) and isinstance(k, torch.Tensor):
                            print(f"\n   💡 Interpretation:")
                            print(f"      Element 0 (stream 0): shape {q.shape} - likely QUERY stream")
                            print(f"      Element 1 (stream 1): shape {k.shape} - likely KEY stream")
                            print(f"      Current training uses --stream 1 → KEY stream activations")
                elif isinstance(captured_output, torch.Tensor):
                    print(f"   ✅ Returns single tensor")
                    print(f"      Shape: {captured_output.shape}")
                    print(f"      Dtype: {captured_output.dtype}")
                    print(f"   💡 This is a single output (no stream selection needed)")
                else:
                    print(f"   ⚠️  Unexpected output type: {type(captured_output)}")
            
            handle.remove()
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("""
Based on the mapping:
  - transformer_blocks.0.attn → double_blocks.0.img_attn (ATTENTION)
  - transformer_blocks.0.ff → double_blocks.0.img_mlp (MLP/FEED-FORWARD)

Your current setup:
  - Location: transformer_blocks.0.attn
  - Stream: 1
  - This captures ATTENTION activations (key stream)

If you want MLP/FF activations instead:
  - Location: transformer_blocks.0.ff
  - Stream: 0 (or not needed if it returns single tensor)
    """)
    print("="*80 + "\n")

if __name__ == "__main__":
    test_activation_hook()

