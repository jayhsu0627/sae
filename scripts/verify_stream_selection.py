#!/usr/bin/env python3
"""
Quick verification script to check what outputs are returned from different hook locations.
"""

import torch
from diffusers import FluxPipeline

def test_hook_location(loc, description):
    """Test what a hook location returns"""
    print(f"\n{'='*60}")
    print(f"Testing: {loc} ({description})")
    print(f"{'='*60}")
    
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16
    )
    
    captured_output = None
    captured_input = None
    
    def capture_output_hook(module, args, output):
        nonlocal captured_output
        captured_output = output
        return None
    
    def capture_input_hook(module, args):
        nonlocal captured_input
        if args and len(args) > 0:
            captured_input = args[0] if isinstance(args[0], torch.Tensor) else args
        return None
    
    module = pipe.transformer.get_submodule(loc)
    handle_input = module.register_forward_pre_hook(capture_input_hook)
    handle_output = module.register_forward_hook(capture_output_hook)
    
    try:
        # Run a single forward pass
        with torch.no_grad():
            _ = pipe(
                ["test prompt"],
                height=256,
                width=256,
                num_inference_steps=1,
                guidance_scale=0.0
            )
        
        # Analyze outputs
        print(f"Input type: {type(captured_input)}")
        if isinstance(captured_input, torch.Tensor):
            print(f"Input shape: {captured_input.shape}")
        elif isinstance(captured_input, (tuple, list)):
            print(f"Input is tuple/list with {len(captured_input)} elements")
            for i, inp in enumerate(captured_input):
                if isinstance(inp, torch.Tensor):
                    print(f"  Input[{i}] shape: {inp.shape}")
        
        print(f"\nOutput type: {type(captured_output)}")
        if isinstance(captured_output, torch.Tensor):
            print(f"Output shape: {captured_output.shape}")
            print("✓ Single tensor output (MLP/FF or single transformer block)")
        elif isinstance(captured_output, tuple):
            print(f"Output is tuple with {len(captured_output)} elements:")
            for i, out in enumerate(captured_output):
                if isinstance(out, torch.Tensor):
                    print(f"  Output[{i}] shape: {out.shape}")
            
            # Determine what it is
            if "transformer_blocks" in loc and ".attn" in loc:
                print("\n→ This is an ATTENTION module")
                print("  Expected: (query, key) tuple")
                print("  stream=0 should get: Output[0] (query)")
                print("  stream=1 should get: Output[1] (key)")
            elif "transformer_blocks" in loc and (".attn" not in loc and ".ff" not in loc):
                print("\n→ This is an ENTIRE TRANSFORMER BLOCK")
                print("  Expected: (text_stream, image_stream) = (encoder_hidden_states, hidden_states)")
                print("  stream=0 (image) should get: Output[1] (hidden_states)")
                print("  stream=1 (text) should get: Output[0] (encoder_hidden_states)")
            else:
                print("\n→ Unknown tuple output type")
        else:
            print(f"Output: {captured_output}")
        
    finally:
        handle_input.remove()
        handle_output.remove()
        del pipe
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
    # Test entire block (groundtruth case)
    test_hook_location(
        "transformer_blocks.18",
        "Entire transformer block (groundtruth)"
    )
    
    # Test attention module
    test_hook_location(
        "transformer_blocks.18.attn",
        "Attention module"
    )
    
    # Test MLP module
    test_hook_location(
        "transformer_blocks.18.ff",
        "MLP/Feedforward module"
    )
    
    print(f"\n{'='*60}")
    print("Verification complete!")
    print(f"{'='*60}")

