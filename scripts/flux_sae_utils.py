"""
Utility functions for FLUX SAE steering - Adapted from SDXL-turbo example

This module provides:
- Steering functions: replace_with_feature, add_feature_on_area
- Visualization: plot_image_heatmap
- Text stream analysis: tokenizer utilities, attention extraction, text-guided sparse maps
- Token analysis: plot_token_activation_strength
"""
import torch
from einops import rearrange
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def replace_with_feature(sae, feature_idx, strength, output, stream=0):
    """
    Replace activations with a specific feature direction scaled by strength.
    Similar to SDXL-turbo's replace_with_feature but adapted for FLUX.
    
    Args:
        sae: The TopkSparseAutoencoder model
        feature_idx: Index of feature to inject
        strength: Scaling factor for the feature
        output: The activation tensor from the hook (module output)
        stream: Which stream to modify (0=query, 1=key) for tuple outputs
    
    Returns:
        Modified activations with feature injected
    """
    # The output is passed directly from the hook
    activation = output
    
    # Handle tuple outputs (query, key) from attention layers
    if isinstance(activation, tuple):
        query, key = activation
        # Select target stream based on stream parameter
        target = query if stream == 0 else key
        other = key if stream == 0 else query
        is_tuple = True
    else:
        target = activation
        other = None
        is_tuple = False
    
    # Flatten spatial dimensions
    original_shape = target.shape
    target_flat = rearrange(target, "b ... d -> (b ...) d")
    num_tokens = target_flat.shape[0]  # Number of tokens (spatial locations or sequence length)
    
    # Ensure dtype/device match
    sae_dtype = next(sae.parameters()).dtype
    sae_device = next(sae.parameters()).device
    target_flat = target_flat.to(dtype=sae_dtype, device=sae_device)
    
    # Validate feature_idx is within bounds
    if feature_idx < 0 or feature_idx >= sae.pages:
        raise ValueError(f"Feature index {feature_idx} is out of bounds. SAE has {sae.pages} pages (valid indices: 0-{sae.pages-1})")
    
    # Get feature direction (decoder weight for this feature)
    # Decoder weight shape: (features, pages)
    if feature_idx >= sae.decoder.weight.shape[1]:
        raise ValueError(f"Feature index {feature_idx} is out of bounds for decoder weight. Decoder has {sae.decoder.weight.shape[1]} pages (valid indices: 0-{sae.decoder.weight.shape[1]-1})")
    
    feature_direction = sae.decoder.weight[:, feature_idx]  # Shape: (features,)
    
    # Replace with scaled feature direction
    # Broadcast to match number of tokens: (num_tokens, features)
    with torch.no_grad():
        target_steered = strength * feature_direction.unsqueeze(0).expand(num_tokens, -1)
    
    # Convert back to original dtype and device
    target_steered = target_steered.to(dtype=target.dtype, device=target.device)
    target_reshaped = target_steered.reshape(original_shape)
    
    # Return in same format as input
    if is_tuple:
        return (target_reshaped, other)
    else:
        return target_reshaped


def add_feature_on_area(sae, feature_idx, strength_map_or_value, output, stream=0):
    """
    Add a feature to activations with spatial strength map. Adapted from SDXL-turbo for FLUX.
    
    Args:
        sae: The TopkSparseAutoencoder model
        feature_idx: Index of feature to add
        strength_map_or_value: Spatial strength map (height, width) or scalar value - will be broadcasted
        output: The activation tensor from the hook (module output)
        stream: Which stream to modify (0=query, 1=key) for tuple outputs
    
    Returns:
        Modified activations with feature added
    """
    # The output is passed directly from the hook
    activation = output
    
    # Handle tuple outputs
    if isinstance(activation, tuple):
        query, key = activation
        target = query if stream == 0 else key
        other = key if stream == 0 else query
        is_tuple = True
    else:
        target = activation
        other = None
        is_tuple = False
    
    original_shape = target.shape
    # For FLUX, activations might be (batch, seq, height, width, dim)
    # We need to handle the spatial dimensions
    if len(original_shape) == 5:
        # (batch, seq, height, width, dim)
        batch, seq, h, w, dim = original_shape
        target_flat = rearrange(target, "b s h w d -> (b s h w) d")
        spatial_shape = (h, w)
    elif len(original_shape) == 4:
        # (batch, height, width, dim) or (batch, seq, length, dim)
        if original_shape[-1] == sae.features:
            # Assume (batch, height, width, dim)
            batch, h, w, dim = original_shape
            target_flat = rearrange(target, "b h w d -> (b h w) d")
            spatial_shape = (h, w)
        else:
            # Assume (batch, seq, length, dim)
            batch, seq, length, dim = original_shape
            target_flat = rearrange(target, "b s l d -> (b s l) d")
            # For sequence-based, we'll need to reshape strength_map
            spatial_shape = (seq, length)
    else:
        # Fallback: flatten all but last dimension
        target_flat = rearrange(target, "b ... d -> (b ...) d")
        spatial_shape = None
    
    # Ensure dtype/device match
    sae_dtype = next(sae.parameters()).dtype
    sae_device = next(sae.parameters()).device
    target_flat = target_flat.to(dtype=sae_dtype, device=sae_device)
    
    # Get feature direction
    feature_direction = sae.decoder.weight[:, feature_idx]  # Shape: (features,)
    
    # Apply feature addition with spatial strength map
    with torch.no_grad():
        # Handle strength_map_or_value (can be tensor map or scalar)
        if isinstance(strength_map_or_value, torch.Tensor):
            strength_map = strength_map_or_value
            # Reshape strength_map to match flattened spatial dimensions
            if spatial_shape is not None:
                # Ensure strength_map matches spatial dimensions
                if len(strength_map.shape) == 2 and strength_map.shape == spatial_shape:
                    # Already matches spatial shape
                    strength_flat = strength_map.flatten().unsqueeze(-1)  # (spatial, 1)
                elif len(strength_map.shape) == 1:
                    # Already flattened
                    strength_flat = strength_map.unsqueeze(-1)  # (spatial, 1)
                else:
                    # Need to reshape/interpolate
                    if strength_map.shape != spatial_shape:
                        strength_map = torch.nn.functional.interpolate(
                            strength_map.unsqueeze(0).unsqueeze(0).float(),
                            size=spatial_shape,
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0).squeeze(0)
                    strength_flat = strength_map.flatten().unsqueeze(-1)
            else:
                # No spatial shape info, flatten what we have
                strength_flat = strength_map.flatten().unsqueeze(-1)
        else:
            # Scalar value - broadcast to all spatial locations
            strength_flat = torch.tensor(strength_map_or_value, device=sae_device, dtype=sae_dtype)
            # Expand to match number of spatial locations
            num_spatial = target_flat.shape[0]
            strength_flat = strength_flat.expand(num_spatial, 1)
        
        # Broadcast: (spatial, 1) * (1, features) -> (spatial, features)
        feature_addition = strength_flat * feature_direction.unsqueeze(0)
        target_steered = target_flat + feature_addition
    
    # Convert back
    target_steered = target_steered.to(dtype=target.dtype, device=target.device)
    target_reshaped = target_steered.reshape(original_shape)
    
    if is_tuple:
        return (target_reshaped, other)
    else:
        return target_reshaped


def plot_image_heatmap(image, sparse_maps, feature_idx, upsample_factor=32):
    """
    Plot heatmap overlay on image for a specific feature.
    Adapted from SDXL-turbo example for FLUX.
    
    Args:
        image: PIL Image
        sparse_maps: Sparse activation maps (height, width, num_features)
        feature_idx: Feature index to visualize
        upsample_factor: Factor to upsample heatmap to match image size
    
    Returns:
        PIL Image with heatmap overlay
    """
    # Extract heatmap for this feature
    # Ensure feature_idx is a Python int for indexing
    if isinstance(feature_idx, torch.Tensor):
        feature_idx = int(feature_idx.item())
    elif not isinstance(feature_idx, int):
        feature_idx = int(feature_idx)
    
    if isinstance(sparse_maps, torch.Tensor):
        # Handle both 2D (spatial, features) and 3D (h, w, features) sparse_maps
        if len(sparse_maps.shape) == 2:
            # 2D case: (spatial_tokens, features) - no spatial dimensions
            # Extract feature activations across all tokens
            heatmap_tensor = sparse_maps[:, feature_idx]  # (spatial_tokens,)
            if heatmap_tensor.dtype == torch.bfloat16:
                heatmap_tensor = heatmap_tensor.float()
            heatmap_1d = heatmap_tensor.cpu().numpy()
            
            # For 2D sparse_maps, we can't create a proper spatial heatmap
            # Create a simple visualization by reshaping to approximate square
            num_tokens = len(heatmap_1d)
            # Try to find reasonable dimensions
            h = int(np.sqrt(num_tokens))
            w = (num_tokens + h - 1) // h  # Ceiling division
            # Pad if necessary
            if h * w > num_tokens:
                padding = h * w - num_tokens
                heatmap_1d = np.pad(heatmap_1d, (0, padding), mode='constant', constant_values=0)
            heatmap = heatmap_1d[:h*w].reshape(h, w)
        else:
            # 3D case: (h, w, features) - has spatial dimensions
            heatmap_tensor = sparse_maps[:, :, feature_idx]
            if heatmap_tensor.dtype == torch.bfloat16:
                heatmap_tensor = heatmap_tensor.float()
            heatmap = heatmap_tensor.cpu().numpy()
    else:
        # NumPy array
        if len(sparse_maps.shape) == 2:
            heatmap_1d = sparse_maps[:, feature_idx]
            num_tokens = len(heatmap_1d)
            h = int(np.sqrt(num_tokens))
            w = (num_tokens + h - 1) // h
            if h * w > num_tokens:
                padding = h * w - num_tokens
                heatmap_1d = np.pad(heatmap_1d, (0, padding), mode='constant', constant_values=0)
            heatmap = heatmap_1d[:h*w].reshape(h, w)
        else:
            heatmap = sparse_maps[:, :, feature_idx]
    
    # Upsample heatmap to match image size using nearest neighbor (no smoothing)
    h, w = heatmap.shape
    # Use PIL Image resize with nearest neighbor to preserve blocky appearance
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
    target_size = (image.size[0], image.size[1])  # (width, height)
    heatmap_img = heatmap_img.resize(target_size, Image.Resampling.NEAREST)
    heatmap_upsampled = np.array(heatmap_img).astype(np.float32) / 255.0
    
    # Normalize heatmap to [0, 1] range (required for matplotlib colormaps)
    # Option 1: Min-max normalization (sensitive to outliers)
    # Option 2: Percentile-based normalization (more robust, recommended)
    
    # Use percentile-based normalization to be more robust to outliers
    # This uses percentiles instead of min/max, so extreme values don't dominate
    p_low = np.percentile(heatmap_upsampled, 5)   # 5th percentile as "minimum"
    p_high = np.percentile(heatmap_upsampled, 95)  # 95th percentile as "maximum"
    
    if p_high > p_low:
        # Normalize using percentiles (more robust than min-max)
        heatmap_normalized = np.clip(
            (heatmap_upsampled - p_low) / (p_high - p_low),
            0, 1
        )
    else:
        # Fallback to min-max if percentiles are the same
        if np.max(heatmap_upsampled) > np.min(heatmap_upsampled):
            heatmap_normalized = (heatmap_upsampled - np.min(heatmap_upsampled)) / (
                np.max(heatmap_upsampled) - np.min(heatmap_upsampled)
            )
        else:
            # All values are the same
            heatmap_normalized = np.zeros_like(heatmap_upsampled)
    
    # Apply threshold to make low activations transparent
    # Only show color for activations above a percentile threshold
    threshold_percentile = 10  # Only show top 90% of activations (hide bottom 40%)
    threshold_value = np.percentile(heatmap_normalized, threshold_percentile)
    
    # Create mask: values below threshold become 0 (will map to transparent color)
    heatmap_thresholded = np.where(heatmap_normalized < threshold_value, 0, heatmap_normalized)
    
    # Re-normalize after thresholding (so max is still 1.0 for colormap)
    if np.max(heatmap_thresholded) > 0:
        heatmap_thresholded = heatmap_thresholded / np.max(heatmap_thresholded)
    
    # Convert image to RGBA
    image = image.convert("RGBA")
    
    # Create colormap with transparency
    jet = plt.cm.jet
    cmap = jet(np.arange(jet.N))
    cmap[:1, -1] = 0  # First color fully transparent (for values = 0)
    cmap[1:, -1] = 0.6  # Rest semi-transparent
    cmap = ListedColormap(cmap)
    
    # Apply colormap to thresholded heatmap
    heatmap_rgba = cmap(heatmap_thresholded)
    heatmap_image = Image.fromarray((heatmap_rgba * 255).astype(np.uint8))
    
    # Resize heatmap to match image size exactly (using nearest neighbor to preserve blocky appearance)
    heatmap_image = heatmap_image.resize(image.size, Image.Resampling.NEAREST)
    
    # Composite images
    heatmap_with_transparency = Image.alpha_composite(image, heatmap_image)
    
    return heatmap_with_transparency


def get_tokenizer(pipe):
    """
    Get tokenizer from FLUX pipeline.
    Tries multiple methods to find the tokenizer.
    
    Returns:
        tokenizer: Tokenizer object or None
        can_decode: Boolean indicating if tokenizer is available
    """
    try:
        if hasattr(pipe, 'tokenizer'):
            return pipe.tokenizer, True
        elif hasattr(pipe, 'text_encoder') and hasattr(pipe.text_encoder, 'tokenizer'):
            return pipe.text_encoder.tokenizer, True
        else:
            # Fallback: try to load T5 tokenizer
            from transformers import T5Tokenizer
            tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
            return tokenizer, True
    except Exception as e:
        return None, False


def get_actual_prompt_length(tokenizer, prompt, max_length=512):
    """
    Get actual prompt length excluding padding tokens.
    
    Args:
        tokenizer: Tokenizer object
        prompt: Text prompt string
        max_length: Maximum token length (default 512)
    
    Returns:
        actual_length: Number of actual prompt tokens (excluding padding)
        token_ids_list: List of token IDs
        token_texts: Dictionary mapping token index to decoded text
    """
    token_texts = {}
    token_ids_list = []
    actual_length = max_length  # Default
    
    if tokenizer is None:
        return actual_length, token_ids_list, token_texts
    
    try:
        tokenized = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        token_ids_list = tokenized['input_ids'][0].cpu().tolist()
        
        # Find actual prompt length
        padding_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1
        
        actual_length = len(token_ids_list)
        for i, token_id in enumerate(token_ids_list):
            if token_id == padding_token_id or (i > 0 and token_id == eos_token_id):
                actual_length = i
                break
        
        # Decode tokens to text
        for i, token_id in enumerate(token_ids_list[:actual_length]):
            try:
                token_text = tokenizer.decode([token_id])
                token_texts[i] = token_text.strip()
            except:
                token_texts[i] = f"<token_{i}>"
    except Exception as e:
        pass
    
    return actual_length, token_ids_list, token_texts


def plot_token_activation_strength(token_activation_strengths, token_texts=None, max_tokens_to_plot=None):
    """
    Plot token index vs activation strength (line plot and bar plot).
    
    Args:
        token_activation_strengths: Array of activation strengths per token
        token_texts: Dictionary mapping token index to text (optional)
        max_tokens_to_plot: Maximum number of tokens to plot (None = all)
    
    Returns:
        None (displays plots)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if max_tokens_to_plot is not None:
        token_activation_strengths = token_activation_strengths[:max_tokens_to_plot]
    
    token_indices_plot = np.arange(len(token_activation_strengths))
    
    # Line plot
    plt.figure(figsize=(14, 6))
    plt.plot(token_indices_plot, token_activation_strengths, marker='o', markersize=4, linewidth=1.5)
    plt.xlabel('Token Index (T5 tokens)', fontsize=12, fontweight='bold')
    plt.ylabel('Average Activation Strength', fontsize=12, fontweight='bold')
    plt.title(f'Token Index vs Activation Strength - First {len(token_activation_strengths)} Tokens', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add token text labels
    if token_texts is not None:
        max_labels = min(20, len(token_activation_strengths))
        for i in range(max_labels):
            token_text = token_texts.get(i, f"T{i}")
            if len(token_text) > 15:
                token_text = token_text[:15] + "..."
            plt.annotate(token_text, 
                        (i, token_activation_strengths[i]),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        fontsize=7,
                        rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Bar plot
    plt.figure(figsize=(14, 6))
    plt.bar(token_indices_plot, token_activation_strengths, alpha=0.7, width=0.8)
    plt.xlabel('Token Index (T5 tokens)', fontsize=12, fontweight='bold')
    plt.ylabel('Average Activation Strength', fontsize=12, fontweight='bold')
    plt.title(f'Token Index vs Activation Strength (Bar Plot) - First {len(token_activation_strengths)} Tokens', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add token text labels on x-axis
    if token_texts is not None:
        max_labels = min(30, len(token_activation_strengths))
        tick_positions = list(range(max_labels))
        tick_labels = []
        for i in range(max_labels):
            token_text = token_texts.get(i, f"T{i}")
            if len(token_text) > 10:
                token_text = token_text[:10] + "..."
            tick_labels.append(f"{i}\n{token_text}")
        plt.xticks(tick_positions, tick_labels, rotation=45, ha='right', fontsize=8)
    else:
        step = max(1, len(token_indices_plot) // 30)
        plt.xticks(token_indices_plot[::step], 
                  [f"T{i}" for i in token_indices_plot[::step]],
                  rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"✅ Token activation strength plot generated")
    print(f"   Max activation: {token_activation_strengths.max():.4f} at token {token_activation_strengths.argmax()}")
    print(f"   Min activation: {token_activation_strengths.min():.4f} at token {token_activation_strengths.argmin()}")
    print(f"   Mean activation: {token_activation_strengths.mean():.4f}")


def extract_dit_self_attention_maps(pipe, hook_location, prompt, T_text, T_img, h_img, w_img, 
                                     actual_prompt_length, num_steps=1, max_tokens=None, generator=None):
    """
    Extract self-attention maps from DiT layer for text tokens.
    DiT uses global self-attention on fused sequence [Text (512), Image (T_img)].
    We extract the bottom-left quadrant: Image Queries × Text Keys.
    
    Args:
        pipe: FLUX pipeline
        hook_location: DiT attention module location (e.g., "single_transformer_blocks.37.attn")
        prompt: Text prompt
        T_text: Number of text tokens (512)
        T_img: Number of image tokens
        h_img: Spatial height in tokens
        w_img: Spatial width in tokens
        actual_prompt_length: Actual prompt length (excluding padding)
        num_steps: Number of inference steps
        max_tokens: Maximum number of tokens to extract (None = all)
        generator: Random generator for reproducibility
    
    Returns:
        attention_maps: Dictionary mapping token_idx to (h_img, w_img) attention map
        token_indices: List of token indices that were extracted
    """
    attention_maps = {}
    
    # Determine which tokens to extract
    if max_tokens is not None:
        token_indices = list(range(min(max_tokens, actual_prompt_length)))
    else:
        token_indices = list(range(actual_prompt_length))
    
    def attention_hook(module, input, output):
        """Hook to extract self-attention weights from DiT"""
        # Method 1: Try to access stored attention weights
        if hasattr(module, 'last_attention_weights'):
            attn_weights = module.last_attention_weights
            if attn_weights is not None:
                attn_weights = attn_weights.mean(dim=1)  # Average over heads
                
                # Extract bottom-left quadrant: Image Queries × Text Keys
                image_text_attention = attn_weights[0, T_text:T_text + T_img, :T_text]  # (T_img, T_text)
                
                for token_idx in token_indices:
                    if token_idx < T_text:
                        token_attention = image_text_attention[:, token_idx]  # (T_img,)
                        
                        if token_attention.dtype == torch.bfloat16:
                            token_attention = token_attention.float()
                        
                        try:
                            attention_map = token_attention.reshape(h_img, w_img).cpu().numpy()
                            attention_maps[token_idx] = attention_map
                        except Exception as e:
                            pass
        
        # Method 2: Compute from input hidden states
        elif isinstance(input, tuple) and len(input) >= 1:
            hidden_states = input[0] if len(input) > 0 else None
            
            if hidden_states is not None and len(hidden_states.shape) == 3:
                batch, seq_len, dim = hidden_states.shape
                
                if seq_len == T_text + T_img:
                    if hasattr(module, 'q_proj') and hasattr(module, 'k_proj'):
                        hidden_comp = hidden_states.float() if hidden_states.dtype == torch.bfloat16 else hidden_states
                        
                        queries = module.q_proj(hidden_comp)
                        keys = module.k_proj(hidden_comp)
                        
                        scale = 1.0 / np.sqrt(float(dim))
                        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale
                        attention_scores = torch.softmax(attention_scores, dim=-1)
                        
                        image_text_attention = attention_scores[0, T_text:T_text + T_img, :T_text]
                        
                        for token_idx in token_indices:
                            if token_idx < T_text:
                                token_attention = image_text_attention[:, token_idx]
                                
                                if token_attention.dtype == torch.bfloat16:
                                    token_attention = token_attention.float()
                                
                                try:
                                    attention_map = token_attention.reshape(h_img, w_img).cpu().numpy()
                                    attention_maps[token_idx] = attention_map
                                except Exception as e:
                                    pass
    
    # Register hook
    try:
        dit_attn_module = pipe.transformer.get_submodule(hook_location)
        hook_handle = dit_attn_module.register_forward_hook(attention_hook)
    except Exception as e:
        return {}, []
    
    # Generate to capture attention
    try:
        _ = pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=0.0,
            generator=generator,
        )
    finally:
        if 'hook_handle' in locals():
            hook_handle.remove()
    
    return attention_maps, token_indices


def extract_mmdit_cross_attention_maps(pipe, hook_location, prompt, h_img, w_img, 
                                         actual_prompt_length, num_steps=1, generator=None):
    """
    Extract cross-attention maps from MMDiT attention layer.
    For MMDiT: output is (image_stream, text_stream)
    Image stream (queries) attends to text stream (keys).
    
    Args:
        pipe: FLUX pipeline
        hook_location: MMDiT attention module location (e.g., "transformer_blocks.0.attn")
        prompt: Text prompt
        h_img: Spatial height in tokens
        w_img: Spatial width in tokens
        actual_prompt_length: Actual prompt length (excluding padding)
        num_steps: Number of inference steps
        generator: Random generator for reproducibility
    
    Returns:
        attention_maps: Dictionary mapping token_idx to (h_img, w_img) attention map
    """
    attention_maps = {}
    
    def attention_hook(module, input, output):
        """Hook to extract cross-attention weights from MMDiT"""
        if isinstance(output, tuple) and len(output) == 2:
            image_stream, text_stream = output
            
            if len(image_stream.shape) == 3 and len(text_stream.shape) == 3:
                batch, num_queries, dim = image_stream.shape
                batch_k, num_keys, dim_k = text_stream.shape
                
                image_stream_comp = image_stream.float() if image_stream.dtype == torch.bfloat16 else image_stream
                text_stream_comp = text_stream.float() if text_stream.dtype == torch.bfloat16 else text_stream
                
                scale = 1.0 / np.sqrt(float(dim))
                attention_scores = torch.matmul(image_stream_comp, text_stream_comp.transpose(-2, -1)) * scale
                attention_scores = torch.softmax(attention_scores, dim=-1)
                
                for token_idx in range(min(num_keys, actual_prompt_length)):
                    token_attention = attention_scores[0, :, token_idx]
                    
                    if token_attention.dtype == torch.bfloat16:
                        token_attention = token_attention.float()
                    
                    try:
                        attention_map = token_attention.reshape(h_img, w_img).cpu().numpy()
                        attention_maps[token_idx] = attention_map
                    except Exception as e:
                        pass
        
        elif hasattr(module, 'last_attention_weights'):
            attn_weights = module.last_attention_weights
            if attn_weights is not None:
                attn_weights = attn_weights.mean(dim=1)
                
                for token_idx in range(min(attn_weights.shape[3], actual_prompt_length)):
                    token_attention = attn_weights[0, :, token_idx]
                    
                    if token_attention.dtype == torch.bfloat16:
                        token_attention = token_attention.float()
                    
                    try:
                        attention_map = token_attention.reshape(h_img, w_img).cpu().numpy()
                        attention_maps[token_idx] = attention_map
                    except Exception as e:
                        pass
    
    # Register hook
    try:
        mmdit_attn_module = pipe.transformer.get_submodule(hook_location)
        hook_handle = mmdit_attn_module.register_forward_hook(attention_hook)
    except Exception as e:
        return {}
    
    # Generate to capture attention
    try:
        _ = pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=0.0,
            generator=generator,
        )
    finally:
        if 'hook_handle' in locals():
            hook_handle.remove()
    
    return attention_maps


def create_text_guided_sparse_maps_mmdit(text_sparse_maps, attention_maps_dict, actual_prompt_length, 
                                          h_img, w_img):
    """
    Create text-guided sparse maps for MMDiT text stream SAEs.
    Strategy: Feature → Top Token → Cross-Attention → Image Heatmap
    
    Args:
        text_sparse_maps: Text sparse maps (num_tokens, num_features)
        attention_maps_dict: Dictionary of attention maps per token
        actual_prompt_length: Actual prompt length (excluding padding)
        h_img: Spatial height in tokens
        w_img: Spatial width in tokens
    
    Returns:
        text_guided_sparse_maps: (h_img, w_img, num_features) spatial sparse maps
    """
    num_features = text_sparse_maps.shape[1]
    text_guided_sparse_maps = torch.zeros(h_img, w_img, num_features, 
                                            device=text_sparse_maps.device, dtype=text_sparse_maps.dtype)
    
    # For each feature, find top activating token and use its attention map
    for feat_idx in range(num_features):
        feature_activations = text_sparse_maps[:actual_prompt_length, feat_idx]
        
        # Find top activating token
        top_token_idx = feature_activations.argmax().item()
        top_token_activation = feature_activations[top_token_idx].item()
        
        # Get attention map for this top token
        if top_token_idx in attention_maps_dict:
            attention_map = torch.from_numpy(attention_maps_dict[top_token_idx]).to(
                device=text_sparse_maps.device, dtype=text_sparse_maps.dtype
            )
            
            # Weight the attention map by the feature activation strength
            text_guided_sparse_maps[:, :, feat_idx] = attention_map * top_token_activation
    
    return text_guided_sparse_maps


def create_text_guided_sparse_maps_dit(text_sparse_maps, attention_maps_dict, token_indices, 
                                        h_img, w_img):
    """
    Create text-guided sparse maps for DiT text tokens.
    Strategy: Aggregate text token activations weighted by attention maps.
    
    Args:
        text_sparse_maps: Text sparse maps (num_tokens, num_features)
        attention_maps_dict: Dictionary of attention maps per token
        token_indices: List of token indices to aggregate
        h_img: Spatial height in tokens
        w_img: Spatial width in tokens
    
    Returns:
        text_guided_sparse_maps: (h_img, w_img, num_features) spatial sparse maps
    """
    num_features = text_sparse_maps.shape[1]
    text_guided_sparse_maps = torch.zeros(h_img, w_img, num_features, 
                                            device=text_sparse_maps.device, dtype=text_sparse_maps.dtype)
    
    # Aggregate: for each image position, sum over text tokens weighted by attention
    for token_idx in token_indices:
        if token_idx in attention_maps_dict:
            attention_map = torch.from_numpy(attention_maps_dict[token_idx]).to(
                device=text_sparse_maps.device, dtype=text_sparse_maps.dtype
            )
            token_activations = text_sparse_maps[token_idx, :]  # (features,)
            
            # Weight by attention: attention_map (h, w) * token_activations (features)
            for feat_idx in range(num_features):
                text_guided_sparse_maps[:, :, feat_idx] += attention_map * token_activations[feat_idx]
    
    return text_guided_sparse_maps

