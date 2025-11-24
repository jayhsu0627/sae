"""
Utility functions for FLUX SAE steering - Adapted from SDXL-turbo example
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
    with torch.no_grad():
        target_steered = strength * feature_direction.unsqueeze(0)
    
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

