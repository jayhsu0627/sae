#!/bin/bash
# Script to train a small FLUX SAE for testing
# This uses smaller parameters to reduce memory usage and training time

# Small SAE configuration:
# - features: 3072 (FLUX activation dimension - auto-detected if wrong)
# - expansion: 2 (instead of 4) - fewer SAE features (6144 total)
# - batch_size: 8 (instead of 32) - 4x smaller batches
# - iters: 1000 (instead of 4096) - faster testing
# - nsamples: 128 (instead of 256) - fewer activations per batch
# - arch: topk (more efficient than standard)
# - loc: transformer_blocks.0.attn (early layer, less memory)

python fluxsae.py \
    --name "small-flux-sae-test" \
    --dataset cc3m \
    --arch topk \
    --batch_size 8 \
    --features 3072 \
    --expansion 2 \
    --lr 5e-5 \
    --lr_warmup_steps 128 \
    --k 16 \
    --auxk 0.03125 \
    --bodycount 8192 \
    --savedir ./checkpoints \
    --lmbda 0.01 \
    --lmbda_warmup_steps 128 \
    --loc "transformer_blocks.0.attn" \
    --stream 1 \
    --iters 1000 \
    --nsamples 128 \
    --normalise false \
    --num_workers 4

echo "Training complete! Checkpoint saved to ./checkpoints/small-flux-sae-test/"

