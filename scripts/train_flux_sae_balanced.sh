#!/bin/bash
# Balanced expansion SAE - good for most use cases
# Recommended for middle layers (blocks 18-24)

python fluxsae.py \
    --name "flux-sae-mid-balanced" \
    --dataset cc3m \
    --arch topk \
    --batch_size 8 \
    --features 3072 \
    --expansion 4 \
    --lr 5e-5 \
    --lr_warmup_steps 256 \
    --k 32 \
    --auxk 0.03125 \
    --bodycount 16384 \
    --savedir ./checkpoints \
    --lmbda 0.01 \
    --lmbda_warmup_steps 256 \
    --loc "transformer_blocks.18.attn" \
    --stream 1 \
    --iters 2000 \
    --nsamples 256 \
    --normalise false \
    --num_workers 4

echo "Training complete! Checkpoint saved to ./checkpoints/flux-sae-mid-balanced/"
echo "Dictionary size: 12,288 features (expansion=4 × 3072)"
echo "Active features per sample: 32 (0.26% sparsity)"

