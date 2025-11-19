#!/bin/bash
# High expansion SAE for capturing rich, high-level concepts
# Recommended for late layers (blocks 37-47)

python fluxsae.py \
    --name "flux-sae-late-rich-concepts" \
    --dataset cc3m \
    --arch topk \
    --batch_size 8 \
    --features 3072 \
    --expansion 8 \
    --lr 5e-5 \
    --lr_warmup_steps 256 \
    --k 64 \
    --auxk 0.03125 \
    --bodycount 16384 \
    --savedir ./checkpoints \
    --lmbda 0.01 \
    --lmbda_warmup_steps 256 \
    --loc "single_transformer_blocks.37" \
    --stream 0 \
    --iters 2000 \
    --nsamples 256 \
    --normalise false \
    --num_workers 4

echo "Training complete! Checkpoint saved to ./checkpoints/flux-sae-late-rich-concepts/"
echo "Dictionary size: 24,576 features (expansion=8 × 3072)"
echo "Active features per sample: 64 (0.26% sparsity)"

