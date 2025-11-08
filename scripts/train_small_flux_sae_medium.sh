#!/bin/bash
# Medium-sized FLUX SAE training script
# A compromise between small (testing) and full-sized (production)

# Medium SAE configuration:
# - features: 1536 (half of full 3072)
# - expansion: 3 (instead of 4) - fewer SAE features (4608 total)
# - batch_size: 16 (half of full 32)
# - iters: 2000 (half of full 4096)
# - nsamples: 256 (full amount)
# - arch: topk (more efficient)

python fluxsae.py \
    --name "medium-flux-sae-test" \
    --dataset cc3m \
    --arch topk \
    --batch_size 16 \
    --features 1536 \
    --expansion 3 \
    --lr 5e-5 \
    --lr_warmup_steps 256 \
    --k 16 \
    --auxk 0.03125 \
    --bodycount 16384 \
    --savedir ./checkpoints \
    --lmbda 0.01 \
    --lmbda_warmup_steps 256 \
    --loc "transformer_blocks.0.attn" \
    --stream 1 \
    --iters 2000 \
    --nsamples 256 \
    --normalise false \
    --num_workers 8

echo "Training complete! Checkpoint saved to ./checkpoints/medium-flux-sae-test/"

