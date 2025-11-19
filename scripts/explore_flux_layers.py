#!/usr/bin/env python3
"""
Explore the FLUX transformer structure and show how Hugging Face's raw checkpoints
(`double_blocks.*`) are converted into Diffusers naming (`transformer_blocks.*`).

Diffusers performs this renaming inside `scripts/convert_flux_to_diffusers.py` in the
official repository, so tools like `FluxPipeline` never expose `img_attn`/`txt_attn`
directly once weights are converted.
"""

from __future__ import annotations

import textwrap

import torch
from diffusers import FluxPipeline

HF_CONVERT_SCRIPT = (
    "https://github.com/huggingface/diffusers/blob/"
    "b7df4a5387fd65edf4f16fd5fc8c87a7c815a4c7/scripts/convert_flux_to_diffusers.py"
)

DOUBLE_BLOCK_MAPPING = [
    ("double_blocks.{i}.img_attn.qkv",      "transformer_blocks.{i}.attn.to_{q,k,v}"),
    ("double_blocks.{i}.img_attn.proj",     "transformer_blocks.{i}.attn.to_out.0"),
    ("double_blocks.{i}.txt_attn.qkv",      "transformer_blocks.{i}.attn.add_{q,k,v}_proj"),
    ("double_blocks.{i}.img_mlp.0",         "transformer_blocks.{i}.ff.net.0.proj"),
    ("double_blocks.{i}.img_mlp.2",         "transformer_blocks.{i}.ff.net.2"),
    ("double_blocks.{i}.txt_mlp.0",         "transformer_blocks.{i}.ff_context.net.0.proj"),
    ("double_blocks.{i}.txt_mlp.2",         "transformer_blocks.{i}.ff_context.net.2"),
    ("double_blocks.{i}.img_attn.norm.*",   "transformer_blocks.{i}.attn.norm_{q,k}"),
    ("double_blocks.{i}.txt_attn.norm.*",   "transformer_blocks.{i}.attn.norm_added_{q,k}"),
]

SINGLE_BLOCK_MAPPING = [
    ("single_blocks.{i}.linear1[q,k,v]",    "single_transformer_blocks.{i}.attn.to_{q,k,v}"),
    ("single_blocks.{i}.linear1.mlp",       "single_transformer_blocks.{i}.proj_mlp"),
    ("single_blocks.{i}.norm.query_norm",   "single_transformer_blocks.{i}.attn.norm_q"),
    ("single_blocks.{i}.norm.key_norm",     "single_transformer_blocks.{i}.attn.norm_k"),
    ("single_blocks.{i}.linear2",           "single_transformer_blocks.{i}.proj_out"),
    ("single_blocks.{i}.modulation.lin",    "single_transformer_blocks.{i}.norm.linear"),
]

def print_conversion_mapping():
    """Summarize how double_blocks and single_blocks map onto Diffusers modules."""
    print("\n" + "=" * 80)
    print("HUGGINGFACE ➜ DIFFUSERS NAMING CONVERSION SUMMARY")
    print("=" * 80)
    print(f"Reference implementation: {HF_CONVERT_SCRIPT}")

    def display_mapping(title, mapping):
        print(f"\n{title}")
        print("-" * len(title))
        for src, dst in mapping:
            print(f"  • {src:<35} → {dst}")

    display_mapping("Double-stream (MMDiT) blocks (0-18):", DOUBLE_BLOCK_MAPPING)
    display_mapping("Single-stream (DiT) blocks (0-37):", SINGLE_BLOCK_MAPPING)

    print(
        "\nDiffusers rewrites the raw checkpoint before saving, so the modules exposed by "
        "`FluxPipeline` already use the transformer_blocks/single_transformer_blocks naming. "
        "This is why `double_blocks.*.img_attn/txt_attn` no longer appear once the model is "
        "loaded through Diffusers."
    )


def print_module_overview(transformer):
    print("\n" + "=" * 80)
    print("FLUX TRANSFORMER ARCHITECTURE OVERVIEW")
    print("=" * 80)
    attrs = [a for a in dir(transformer) if not a.startswith("_")]
    buckets = [a for a in attrs if "block" in a.lower()]
    for attr in buckets:
        obj = getattr(transformer, attr, None)
        if obj is None:
            continue
        length = len(obj) if hasattr(obj, "__len__") else "N/A"
        print(f"  • {attr}: {type(obj).__name__} (length={length})")


def inspect_block(transformer, index=0):
    blocks = getattr(transformer, "transformer_blocks", [])
    if not blocks:
        return
    block = blocks[index]
    print("\n" + "=" * 80)
    print(f"INSPECTING transformer_blocks[{index}]")
    print("=" * 80)
    for name, module in block.named_modules():
        if not name:
            continue
        if any(key in name for key in ("attn", "ff", "norm")):
            print(f"  {name:<35} → {type(module).__name__}")


def test_locations(transformer):
    locations = [
        "transformer_blocks.0.attn",
        "transformer_blocks.0.ff",
        "transformer_blocks.18.attn",
        "single_transformer_blocks.0",
        "single_transformer_blocks.18",
        "single_transformer_blocks.37",
    ]
    print("\n" + "=" * 80)
    print("TESTING ACCESSIBLE MODULE PATHS")
    print("=" * 80)
    for loc in locations:
        try:
            module = transformer.get_submodule(loc)
            note = "tuple output (use --stream)" if "attn" in loc else "single tensor"
            print(f"  ✅ {loc:<35} → {type(module).__name__} [{note}]")
        except Exception as exc:
            print(f"  ❌ {loc:<35} → {exc}")


def recommend_locations(transformer):
    print("\n" + "=" * 80)
    print("SUGGESTED SAE HOOK LOCATIONS (GROUPED BY RAW BLOCK TYPE)")
    print("=" * 80)

    groups = [
        (
            "double_blocks → transformer_blocks (attention modules)",
            [
                ("transformer_blocks.0.attn", 0, "Stream 0 = query output (block 0)"),
                ("transformer_blocks.0.attn", 1, "Stream 1 = key output (block 0)"),
                ("transformer_blocks.18.attn", 1, "Stream 1 = key output (block 18)"),
            ],
        ),
        (
            "double_blocks → transformer_blocks (MLP modules)",
            [
                ("transformer_blocks.0.ff", 0, "Image MLP output (block 0)"),
                ("transformer_blocks.0.ff_context", 0, "Text/context MLP output (block 0)"),
                ("transformer_blocks.18.ff", 0, "Image MLP output (block 18)"),
            ],
        ),
    ]

    if hasattr(transformer, "single_transformer_blocks"):
        groups.extend(
            [
                (
                    "single_blocks → single_transformer_blocks (attention only)",
                    [
                        ("single_transformer_blocks.0.attn", 0, "Early DiT attention (stream 0 = query)"),
                        ("single_transformer_blocks.0.attn", 1, "Early DiT attention (stream 1 = key)"),
                        ("single_transformer_blocks.18.attn", 1, "Mid DiT attention (stream 1)"),
                    ],
                ),
                (
                    "single_blocks → single_transformer_blocks (MLP / block outputs)",
                    [
                        ("single_transformer_blocks.0.proj_mlp", 0, "Early DiT MLP projection"),
                        ("single_transformer_blocks.18.proj_mlp", 0, "Mid DiT MLP projection"),
                        ("single_transformer_blocks.37", 0, "Late DiT full block output (attn+MLP fused)"),
                    ],
                ),
            ]
        )

    for title, items in groups:
        print(f"\n{title}")
        print("-" * len(title))
        for loc, stream, desc in items:
            print(f"  • {loc:<40} stream={stream} → {desc}")
            print(f"      e.g. --loc \"{loc}\" --stream {stream}")


def explain_streams():
    print("\n" + "=" * 80)
    print("WHAT DOES `--stream` MEAN?")
    print("=" * 80)
    print(
        textwrap.dedent(
            """
            Attention modules return tuples (query, key[, value]). Diffusers surfaces them as:
              • stream 0 → query activations
              • stream 1 → key activations

            Most SAE training scripts expect a single tensor. When hooking attention, pass
            `--stream 0` or `--stream 1` to pick which element of the tuple is fed into the SAE.
            MLP / block outputs return a single tensor, so the stream argument is ignored.
            """
        ).strip()
    )


def explore_flux_layers():
    print("Loading FLUX.1-schnell via Diffusers (bfloat16 weights)...")
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    transformer = pipe.transformer

    print_conversion_mapping()
    print_module_overview(transformer)
    inspect_block(transformer, index=0)
    test_locations(transformer)
    explain_streams()
    recommend_locations(transformer)


if __name__ == "__main__":
    explore_flux_layers()

