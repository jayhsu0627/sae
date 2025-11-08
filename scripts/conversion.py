import io
import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
from PIL import Image
from tqdm import tqdm
import polars as pl

import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor, CLIPTextModel, CLIPVisionModel

from accelerate import Accelerator
from huggingface_hub import HfFileSystem
from safetensors.torch import save_file


class CC3M(Dataset):
    def __init__(self, path:Path):
        data = []
        for file in path.glob("*.parquet"):
            data.append(pl.read_parquet(file))
        self.dataset = pl.concat(data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.row(idx, named=True)
        image = Image.open(io.BytesIO(row['image']['bytes']))
        text = row["conversations"][-1]['value']
        return image, text


@click.group()
def cli():
    pass

# for generating clip embeddings on cc3m and saving them to safetensor file
@cli.command("embed")
@click.option("--batch-size", default=256, type=int, help="Batch size for processing (default: 256, increase for more GPU utilization but uses more memory)")
@click.option("--num-workers", default=8, type=int, help="Number of data loading workers (default: 8)")
@click.option("--process-separately/--process-together", default=True, help="Process text and vision separately to reduce peak memory (default: separately)")
def embed(batch_size, num_workers, process_separately):
    accelerator = Accelerator(mixed_precision="bf16")

    # Load the processor and the model
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    vision = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", attn_implementation="sdpa")
    text = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", attn_implementation="sdpa")
    cc3m = CC3M(Path("/mnt/drive_a/Projects/sae/data/cc3m/"))

    def collate_fn(batch):
        images = [im for im, _ in batch]
        texts = [txt for _, txt in batch]

        return processor(text=texts, images=images, return_tensors="pt", truncation=True, padding=True)

    # Optimize DataLoader for speed: parallel workers, pinned memory, prefetching
    # Reduced prefetch_factor to save memory - prefetching multiple large batches can consume significant RAM
    dataloader = torch.utils.data.DataLoader(
        cc3m, 
        batch_size=batch_size, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,  # Faster CPU->GPU transfers
        prefetch_factor=1 if num_workers > 0 else None,  # Reduced from 2 to save memory
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
    )
    print(f"Using batch_size={batch_size}, num_workers={num_workers}, process_separately={process_separately}")
    
    # Compile models for speed (but this may increase initial memory usage)
    vision = torch.compile(vision)
    text = torch.compile(text)
    vision, text, dataloader = accelerator.prepare(vision, text, dataloader)

    # Get embedding dimensions from model config (no need to process a batch)
    text_dim = text.config.hidden_size  # 768 for CLIP ViT-Large
    vision_dim = vision.config.hidden_size  # 1024 for CLIP ViT-Large
    
    total_samples = len(cc3m)
    output_path = Path("/mnt/drive_a/Projects/sae/data/cc3m/embeddings.safetensors")
    
    # Pre-allocate tensors to avoid memory doubling during concatenation
    # NOTE: This only accounts for final embedding storage (~19GB for 3M samples)
    # Actual peak memory during processing is much higher due to:
    # - Model weights (~1-2GB for CLIP ViT-Large)
    # - Batch activations (batch_size × image_size × hidden_size × num_layers)
    # - Attention matrices (batch_size × seq_len × seq_len)
    # - DataLoader prefetching
    # With batch_size=256: peak memory ~30-50GB
    # With batch_size=4096: peak memory ~200GB+ (why you're seeing this!)
    print(f"Pre-allocating tensors for {total_samples:,} samples...")
    print(f"  Text: {total_samples:,} × {text_dim} = {total_samples * text_dim * 4 / 1e9:.2f} GB")
    print(f"  Vision: {total_samples:,} × {vision_dim} = {total_samples * vision_dim * 4 / 1e9:.2f} GB")
    print(f"  Total embeddings: {(total_samples * text_dim * 4 + total_samples * vision_dim * 4) / 1e9:.2f} GB")
    print(f"  NOTE: Peak memory during processing will be MUCH higher due to:")
    print(f"    - Model weights (~2-3GB for CLIP ViT-Large)")
    print(f"    - Batch activations (scales with batch_size)")
    print(f"    - Attention matrices (batch_size × seq_len²)")
    print(f"    - DataLoader prefetching")
    print(f"  With batch_size={batch_size}: expect ~30-80GB peak (much less than 200GB!)")
    print(f"  With batch_size=4096: expect ~200GB+ peak (why you saw this!)")
    
    text_embeddings = torch.zeros(total_samples, text_dim, dtype=torch.float32)
    vision_embeddings = torch.zeros(total_samples, vision_dim, dtype=torch.float32)
    
    # Process batches and write directly into pre-allocated tensors
    # This avoids the memory doubling issue - no torch.cat() needed!
    current_idx = 0
    start_time = time.time()
    
    # Process with optimized batching
    with torch.no_grad():  # Move no_grad outside loop for efficiency
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing embeddings"):
            if process_separately:
                # Process text and vision separately to reduce peak memory
                # This way we don't have both model activations in memory simultaneously
                text_outputs = text(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                text_outputs = text_outputs.pooler_output.cpu()
                del batch["input_ids"], batch["attention_mask"]  # Free memory immediately
                
                # Clear GPU cache between model runs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                vision_model_outputs = vision(pixel_values=batch["pixel_values"], output_hidden_states=True)
                vision_outputs = vision_model_outputs.hidden_states[-2][:, 0, :].cpu()
                del batch["pixel_values"], vision_model_outputs  # Free memory immediately
            else:
                # Process simultaneously (uses more memory but may be slightly faster)
                text_outputs = text(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                text_outputs = text_outputs.pooler_output.cpu()
                vision_model_outputs = vision(pixel_values=batch["pixel_values"], output_hidden_states=True)
                vision_outputs = vision_model_outputs.hidden_states[-2][:, 0, :].cpu()
                del vision_model_outputs  # Free intermediate outputs
            
            actual_batch_size = text_outputs.shape[0]
            # Write directly into pre-allocated tensors (no memory doubling!)
            text_embeddings[current_idx:current_idx + actual_batch_size] = text_outputs
            vision_embeddings[current_idx:current_idx + actual_batch_size] = vision_outputs
            current_idx += actual_batch_size
            
            # Clear references to help garbage collection
            del text_outputs, vision_outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Periodic progress updates
            if (idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                samples_per_sec = current_idx / elapsed
                remaining = (total_samples - current_idx) / samples_per_sec if samples_per_sec > 0 else 0
                if torch.cuda.is_available():
                    memory_gb = torch.cuda.max_memory_allocated() / 1e9
                    torch.cuda.reset_peak_memory_stats()
                    print(f"  Processed {current_idx:,}/{total_samples:,} samples "
                          f"({samples_per_sec:.1f} samples/sec, ~{remaining/60:.1f} min remaining, "
                          f"peak GPU: {memory_gb:.1f}GB)")
                else:
                    print(f"  Processed {current_idx:,}/{total_samples:,} samples "
                          f"({samples_per_sec:.1f} samples/sec, ~{remaining/60:.1f} min remaining)")
    
    print(f"Processing complete. Saving to {output_path}...")
    # Save the embeddings to a file (no concatenation needed, already in single tensors)
    save_file({ 
        "vision": vision_embeddings, 
        "text": text_embeddings,
    }, str(output_path))
    print("Saved successfully!")  

# for writing images to disk
@cli.command("write")
@click.option("--max-workers", default=None, type=int, help="Number of parallel workers (default: auto-detect based on CPU count)")
def write(max_workers):
    datadir = Path("/mnt/drive_a/Projects/sae/data/cc3m/")
    outdir = Path("/mnt/drive_a/Projects/sae/data/cc3m/images/")
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect optimal worker count if not specified
    # For I/O-bound tasks like image writing, we can use more workers than CPU cores
    if max_workers is None:
        cpu_count = os.cpu_count() or 8
        max_workers = min(32, max(4, cpu_count * 2))
        print(f"Detected {cpu_count} CPU cores, using {max_workers} parallel workers")
    
    # Get all parquet files
    parquet_files = sorted(datadir.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")
    
    def write_image_from_row(parquet_name, local_idx, row, global_idx):
        """Write a single image to disk and return caption"""
        # Create folder for this parquet file (remove .parquet extension)
        folder_name = parquet_name.stem  # e.g., "train-00000-of-00281"
        file_folder = outdir / folder_name
        file_folder.mkdir(parents=True, exist_ok=True)
        
        # Use local index within the parquet file for filename
        output_path = file_folder / f"{local_idx}.jpg"
        
        # Skip if file already exists and is valid (non-zero size)
        if output_path.exists() and output_path.stat().st_size > 0:
            # Still need to get caption for JSON even if image is skipped
            try:
                text = row["conversations"][-1]['value']
                return global_idx, "skipped", text, str(output_path.relative_to(outdir.parent))
            except Exception as e:
                return global_idx, "skipped", None, None
        
        try:
            image = Image.open(io.BytesIO(row['image']['bytes']))
            text = row["conversations"][-1]['value']
            image.save(output_path, "JPEG", quality=95)
            return global_idx, "success", text, str(output_path.relative_to(outdir.parent))
        except Exception as e:
            # Clean up partial/corrupted file on error
            if output_path.exists():
                output_path.unlink()
            return global_idx, f"error: {str(e)}", None, None
    
    # Process all parquet files
    captions = {}  # Dictionary to store global_idx -> caption mapping
    image_paths = {}  # Dictionary to store global_idx -> relative path mapping
    all_results = []
    global_idx = 0
    
    for parquet_file in parquet_files:
        print(f"\nProcessing {parquet_file.name}...")
        df = pl.read_parquet(parquet_file)
        total_rows = len(df)
        
        def process_row(local_idx, global_idx_val):
            """Process a single row from the dataframe"""
            row = df.row(local_idx, named=True)
            return write_image_from_row(parquet_file, local_idx, row, global_idx_val)
        
        # Prepare tasks: (local_idx, global_idx)
        tasks = [(local_idx, global_idx + local_idx) for local_idx in range(total_rows)]
        
        # Process rows in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_row, local_idx, gidx): (local_idx, gidx) 
                      for local_idx, gidx in tasks}
            
            # Process completed writes
            for future in tqdm(as_completed(futures), total=total_rows, desc=f"  {parquet_file.name}"):
                gidx, status, caption, rel_path = future.result()
                all_results.append((gidx, status))
                
                # Store caption and path if available
                if caption is not None:
                    captions[str(gidx)] = caption
                if rel_path is not None:
                    image_paths[str(gidx)] = rel_path
        
        global_idx += total_rows
    
    # Save captions to JSON file
    captions_path = outdir.parent / "captions.json"
    with open(captions_path, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)
    print(f"\nCaptions saved to: {captions_path}")
    print(f"Total captions: {len(captions)}")
    
    # Save image paths mapping to JSON file (optional, for reference)
    paths_path = outdir.parent / "image_paths.json"
    with open(paths_path, "w", encoding="utf-8") as f:
        json.dump(image_paths, f, ensure_ascii=False, indent=2)
    print(f"Image paths saved to: {paths_path}")
    
    # Print summary
    success = sum(1 for _, status in all_results if status == "success")
    skipped = sum(1 for _, status in all_results if status == "skipped")
    errors = [(idx, status) for idx, status in all_results if status != "success" and status != "skipped"]
    
    print(f"\nWrite complete: {success} written, {skipped} skipped")
    if errors:
        print(f"\nErrors: {len(errors)} images had issues")
        for idx, status in errors[:10]:  # Show first 10 errors
            print(f"  - Image {idx}: {status}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")


# for downloading LLaVA recap cc3m dataset
@cli.command("download")
@click.option("--max-workers", default=None, type=int, help="Number of parallel download workers (default: auto-detect based on CPU count)")
def download(max_workers):
    # Auto-detect optimal worker count if not specified
    # For I/O-bound tasks like downloads, we can use more workers than CPU cores
    # Cap at 32 to avoid overwhelming network/rate limits
    if max_workers is None:
        cpu_count = os.cpu_count() or 8
        max_workers = min(32, max(4, cpu_count * 2))
        print(f"Detected {cpu_count} CPU cores, using {max_workers} parallel workers")
    fs = HfFileSystem()
    outdir = Path("/mnt/drive_a/Projects/sae/data/cc3m")
    outdir.mkdir(parents=True, exist_ok=True)
    files = fs.glob("datasets/lmms-lab/LLaVA-ReCap-CC3M/data/*.parquet")
    
    def download_file(hf_path):
        """Download a single file from HuggingFace"""
        file_path = Path(hf_path)
        output_path = outdir / file_path.name
        
        # Re-download if file doesn't exist, is zero bytes, or is suspiciously small (< 1KB)
        # Parquet files should be much larger, so this catches corrupted downloads
        if output_path.exists() and output_path.stat().st_size > 1024:
            return hf_path, "skipped"
        
        # If file exists but is zero bytes or too small, remove it and re-download
        if output_path.exists():
            output_path.unlink()
        
        try:
            with fs.open(hf_path, "rb") as r:
                with open(output_path, "wb") as w:
                    w.write(r.read())
            
            # Verify download completed successfully
            if output_path.stat().st_size == 0:
                output_path.unlink()
                return hf_path, "error: downloaded file is zero bytes"
            
            return hf_path, "success"
        except Exception as e:
            # Clean up partial download on error
            if output_path.exists():
                output_path.unlink()
            return hf_path, f"error: {str(e)}"
    
    # Download files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        futures = {executor.submit(download_file, file): file for file in files}
        
        # Process completed downloads with progress bar
        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            result = future.result()
            results.append(result)
    
    # Print summary
    success = sum(1 for _, status in results if status == "success")
    skipped = sum(1 for _, status in results if status == "skipped")
    errors = [(f, status) for f, status in results if status != "success" and status != "skipped"]
    
    print(f"\nDownload complete: {success} downloaded, {skipped} skipped")
    if errors:
        print(f"\nErrors: {len(errors)} files had issues")
        for f, status in errors[:10]:  # Show first 10 errors
            print(f"  - {Path(f).name}: {status}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors") 

if __name__ == "__main__":
    cli()



