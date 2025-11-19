from pathlib import Path

import click
import polars as pl
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import FluxPipeline
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup

from autoencoder import (
    GatedAutoEncoder,
    GatedTrainer,
    SparseAutoencoder,
    StandardTrainer,
    TopkSparseAutoencoder,
    TopkTrainer,
)


class FluxActivationSampler:
    def __init__(self, tag:str, loc:str):
        self.handle = None
        self.output = None
        self.timestamps:list[tuple[float,float]] = []
        self.loc = loc
        
        self.pipe = FluxPipeline.from_pretrained(tag, torch_dtype=torch.bfloat16)
        self.pipe.vae = torch.compile(self.pipe.vae)
        self.pipe.text_encoder = torch.compile(self.pipe.text_encoder)
        self.pipe.text_encoder_2 = torch.compile(self.pipe.text_encoder_2)

    def __exit__(self, exc_type, exc_value, traceback):
        self.handle.remove()
        self.handle = None
        self.output = None
        self.timestamps = []

    def __enter__(self):
        def _set(m, args, output):
            self.output = output
        self.handle = self.pipe.transformer.get_submodule(self.loc).register_forward_hook(_set)

        return self
    
    def __call__(self, *args, **kwargs):
        def callback(pipe:FluxPipeline, step: int, timestep: int, callback_kwargs: dict):
            self.timestamps.append((step, timestep))
            return callback_kwargs
        
        output = self.pipe(*args, **kwargs, callback_on_step_end=callback)

        return { "activations": self.output, "outputs": output }
    
class CC3MPromptDataset(Dataset):
    def __init__(self, folder: Path | str = None, shuffle:bool=False):
        if folder is None:
            # Try both common paths
            folder = Path("/mnt/drive_a/Projects/sae/data/cc3m/")
            if not folder.exists():
                folder = Path("/data/cc3m/")
        self.folder = Path(folder)
        
        data = []
        for file in self.folder.glob("*.parquet"):
            data.append(pl.read_parquet(file, columns=["conversations"]))
        self.dataset = pl.concat(data)
        if shuffle:
            self.dataset = self.dataset.sample(fraction=1.0, shuffle=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.row(idx, named=True)
        text = row["conversations"][-1]['value']

        return text


def train(**config):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[kwargs])
    accelerator.init_trackers("fluxsae", config=config, init_kwargs={"wandb":{"name":config["name"]}})
    
    match config["dataset"]:
        case "cc3m":
            dataset_folder = config.get("dataset_folder") or None
            shuffle_dataset = config.get("shuffle_dataset", False)
            dataset = CC3MPromptDataset(folder=dataset_folder, shuffle=shuffle_dataset)
        case _:
            raise ValueError(f"Unknown dataset {config['dataset']}")
    
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    sampler = FluxActivationSampler("black-forest-labs/FLUX.1-schnell", loc=config["loc"])

    sampler.pipe.transformer, sampler.pipe.vae, sampler.pipe.text_encoder, sampler.pipe.text_encoder_2 = accelerator.prepare(sampler.pipe.transformer, sampler.pipe.vae, sampler.pipe.text_encoder, sampler.pipe.text_encoder_2)
    dataloader = accelerator.prepare(dataloader)

    # Auto-detect activation dimension from first batch to validate SAE configuration
    print("Auto-detecting activation dimension...")
    dataloader_iter = iter(dataloader)
    with torch.no_grad():
        first_batch = next(dataloader_iter)
        with sampler as s:
            test_outputs = s(first_batch, height=256, width=256, guidance_scale=0., max_sequence_length=256, num_inference_steps=1,)
            test_outputs = test_outputs["activations"]
            # Handle both tuple outputs (attention) and single tensor outputs (MLP, single blocks)
            if isinstance(test_outputs, tuple):
                # Attention modules return tuple (query, key)
                if config["stream"] == 0:
                    test_x, _ = test_outputs
                elif config["stream"] == 1:
                    _, test_x = test_outputs
                else:
                    raise ValueError(f"Invalid stream {config['stream']} for tuple output. Use 0 (query) or 1 (key).")
            else:
                # Single tensor output (MLP/FF, single_transformer_blocks, etc.)
                test_x = test_outputs
            test_x = rearrange(test_x, "b ... d -> (b ...) d")
            actual_features = test_x.shape[-1]
    
    # Validate or auto-correct feature dimension
    if config["features"] != actual_features:
        print(f"WARNING: Configured features ({config['features']}) doesn't match actual activation dimension ({actual_features})!")
        print(f"Auto-correcting to use activation dimension: {actual_features}")
        config["features"] = actual_features
    
    # Create SAE with correct (or auto-corrected) feature dimension
    pages = config["expansion"] * config["features"]
    match config["arch"]:
        case "standard":
            sae = SparseAutoencoder(features=config["features"], pages=pages)
        case "topk":
            sae = TopkSparseAutoencoder(features=config["features"], pages=pages, k=config["k"])
        case "gated":
            sae = GatedAutoEncoder(features=config["features"], pages=pages)
        case _:
            raise ValueError(f"Unknown architecture: {config['arch']}")
    
    optimizer = torch.optim.AdamW(sae.parameters(), lr=config["lr"], betas=(config["beta1"], config["beta2"]), weight_decay=config["wd"])
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=config["lr_warmup_steps"])
    sae, optimizer, scheduler = accelerator.prepare(sae, optimizer, scheduler)
    
    print(f"Using activation dimension: {config['features']}, SAE pages: {pages}")

    steps = 0

    match config["arch"]:
        case "standard":
            trainer = StandardTrainer(sae, optimizer, scheduler, lmbda=config["lmbda"], lmbda_warmup_steps=config["lmbda_warmup_steps"], accelerator=accelerator)
        case "topk":
            trainer = TopkTrainer(sae, optimizer, scheduler, pages=pages, auxk=config["auxk"], bodycount=config["bodycount"], normalise=config["normalise"], accelerator=accelerator)
        case "gated":
            trainer = GatedTrainer(sae, optimizer, scheduler, lmbda=config["lmbda"], lmbda_warmup_steps=config["lmbda_warmup_steps"], accelerator=accelerator)
        
    sae.train()
    # Continue with the dataloader (first batch was used for dimension detection)
    # Since dataloader shuffles, this is fine - we'll process all batches
    for prompts in tqdm(dataloader_iter, total=len(dataloader) - 1, desc="Training"):
        with sampler as s:
            outputs = s(prompts, height=256, width=256, guidance_scale=0., max_sequence_length=256, num_inference_steps=1,)
            outputs = outputs["activations"]
            # Handle both tuple outputs (attention) and single tensor outputs (MLP, single blocks)
            if isinstance(outputs, tuple):
                # Attention modules return tuple (query, key)
                if config["stream"] == 0:
                    x, _ = outputs
                elif config["stream"] == 1:
                    _, x = outputs
                else:
                    raise ValueError(f"Invalid stream {config['stream']} for tuple output. Use 0 (query) or 1 (key).")
            else:
                # Single tensor output (MLP/FF, single_transformer_blocks, etc.)
                x = outputs

            x = rearrange(x, "b ... d -> (b ...) d")

            bdots, _ = x.shape
            shuffle = torch.randperm(bdots, device=x.device)
            x = x[shuffle][:config["nsamples"]] # sample a subset of the activations
            
        trainer.step(x)
        steps += 1

        if steps > config["iters"]:
            break

    accelerator.wait_for_everyone()

    if config["savedir"] is not None:
        savedir = Path(config["savedir"]) / config["name"]
        savedir.mkdir(parents=True, exist_ok=True)
        _model = accelerator.unwrap_model(sae)
        _model.save_pretrained(savedir)
    
    accelerator.end_training()


@click.command()
@click.option("--name", type=str, help="Name of the run")
@click.option("--dataset", type=str, default="cc3m", help="Dataset to use (currently only 'cc3m' is supported)")
@click.option("--dataset-folder", type=str, default=None, help="Path to dataset folder (default: auto-detect)")
@click.option("--shuffle-dataset/--no-shuffle-dataset", default=False, help="Shuffle dataset at initialization")
@click.option("--arch", type=str, default="standard")
@click.option("--num_workers", type=int, default=96)
@click.option("--batch_size", type=int, default=32)
@click.option("--features", type=int, default=3072)
@click.option("--expansion", type=int, default=4)
@click.option("--lr", type=float, default=5e-5)
@click.option("--beta1", type=float, default=0.9)
@click.option("--beta2", type=float, default=0.999)
@click.option("--wd", type=float, default=0.)
@click.option("--lr_warmup_steps", type=int, default=256)
@click.option("--k", type=int, default=16)
@click.option("--auxk", type=float, default=1/32)
@click.option("--bodycount", type=int, default=16384)
@click.option("--savedir", type=str, default="./checkpoints")
@click.option("--lmbda", type=float, default=0.01)
@click.option("--lmbda_warmup_steps", type=int, default=256)
@click.option("--loc", type=str, default="transformer_blocks.0.attn")
@click.option("--stream", type=int, default=1)
@click.option("--iters", type=int, default=4096)
@click.option("--nsamples", type=int, default=256)
@click.option("--normalise", type=bool, default=False)
def main(**config):
    train(**config)


if __name__ == "__main__":
    main()
    