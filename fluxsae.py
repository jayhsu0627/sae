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
    init_pre_bias_from_data,
    compute_geometric_median,
)


class FluxActivationSampler:
    def __init__(self, tag:str, loc:str, use_residual:bool=False):
        self.handle_input = None
        self.handle_output = None
        self.input_activation = None
        self.output_activation = None
        self.timestamps:list[tuple[float,float]] = []
        self.loc = loc
        self.use_residual = use_residual
        
        self.pipe = FluxPipeline.from_pretrained(tag, torch_dtype=torch.bfloat16)
        self.pipe.vae = torch.compile(self.pipe.vae)
        self.pipe.text_encoder = torch.compile(self.pipe.text_encoder)
        self.pipe.text_encoder_2 = torch.compile(self.pipe.text_encoder_2)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle_input is not None:
            self.handle_input.remove()
        if self.handle_output is not None:
            self.handle_output.remove()
        self.handle_input = None
        self.handle_output = None
        self.input_activation = None
        self.output_activation = None
        self.timestamps = []

    def __enter__(self):
        module = self.pipe.transformer.get_submodule(self.loc)
        
        if self.use_residual:
            # Capture input for residual computation (pre-hook: called before forward)
            # For entire blocks, inputs may come as kwargs, but we'll try to capture from args first
            def _capture_input(m, args, kwargs=None):
                # Try to capture from kwargs if available (entire blocks often use kwargs)
                if kwargs and isinstance(kwargs, dict):
                    encoder_input = kwargs.get("encoder_hidden_states")
                    hidden_input = kwargs.get("hidden_states")
                    if encoder_input is not None and hidden_input is not None:
                        # Store as tuple (text_input, image_input) to match output format
                        self.input_activation = (encoder_input.clone().detach(), hidden_input.clone().detach())
                        return None
                
                # Fallback: try to capture from positional args
                if args and len(args) > 0:
                    first_arg = args[0]
                    if isinstance(first_arg, torch.Tensor):
                        self.input_activation = first_arg.clone().detach()
                    elif isinstance(first_arg, (list, tuple)) and len(first_arg) >= 2:
                        # Entire block might receive inputs as tuple
                        self.input_activation = (first_arg[0].clone().detach(), first_arg[1].clone().detach())
                    elif isinstance(first_arg, (list, tuple)) and len(first_arg) > 0:
                        # Single element tuple/list
                        self.input_activation = first_arg[0].clone().detach() if isinstance(first_arg[0], torch.Tensor) else None
                    elif len(args) >= 2:
                        # Multiple positional args
                        self.input_activation = (args[0].clone().detach() if isinstance(args[0], torch.Tensor) else None,
                                                args[1].clone().detach() if isinstance(args[1], torch.Tensor) else None)
                    else:
                        self.input_activation = None
                else:
                    self.input_activation = None
                return None  # Pre-hook should not modify args
            
            # Capture output (post-hook: called after forward)
            def _capture_output(m, args, output):
                self.output_activation = output
                return None  # Post-hook can return None or modified output
            
            # Try to register pre-hook with kwargs support (PyTorch 1.9+)
            try:
                self.handle_input = module.register_forward_pre_hook(_capture_input, with_kwargs=True)
            except (TypeError, ValueError):
                # Fallback: register without kwargs support (older PyTorch or module doesn't support it)
                def _capture_input_no_kwargs(m, args):
                    # Simplified version without kwargs
                    if args and len(args) > 0:
                        first_arg = args[0]
                        if isinstance(first_arg, torch.Tensor):
                            self.input_activation = first_arg.clone().detach()
                        elif isinstance(first_arg, (list, tuple)) and len(first_arg) >= 2:
                            self.input_activation = (first_arg[0].clone().detach(), first_arg[1].clone().detach())
                        elif isinstance(first_arg, (list, tuple)) and len(first_arg) > 0:
                            self.input_activation = first_arg[0].clone().detach() if isinstance(first_arg[0], torch.Tensor) else None
                        else:
                            self.input_activation = None
                    else:
                        self.input_activation = None
                    return None
                self.handle_input = module.register_forward_pre_hook(_capture_input_no_kwargs)
            
            self.handle_output = module.register_forward_hook(_capture_output)
        else:
            # Original behavior: only capture output
            def _set(m, args, output):
                self.output_activation = output
                return None
            self.handle_output = module.register_forward_hook(_set)

        return self
    
    def __call__(self, *args, **kwargs):
        def callback(pipe:FluxPipeline, step: int, timestep: int, callback_kwargs: dict):
            self.timestamps.append((step, timestep))
            return callback_kwargs
        
        # Reset activations for this forward pass
        self.input_activation = None
        self.output_activation = None
        
        output = self.pipe(*args, **kwargs, callback_on_step_end=callback)

        # Determine what to return as activations
        if self.use_residual and self.input_activation is not None and self.output_activation is not None:
            # Compute residual: output - input
            # Handle tuple outputs (attention) and single tensor outputs
            if isinstance(self.output_activation, tuple):
                if isinstance(self.input_activation, tuple):
                    activations = tuple(out - inp for out, inp in zip(self.output_activation, self.input_activation))
                else:
                    # If input is single tensor but output is tuple, use output only
                    activations = self.output_activation
            else:
                if isinstance(self.input_activation, tuple):
                    # If input is tuple but output is single, use output only
                    activations = self.output_activation
                else:
                    # Both are single tensors
                    activations = self.output_activation - self.input_activation
        else:
            activations = self.output_activation

        return { "activations": activations, "outputs": output }
    
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


def sample_activations(x: torch.Tensor, config: dict) -> torch.Tensor:
    """
    Sample activations based on the configured strategy.
    
    Args:
        x: Activation tensor of shape (batch, ..., features)
        config: Configuration dictionary containing sampling parameters
        
    Returns:
        Sampled activations tensor of shape (n_samples, features)
    """
    sample_percentage = config.get("sample_percentage", None)
    
    if sample_percentage is not None and sample_percentage > 0:
        # Percentage-based sampling (groundtruth style): sample X% per prompt
        # Need to sample before flattening to maintain per-prompt structure
        original_shape = x.shape  # (batch, ..., features)
        batch_size = original_shape[0]
        
        # Calculate spatial/token dimensions (everything except batch and feature dim)
        spatial_dims = original_shape[1:-1]  # All dims except batch and features
        spatial_size = int(torch.prod(torch.tensor(spatial_dims)).item())
        
        # Calculate samples per prompt
        samples_per_prompt = max(1, int(torch.ceil(torch.tensor(spatial_size * sample_percentage)).item()))
        
        # Sample per prompt
        sampled_activations = []
        for b in range(batch_size):
            prompt_activations = x[b]  # Shape: (..., features)
            prompt_flat = rearrange(prompt_activations, "... d -> (...) d")
            
            # Random sample from this prompt
            n_available = prompt_flat.shape[0]
            n_sample = min(samples_per_prompt, n_available)
            indices = torch.randperm(n_available, device=x.device)[:n_sample]
            sampled = prompt_flat[indices]
            sampled_activations.append(sampled)
        
        # Concatenate all sampled activations from all prompts
        return torch.cat(sampled_activations, dim=0)
    else:
        # Fixed size sampling (current default behavior)
        x = rearrange(x, "b ... d -> (b ...) d")
        bdots, _ = x.shape
        shuffle = torch.randperm(bdots, device=x.device)
        return x[shuffle][:config.get("nsamples", 256)]  # sample a subset of the activations


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
    use_residual = config.get("use_residual", False)
    sampler = FluxActivationSampler("black-forest-labs/FLUX.1-schnell", loc=config["loc"], use_residual=use_residual)

    sampler.pipe.transformer, sampler.pipe.vae, sampler.pipe.text_encoder, sampler.pipe.text_encoder_2 = accelerator.prepare(sampler.pipe.transformer, sampler.pipe.vae, sampler.pipe.text_encoder, sampler.pipe.text_encoder_2)
    dataloader = accelerator.prepare(dataloader)

    # Auto-detect activation dimension and collect samples for initialization
    print("Auto-detecting activation dimension and collecting samples for initialization...")
    dataloader_iter = iter(dataloader)
    
    # Collect samples for pre-bias initialization and MSE scale computation (like groundtruth)
    num_stat_batches = config.get("num_stat_batches", 10)
    stats_samples = []
    
    with torch.no_grad():
        actual_features = None
        for batch_idx in range(num_stat_batches):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                # If we run out of batches, re-create iterator
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            
            with sampler as s:
                test_outputs = s(batch, height=256, width=256, guidance_scale=0., max_sequence_length=256, num_inference_steps=1,)
                test_outputs = test_outputs["activations"]
                # Handle both tuple outputs (attention or entire blocks) and single tensor outputs (MLP, single blocks)
                if isinstance(test_outputs, tuple):
                    # Detect if this is an entire transformer block or an attention module
                    is_entire_block = "transformer_blocks" in config["loc"] and \
                                     ".attn" not in config["loc"] and \
                                     ".ff" not in config["loc"]
                    
                    if is_entire_block:
                        # Entire block returns (text_stream, image_stream) = (encoder_hidden_states, hidden_states)
                        # stream=0 (image) should get output[1] (hidden_states)
                        # stream=1 (text) should get output[0] (encoder_hidden_states)
                        if config["stream"] == 0:  # Image stream
                            _, test_x = test_outputs  # Get second element (hidden_states)
                        elif config["stream"] == 1:  # Text stream
                            test_x, _ = test_outputs  # Get first element (encoder_hidden_states)
                        else:
                            raise ValueError(f"Invalid stream {config['stream']} for entire block. Use 0 (image) or 1 (text).")
                    else:
                        # Attention module returns (query, key)
                        # stream=0 (query) should get output[0]
                        # stream=1 (key) should get output[1]
                        if config["stream"] == 0:  # Query
                            test_x, _ = test_outputs
                        elif config["stream"] == 1:  # Key
                            _, test_x = test_outputs
                        else:
                            raise ValueError(f"Invalid stream {config['stream']} for attention module. Use 0 (query) or 1 (key).")
                else:
                    # Single tensor output (MLP/FF, single_transformer_blocks, etc.)
                    test_x = test_outputs
            
            # Store first batch dimension info (before sampling) - applies to both tuple and non-tuple outputs
            if actual_features is None:
                # Need to check feature dimension - use first element of batch
                temp_shape = test_x.shape
                actual_features = temp_shape[-1]
            
            # Apply same sampling strategy as training for consistency
            test_x_sampled = sample_activations(test_x, config)
            
            # Collect samples (move to CPU to save GPU memory)
            stats_samples.append(test_x_sampled.cpu())
        
        # Concatenate all samples
        if stats_samples:
            stats_acts_sample = torch.cat(stats_samples, dim=0)
            print(f"Collected {len(stats_acts_sample)} samples from {num_stat_batches} batches for initialization")
        else:
            stats_acts_sample = None
            print("WARNING: No samples collected for initialization")
    
    # Validate or auto-correct feature dimension
    if config["features"] != actual_features:
        print(f"WARNING: Configured features ({config['features']}) doesn't match actual activation dimension ({actual_features})!")
        print(f"Auto-correcting to use activation dimension: {actual_features}")
        config["features"] = actual_features
    
    # Create SAE with correct (or auto-corrected) feature dimension
    pages = int(round(config["expansion"] * config["features"]))
    match config["arch"]:
        case "standard":
            sae = SparseAutoencoder(features=config["features"], pages=pages)
        case "topk":
            auxk = config.get("auxk", None)
            # Reference implementation divides by batch_size: dead_toks_threshold // cfg.bs
            # dead_steps_threshold is in "tokens/samples", but stats_last_nonzero increments per batch
            dead_toks_threshold = config.get("dead_steps_threshold", 10000000)
            dead_steps_threshold = dead_toks_threshold // config["batch_size"]
            sae = TopkSparseAutoencoder(
                features=config["features"], 
                pages=pages, 
                k=config["k"],
                auxk=auxk,
                dead_steps_threshold=dead_steps_threshold
            )
        case "gated":
            sae = GatedAutoEncoder(features=config["features"], pages=pages)
        case _:
            raise ValueError(f"Unknown architecture: {config['arch']}")
    
    # Initialize pre_bias from geometric median (groundtruth approach)
    mse_scale = config.get("mse_scale", 1.0)
    if stats_acts_sample is not None:
        # Initialize pre_bias
        if hasattr(sae, 'pre_bias'):
            print("Initializing pre_bias from geometric median of sample activations...")
            init_pre_bias_from_data(sae, stats_acts_sample)
            print("Pre-bias initialization complete")
        
        # Compute MSE scale from data statistics (groundtruth approach)
        print("Computing MSE scale from data statistics...")
        stats_acts_float = stats_acts_sample.float()
        mse_scale = (1 / ((stats_acts_float.mean(dim=0) - stats_acts_float) ** 2).mean()).item()
        print(f"Computed MSE scale: {mse_scale:.6f}")
        
        # Free memory
        del stats_acts_sample
        del stats_acts_float
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        print(f"Using configured MSE scale: {mse_scale}")
    
    optimizer = torch.optim.AdamW(sae.parameters(), lr=config["lr"], betas=(config["beta1"], config["beta2"]), weight_decay=config["wd"])
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=config["lr_warmup_steps"])
    sae, optimizer, scheduler = accelerator.prepare(sae, optimizer, scheduler)
    
    print(f"Using activation dimension: {config['features']}, SAE pages: {pages}")

    steps = 0

    match config["arch"]:
        case "standard":
            trainer = StandardTrainer(sae, optimizer, scheduler, lmbda=config["lmbda"], lmbda_warmup_steps=config["lmbda_warmup_steps"], accelerator=accelerator)
        case "topk":
            auxk_coef = config.get("auxk_coef", 1.0)
            log_interval = config.get("log_interval", 100)
            # Use computed MSE scale from data statistics
            trainer = TopkTrainer(
                sae, optimizer, scheduler, 
                pages=pages, 
                auxk=config.get("auxk", None), 
                bodycount=config.get("bodycount", 0),
                normalise=config.get("normalise", True),
                accelerator=accelerator,
                mse_scale=mse_scale,
                auxk_coef=auxk_coef,
                log_interval=log_interval
            )
        case "gated":
            trainer = GatedTrainer(sae, optimizer, scheduler, lmbda=config["lmbda"], lmbda_warmup_steps=config["lmbda_warmup_steps"], accelerator=accelerator)
        
    sae.train()
    # Continue with the dataloader (some batches were used for initialization)
    # Since dataloader shuffles, we'll process all batches from the current iterator position
    remaining_batches = len(dataloader) - num_stat_batches
    steps = 0
    
    # Check sampling strategy and print info once
    sample_percentage = config.get("sample_percentage", None)
    if sample_percentage is not None and sample_percentage > 0:
        print(f"Using percentage-based sampling: {sample_percentage*100:.1f}% per prompt (groundtruth style)")
    else:
        print(f"Using fixed-size sampling: {config.get('nsamples', 256)} activations per batch")
    
    for prompts in tqdm(dataloader_iter, total=max(0, remaining_batches), desc="Training"):
        with sampler as s:
            outputs = s(prompts, height=256, width=256, guidance_scale=0., max_sequence_length=256, num_inference_steps=1,)
            outputs = outputs["activations"]
            # Handle both tuple outputs (attention or entire blocks) and single tensor outputs (MLP, single blocks)
            if isinstance(outputs, tuple):
                # Detect if this is an entire transformer block or an attention module
                is_entire_block = "transformer_blocks" in config["loc"] and \
                                 ".attn" not in config["loc"] and \
                                 ".ff" not in config["loc"]
                
                if is_entire_block:
                    # Entire block returns (text_stream, image_stream) = (encoder_hidden_states, hidden_states)
                    # stream=0 (image) should get output[1] (hidden_states)
                    # stream=1 (text) should get output[0] (encoder_hidden_states)
                    if config["stream"] == 0:  # Image stream
                        _, x = outputs  # Get second element (hidden_states)
                    elif config["stream"] == 1:  # Text stream
                        x, _ = outputs  # Get first element (encoder_hidden_states)
                    else:
                        raise ValueError(f"Invalid stream {config['stream']} for entire block. Use 0 (image) or 1 (text).")
                else:
                    # Attention module returns (query, key)
                    # stream=0 (query) should get output[0]
                    # stream=1 (key) should get output[1]
                    if config["stream"] == 0:  # Query
                        x, _ = outputs
                    elif config["stream"] == 1:  # Key
                        _, x = outputs
                    else:
                        raise ValueError(f"Invalid stream {config['stream']} for attention module. Use 0 (query) or 1 (key).")
            else:
                # Single tensor output (MLP/FF, single_transformer_blocks, etc.)
                x = outputs

            # Sample activations based on strategy (same as initialization)
            x = sample_activations(x, config)
            
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
@click.option("--expansion", type=float, default=4)
@click.option("--lr", type=float, default=5e-5)
@click.option("--beta1", type=float, default=0.9)
@click.option("--beta2", type=float, default=0.999)
@click.option("--wd", type=float, default=0.)
@click.option("--lr_warmup_steps", type=int, default=256)
@click.option("--k", type=int, default=20, help="K value for TopK SAE (default: 20, matches groundtruth)")
@click.option("--auxk", type=float, default=1/32)
@click.option("--auxk_coef", type=float, default=1/32, help="Coefficient for auxk loss (default: 1/32)")
@click.option("--dead_steps_threshold", type=int, default=10000000, help="Threshold for dead feature detection (default: 10M)")
@click.option("--bodycount", type=int, default=16384)
@click.option("--savedir", type=str, default="./checkpoints")
@click.option("--lmbda", type=float, default=0.01)
@click.option("--lmbda_warmup_steps", type=int, default=256)
@click.option("--loc", type=str, default="transformer_blocks.18.ff", help="Layer location (default: layer 18, matches groundtruth)")
@click.option("--stream", type=int, default=0, help="Stream index: 0=image stream, 1=text stream (default: 0, matches groundtruth)")
@click.option("--iters", type=int, default=4096)
@click.option("--nsamples", type=int, default=256, help="Fixed number of activations to sample per batch (used if --sample-percentage is not set)")
@click.option("--sample-percentage", type=float, default=None, help="Percentage of activations to sample per prompt (e.g., 0.1 for 10%%). If set, overrides --nsamples")
@click.option("--normalise", type=bool, default=False)
@click.option("--use-residual/--no-use-residual", default=False, help="Use residual activations (output - input) instead of direct output")
@click.option("--num-stat-batches", type=int, default=10, help="Number of batches to sample for initialization statistics (default: 10)")
@click.option("--mse-scale", type=float, default=None, help="MSE scale (if None, will be computed from data)")
def main(**config):
    train(**config)


if __name__ == "__main__":
    main()
    