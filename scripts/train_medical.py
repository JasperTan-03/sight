#!/usr/bin/env python3
"""
Training Script for Medical Text-to-3D Fine-tuning

This script implements the "Surgery & Stitch" fine-tuning strategy:
1. Load pre-trained Shap-E diffusion model
2. Replace CLIP conditioning with BioMedCLIP
3. Freeze most parameters, only train:
   - New medical_projection layer (BioMedCLIP 512 -> transformer width)
   - Cross-attention layers in the transformer

Usage:
    python scripts/train_medical.py \
        --data_dir /path/to/cached_data \
        --output_dir /path/to/checkpoints \
        --batch_size 4 \
        --learning_rate 1e-4 \
        --num_epochs 100

The data_dir should contain pre-cached .pt files from prepare_data.py:
    - sample_001_latent.pt
    - sample_001_text.pt
    - ...
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shap_e.diffusion.gaussian_diffusion import diffusion_from_config, GaussianDiffusion
from shap_e.models.download import load_model, load_config, load_checkpoint
from shap_e.models.configs import model_from_config
from shap_e.models.generation.latent_diffusion import SplitVectorDiffusion
from shap_e.models.generation.transformer import BioMedCLIPTextDiffusionTransformer


class MedicalLatentDataset(Dataset):
    """
    Dataset for loading pre-cached latent and text embedding pairs.
    """
    
    def __init__(
        self,
        data_dir: str,
        d_latent: int = 1048576,  # Shap-E default latent dimension
    ):
        """
        Args:
            data_dir: Directory containing cached .pt files
            d_latent: Expected latent dimension
        """
        self.data_dir = Path(data_dir)
        self.d_latent = d_latent
        
        # Find all latent files
        self.samples = []
        for latent_file in self.data_dir.glob("*_latent.pt"):
            sample_name = latent_file.stem.replace("_latent", "")
            text_file = self.data_dir / f"{sample_name}_text.pt"
            
            if text_file.exists():
                self.samples.append({
                    "name": sample_name,
                    "latent_path": str(latent_file),
                    "text_path": str(text_file),
                })
        
        print(f"Found {len(self.samples)} samples in {data_dir}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load tensors
        latent = torch.load(sample["latent_path"], map_location="cpu")
        text_embedding = torch.load(sample["text_path"], map_location="cpu")
        
        return {
            "latent": latent.float(),
            "text_embedding": text_embedding.float(),
            "name": sample["name"],
        }


def create_medical_diffusion_model(
    device: torch.device,
    pretrained_config: Optional[dict] = None,
    pretrained_weights: Optional[dict] = None,
    context_dim: int = 512,
    use_pooled_embeddings: bool = True,
) -> Tuple[nn.Module, int]:
    """
    Create a BioMedCLIP-conditioned diffusion model.
    
    This function:
    1. Creates a new BioMedCLIPTextDiffusionTransformer
    2. Optionally loads pre-trained Shap-E weights (excluding CLIP components)
    3. Wraps it in SplitVectorDiffusion
    
    Args:
        device: Device to create model on
        pretrained_config: Configuration from Shap-E text300M
        pretrained_weights: Pre-trained weights from Shap-E text300M
        context_dim: BioMedCLIP embedding dimension (512)
        use_pooled_embeddings: Whether to use pooled embeddings
    
    Returns:
        Tuple of (model, d_latent)
    """
    # Default configuration based on Shap-E text300M
    if pretrained_config is None:
        pretrained_config = load_config('text300M')
    
    # Extract inner model config (the transformer)
    inner_config = pretrained_config.get("inner", {}).copy()
    d_latent = pretrained_config.get("d_latent", 1048576)
    latent_ctx = pretrained_config.get("latent_ctx", 1024)
    
    # Get transformer parameters from pretrained config
    width = inner_config.get("width", 1024)
    layers = inner_config.get("layers", 24)
    heads = inner_config.get("heads", 16)
    init_scale = inner_config.get("init_scale", 0.25)
    time_token_cond = inner_config.get("time_token_cond", True)
    
    # Create BioMedCLIP-conditioned transformer
    inner_model = BioMedCLIPTextDiffusionTransformer(
        device=device,
        dtype=torch.float32,
        n_ctx=latent_ctx,
        width=width,
        layers=layers,
        heads=heads,
        init_scale=init_scale,
        time_token_cond=time_token_cond,
        input_channels=d_latent // latent_ctx,
        output_channels=d_latent // latent_ctx * 2,  # For variance prediction
        token_cond=True,
        cond_drop_prob=0.1,  # For classifier-free guidance
        context_dim=context_dim,
        use_pooled_embeddings=use_pooled_embeddings,
    )
    
    # Load pre-trained weights (excluding CLIP-specific components)
    if pretrained_weights is not None:
        print("Loading pre-trained Shap-E weights...")
        
        # Filter out CLIP-related weights
        filtered_weights = {}
        skipped_keys = []
        
        for key, value in pretrained_weights.items():
            # Skip CLIP-related keys
            if any(clip_key in key.lower() for clip_key in ['clip', 'clip_embed']):
                skipped_keys.append(key)
                continue
            
            # Map wrapped model keys
            if key.startswith('wrapped.'):
                new_key = key.replace('wrapped.', '')
                filtered_weights[new_key] = value
            else:
                filtered_weights[new_key] = value
        
        print(f"Skipped {len(skipped_keys)} CLIP-related keys")
        
        # Load compatible weights
        missing, unexpected = inner_model.load_state_dict(filtered_weights, strict=False)
        
        print(f"Missing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")
        
        if len(missing) > 0:
            print("Missing keys (first 10):", missing[:10])
        if len(unexpected) > 0:
            print("Unexpected keys (first 10):", unexpected[:10])
    
    # Wrap in SplitVectorDiffusion
    model = SplitVectorDiffusion(
        device=device,
        wrapped=inner_model,
        n_ctx=latent_ctx,
        d_latent=d_latent,
    )
    
    return model, d_latent


def setup_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    freeze_strategy: str = "surgery_stitch",
) -> torch.optim.Optimizer:
    """
    Setup optimizer with selective parameter freezing.
    
    Args:
        model: The diffusion model
        learning_rate: Learning rate
        weight_decay: Weight decay
        freeze_strategy: One of:
            - "surgery_stitch": Only train projection + attention layers
            - "projection_only": Only train the projection layer
            - "full": Train all parameters
    
    Returns:
        Configured optimizer
    """
    inner_model = model.wrapped
    
    if freeze_strategy == "surgery_stitch":
        # Freeze everything
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze medical projection
        for param in inner_model.medical_projection.parameters():
            param.requires_grad = True
        
        # Unfreeze attention layers
        for block in inner_model.backbone.resblocks:
            for param in block.attn.parameters():
                param.requires_grad = True
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
    elif freeze_strategy == "projection_only":
        # Freeze everything
        for param in model.parameters():
            param.requires_grad = False
        
        # Only unfreeze medical projection
        for param in inner_model.medical_projection.parameters():
            param.requires_grad = True
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
    elif freeze_strategy == "full":
        # Train everything
        trainable_params = list(model.parameters())
    
    else:
        raise ValueError(f"Unknown freeze strategy: {freeze_strategy}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Trainable ratio: {100 * trainable_count / total_params:.2f}%")
    
    optimizer = AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )
    
    return optimizer


def compute_loss(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    latents: torch.Tensor,
    text_embeddings: torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Compute diffusion training loss.
    
    Args:
        model: The diffusion model
        diffusion: Gaussian diffusion instance
        latents: Ground truth latents [B, d_latent]
        text_embeddings: BioMedCLIP embeddings [B, 512] or [B, seq_len, 512]
        device: Device
    
    Returns:
        Dictionary with loss terms
    """
    batch_size = latents.shape[0]
    
    # Sample random timesteps
    t = torch.randint(
        0, diffusion.num_timesteps, (batch_size,), device=device
    ).long()
    
    # Compute diffusion loss
    model_kwargs = {"embeddings": text_embeddings}
    
    losses = diffusion.training_losses(
        model=model,
        x_start=latents,
        t=t,
        model_kwargs=model_kwargs,
    )
    
    return losses


def train_epoch(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    gradient_clip: float = 1.0,
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Returns:
        Dictionary with average losses
    """
    model.train()
    
    total_loss = 0.0
    total_mse = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        latents = batch["latent"].to(device)
        text_embeddings = batch["text_embedding"].to(device)
        
        optimizer.zero_grad()
        
        losses = compute_loss(
            model=model,
            diffusion=diffusion,
            latents=latents,
            text_embeddings=text_embeddings,
            device=device,
        )
        
        loss = losses["loss"].mean()
        loss.backward()
        
        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                gradient_clip
            )
        
        optimizer.step()
        
        total_loss += loss.item()
        if "mse" in losses:
            total_mse += losses["mse"].mean().item()
        num_batches += 1
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg_loss": f"{total_loss / num_batches:.4f}",
        })
    
    return {
        "loss": total_loss / num_batches,
        "mse": total_mse / num_batches if total_mse > 0 else 0.0,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    losses: Dict[str, float],
    output_dir: str,
    is_best: bool = False,
):
    """Save a training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": losses,
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch:04d}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    # Save latest
    latest_path = os.path.join(output_dir, "checkpoint_latest.pt")
    torch.save(checkpoint, latest_path)
    
    # Save best if applicable
    if is_best:
        best_path = os.path.join(output_dir, "checkpoint_best.pt")
        torch.save(checkpoint, best_path)
    
    print(f"Saved checkpoint to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Shap-E with BioMedCLIP for medical text-to-3D"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing cached latent and text embedding .pt files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save checkpoints and logs",
    )
    
    # Model arguments
    parser.add_argument(
        "--freeze_strategy",
        type=str,
        default="surgery_stitch",
        choices=["surgery_stitch", "projection_only", "full"],
        help="Which parameters to train",
    )
    parser.add_argument(
        "--use_pretrained",
        action="store_true",
        default=True,
        help="Load pre-trained Shap-E weights",
    )
    parser.add_argument(
        "--context_dim",
        type=int,
        default=512,
        help="BioMedCLIP embedding dimension",
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--gradient_clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm (0 to disable)",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )
    
    # Hardware arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Use mixed precision training",
    )
    
    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print("=" * 60)
    print("Medical Text-to-3D Fine-tuning")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Freeze strategy: {args.freeze_strategy}")
    print("=" * 60)
    
    # Create dataset and dataloader
    dataset = MedicalLatentDataset(args.data_dir)
    
    if len(dataset) == 0:
        print("ERROR: No samples found in data directory!")
        print("Make sure to run prepare_data.py first.")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Load diffusion config
    diffusion_config = load_config('diffusion')
    diffusion = diffusion_from_config(diffusion_config)
    
    # Create model
    pretrained_config = None
    pretrained_weights = None
    
    if args.use_pretrained:
        print("Loading pre-trained Shap-E configuration and weights...")
        pretrained_config = load_config('text300M')
        pretrained_weights = load_checkpoint('text300M', device=device)
    
    model, d_latent = create_medical_diffusion_model(
        device=device,
        pretrained_config=pretrained_config,
        pretrained_weights=pretrained_weights,
        context_dim=args.context_dim,
        use_pooled_embeddings=True,
    )
    
    model = model.to(device)
    
    # Setup optimizer
    optimizer = setup_optimizer(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        freeze_strategy=args.freeze_strategy,
    )
    
    # Setup scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.learning_rate * 0.01,
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        if "losses" in checkpoint:
            best_loss = checkpoint["losses"].get("loss", float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    
    # Setup mixed precision
    scaler = torch.cuda.amp.GradScaler() if args.use_fp16 and device.type == 'cuda' else None
    
    # Training loop
    print("\nStarting training...")
    training_log = []
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.2e}")
        print(f"{'=' * 60}")
        
        # Train one epoch
        epoch_losses = train_epoch(
            model=model,
            diffusion=diffusion,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch + 1,
            gradient_clip=args.gradient_clip,
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log
        log_entry = {
            "epoch": epoch + 1,
            "loss": epoch_losses["loss"],
            "mse": epoch_losses["mse"],
            "lr": scheduler.get_last_lr()[0],
            "timestamp": datetime.now().isoformat(),
        }
        training_log.append(log_entry)
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Loss: {epoch_losses['loss']:.6f}")
        print(f"  MSE: {epoch_losses['mse']:.6f}")
        
        # Check if best
        is_best = epoch_losses["loss"] < best_loss
        if is_best:
            best_loss = epoch_losses["loss"]
            print(f"  New best loss!")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                losses=epoch_losses,
                output_dir=args.output_dir,
                is_best=is_best,
            )
        elif is_best:
            # Always save best model
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                losses=epoch_losses,
                output_dir=args.output_dir,
                is_best=True,
            )
        
        # Save training log
        log_path = os.path.join(args.output_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
