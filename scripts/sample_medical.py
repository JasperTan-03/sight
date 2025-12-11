#!/usr/bin/env python3
"""
Sampling Script for Medical Text-to-3D Generation

This script generates 3D models from medical text descriptions using
a fine-tuned Shap-E model with BioMedCLIP conditioning.

Usage:
    python scripts/sample_medical.py \
        --checkpoint /path/to/checkpoint_best.pt \
        --prompt "A liver with a tumor in the right lobe" \
        --output_dir /path/to/outputs \
        --num_samples 4
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, decode_latent_mesh
from shap_e.models.generation.latent_diffusion import SplitVectorDiffusion
from shap_e.models.generation.transformer import BioMedCLIPTextDiffusionTransformer


def load_biomedclip(device: torch.device):
    """Load BioMedCLIP model for text encoding."""
    try:
        from open_clip import create_model_and_transforms, get_tokenizer
    except ImportError:
        raise ImportError(
            "open_clip is required for BioMedCLIP. "
            "Install with: pip install open_clip_torch"
        )
    
    print("Loading BioMedCLIP model...")
    model, _, _ = create_model_and_transforms(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    tokenizer = get_tokenizer(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    
    model = model.to(device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model, tokenizer


def encode_text(model, tokenizer, text: str, device: torch.device) -> torch.Tensor:
    """Encode text using BioMedCLIP."""
    with torch.no_grad():
        tokens = tokenizer([text], context_length=256).to(device)
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features


def create_medical_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> nn.Module:
    """
    Create and load a medical diffusion model from checkpoint.
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint or use defaults
    config = load_config('text300M')
    inner_config = config.get("inner", {}).copy()
    d_latent = config.get("d_latent", 1048576)
    latent_ctx = config.get("latent_ctx", 1024)
    
    width = inner_config.get("width", 1024)
    layers = inner_config.get("layers", 24)
    heads = inner_config.get("heads", 16)
    init_scale = inner_config.get("init_scale", 0.25)
    time_token_cond = inner_config.get("time_token_cond", True)
    
    # Create model
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
        output_channels=d_latent // latent_ctx * 2,
        token_cond=True,
        cond_drop_prob=0.0,  # No dropout during inference
        context_dim=512,
        use_pooled_embeddings=True,
    )
    
    model = SplitVectorDiffusion(
        device=device,
        wrapped=inner_model,
        n_ctx=latent_ctx,
        d_latent=d_latent,
    )
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model


def sample_medical_latents(
    model: nn.Module,
    diffusion,
    text_embeddings: torch.Tensor,
    batch_size: int,
    device: torch.device,
    guidance_scale: float = 15.0,
    use_karras: bool = True,
    karras_steps: int = 64,
    progress: bool = True,
) -> torch.Tensor:
    """
    Sample latents from the medical diffusion model.
    
    This is a modified version of sample_latents that works with
    our BioMedCLIP-conditioned model.
    """
    from shap_e.diffusion.k_diffusion import karras_sample
    
    sample_shape = (batch_size, model.d_latent)
    
    # Prepare model kwargs
    if guidance_scale != 1.0:
        # Double embeddings for classifier-free guidance
        model_kwargs = {
            "embeddings": torch.cat([
                text_embeddings.expand(batch_size, -1),
                torch.zeros_like(text_embeddings).expand(batch_size, -1),
            ], dim=0)
        }
    else:
        model_kwargs = {
            "embeddings": text_embeddings.expand(batch_size, -1)
        }
    
    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=True):
            if use_karras:
                samples = karras_sample(
                    diffusion=diffusion,
                    model=model,
                    shape=sample_shape,
                    steps=karras_steps,
                    clip_denoised=True,
                    model_kwargs=model_kwargs,
                    device=device,
                    sigma_min=1e-3,
                    sigma_max=160,
                    s_churn=0,
                    guidance_scale=guidance_scale,
                    progress=progress,
                )
            else:
                # Use DDPM sampling
                if guidance_scale != 1.0:
                    internal_batch_size = batch_size * 2
                else:
                    internal_batch_size = batch_size
                
                samples = diffusion.p_sample_loop(
                    model,
                    shape=(internal_batch_size, model.d_latent),
                    model_kwargs=model_kwargs,
                    device=device,
                    clip_denoised=True,
                    progress=progress,
                )
                
                if guidance_scale != 1.0:
                    samples = samples[:batch_size]
    
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D models from medical text descriptions"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Medical text description",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./medical_outputs",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=15.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--karras_steps",
        type=int,
        default=64,
        help="Number of Karras sampling steps",
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default="stf",
        choices=["stf", "nerf"],
        help="Rendering mode for visualization",
    )
    parser.add_argument(
        "--render_size",
        type=int,
        default=64,
        help="Size of rendered images",
    )
    parser.add_argument(
        "--save_mesh",
        action="store_true",
        default=True,
        help="Save meshes as PLY and OBJ files",
    )
    parser.add_argument(
        "--save_gif",
        action="store_true",
        default=True,
        help="Save rotating GIF renders",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Medical Text-to-3D Generation")
    print("=" * 60)
    print(f"Prompt: {args.prompt}")
    print(f"Num samples: {args.num_samples}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load models
    print("\nLoading models...")
    
    # Load transmitter for decoding
    xm = load_model('transmitter', device=device)
    xm.eval()
    
    # Load fine-tuned diffusion model
    model = create_medical_model_from_checkpoint(args.checkpoint, device)
    
    # Load diffusion
    diffusion = diffusion_from_config(load_config('diffusion'))
    
    # Load BioMedCLIP for text encoding
    biomedclip, tokenizer = load_biomedclip(device)
    
    # Encode text
    print(f"\nEncoding prompt: {args.prompt}")
    text_embedding = encode_text(biomedclip, tokenizer, args.prompt, device)
    print(f"Text embedding shape: {text_embedding.shape}")
    
    # Generate samples
    print(f"\nGenerating {args.num_samples} samples...")
    latents = sample_medical_latents(
        model=model,
        diffusion=diffusion,
        text_embeddings=text_embedding,
        batch_size=args.num_samples,
        device=device,
        guidance_scale=args.guidance_scale,
        karras_steps=args.karras_steps,
        progress=True,
    )
    
    print(f"Generated latents shape: {latents.shape}")
    
    # Decode and save outputs
    print("\nDecoding and saving outputs...")
    
    cameras = create_pan_cameras(args.render_size, device)
    
    for i, latent in enumerate(tqdm(latents, desc="Processing samples")):
        sample_name = f"sample_{i:03d}"
        
        # Save mesh
        if args.save_mesh:
            try:
                mesh = decode_latent_mesh(xm, latent)
                tri_mesh = mesh.tri_mesh()
                
                ply_path = os.path.join(args.output_dir, f"{sample_name}.ply")
                obj_path = os.path.join(args.output_dir, f"{sample_name}.obj")
                
                with open(ply_path, "wb") as f:
                    tri_mesh.write_ply(f)
                
                with open(obj_path, "w") as f:
                    tri_mesh.write_obj(f)
                
                print(f"  Saved mesh: {ply_path}")
            except Exception as e:
                print(f"  Warning: Failed to save mesh for sample {i}: {e}")
        
        # Save GIF
        if args.save_gif:
            try:
                images = decode_latent_images(
                    xm, latent, cameras, rendering_mode=args.render_mode
                )
                
                gif_path = os.path.join(args.output_dir, f"{sample_name}.gif")
                images[0].save(
                    gif_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=100,
                    loop=0,
                )
                
                print(f"  Saved GIF: {gif_path}")
            except Exception as e:
                print(f"  Warning: Failed to save GIF for sample {i}: {e}")
    
    # Save latents
    latents_path = os.path.join(args.output_dir, "latents.pt")
    torch.save(latents.cpu(), latents_path)
    print(f"\nSaved latents to {latents_path}")
    
    # Save metadata
    import json
    metadata = {
        "prompt": args.prompt,
        "num_samples": args.num_samples,
        "guidance_scale": args.guidance_scale,
        "karras_steps": args.karras_steps,
        "checkpoint": args.checkpoint,
    }
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Generation complete!")
    print(f"Outputs saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
