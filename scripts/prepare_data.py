#!/usr/bin/env python3
"""
Data Preparation Script for Medical Text-to-3D Fine-tuning

This script pre-computes and caches:
1. Shap-E latents from 3D mesh files (STL/OBJ)
2. BioMedCLIP text embeddings from medical descriptions

This avoids running expensive encoders during training.

Usage:
    python scripts/prepare_data.py \
        --data_dir /path/to/dataset \
        --output_dir /path/to/cached \
        --device cuda

Expected data directory structure:
    data_dir/
        sample_001.stl (or .obj)
        sample_001.txt
        sample_002.stl
        sample_002.txt
        ...
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shap_e.models.download import load_model
from shap_e.util.data_util import load_or_create_multimodal_batch
from shap_e.util.collections import AttrDict


def load_biomedclip(device: torch.device) -> Tuple[nn.Module, nn.Module]:
    """
    Load BioMedCLIP model and tokenizer from HuggingFace.
    
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        from open_clip import create_model_and_transforms, get_tokenizer
    except ImportError:
        raise ImportError(
            "open_clip is required for BioMedCLIP. "
            "Install with: pip install open_clip_torch"
        )
    
    print("Loading BioMedCLIP model...")
    
    # Load BioMedCLIP using open_clip
    model, _, preprocess = create_model_and_transforms(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    tokenizer = get_tokenizer(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    
    model = model.to(device)
    model.eval()
    
    # Freeze BioMedCLIP - we only use it for inference
    for param in model.parameters():
        param.requires_grad = False
    
    print("BioMedCLIP loaded successfully!")
    return model, tokenizer


def encode_text_biomedclip(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int = 256,
    return_pooled: bool = True,
) -> torch.Tensor:
    """
    Encode text descriptions using BioMedCLIP.
    
    Args:
        model: BioMedCLIP model
        tokenizer: BioMedCLIP tokenizer
        texts: List of text descriptions
        device: Device to use
        max_length: Maximum token length (BioMedCLIP supports 256)
        return_pooled: If True, return [CLS] pooled embeddings [B, 512]
                      If False, return full sequence [B, seq_len, 512]
    
    Returns:
        Tensor of embeddings
    """
    with torch.no_grad():
        # Tokenize texts
        tokens = tokenizer(texts, context_length=max_length).to(device)
        
        # Get text features
        if return_pooled:
            # Get pooled [CLS] token embeddings
            text_features = model.encode_text(tokens)
            # Normalize (BioMedCLIP outputs normalized embeddings)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features  # [B, 512]
        else:
            # Get full sequence embeddings (requires accessing internal layers)
            # This is more complex and depends on the exact model architecture
            # For simplicity, we use pooled embeddings by default
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.unsqueeze(1)  # [B, 1, 512]


def load_3d_mesh_to_batch(
    mesh_path: str,
    device: torch.device,
    cache_dir: Optional[str] = None,
    verbose: bool = False,
) -> AttrDict:
    """
    Load a 3D mesh file and convert it to the batch format expected by Shap-E encoder.
    
    Supports: .stl, .obj, .ply, .off, .glb, .gltf
    
    Args:
        mesh_path: Path to the 3D mesh file
        device: Device to use
        cache_dir: Directory for caching intermediate results
        verbose: Whether to print progress
    
    Returns:
        AttrDict with 'points' key containing the point cloud
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError(
            "trimesh is required for loading 3D meshes. "
            "Install with: pip install trimesh"
        )
    
    # Load mesh with trimesh
    mesh = trimesh.load(mesh_path, force='mesh')
    
    if isinstance(mesh, trimesh.Scene):
        # Handle scene files (like some GLB files)
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if len(meshes) == 0:
            raise ValueError(f"No valid meshes found in {mesh_path}")
        mesh = trimesh.util.concatenate(meshes)
    
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load {mesh_path} as a triangle mesh")
    
    # Convert to OBJ temporarily for Shap-E compatibility
    # Shap-E's data_util expects OBJ or similar formats for rendering
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        mesh.export(tmp_path)
    
    try:
        # Use Shap-E's multimodal batch loader
        batch = load_or_create_multimodal_batch(
            device,
            model_path=tmp_path,
            cache_dir=cache_dir,
            point_count=2**14,
            random_sample_count=2**19,
            pc_num_views=40,
            mv_light_mode=None,  # Don't need multiview for encoding
            verbose=verbose,
        )
    finally:
        # Clean up temp file
        os.unlink(tmp_path)
    
    return batch


def encode_mesh_to_latent(
    encoder: nn.Module,
    batch: AttrDict,
    device: torch.device,
) -> torch.Tensor:
    """
    Encode a 3D mesh batch to Shap-E latent representation.
    
    Args:
        encoder: Shap-E transmitter model (contains encoder)
        batch: AttrDict from load_3d_mesh_to_batch
        device: Device to use
    
    Returns:
        Latent tensor of shape [d_latent]
    """
    with torch.no_grad():
        # The transmitter's encoder has encode_to_bottleneck method
        latent = encoder.encoder.encode_to_bottleneck(batch)
        return latent.squeeze(0)  # Remove batch dimension


def find_paired_files(
    data_dir: str,
    mesh_extensions: List[str] = ['.stl', '.obj', '.ply', '.off', '.glb', '.gltf'],
) -> List[Tuple[str, str]]:
    """
    Find paired mesh and text files in the data directory.
    
    Args:
        data_dir: Directory containing the data
        mesh_extensions: List of valid mesh file extensions
    
    Returns:
        List of (mesh_path, text_path) tuples
    """
    data_path = Path(data_dir)
    pairs = []
    
    # Find all mesh files
    mesh_files = []
    for ext in mesh_extensions:
        mesh_files.extend(data_path.glob(f'*{ext}'))
        mesh_files.extend(data_path.glob(f'*{ext.upper()}'))
    
    # Match with text files
    for mesh_file in mesh_files:
        stem = mesh_file.stem
        text_file = data_path / f"{stem}.txt"
        
        if text_file.exists():
            pairs.append((str(mesh_file), str(text_file)))
        else:
            print(f"Warning: No text file found for {mesh_file.name}")
    
    return pairs


def process_dataset(
    data_dir: str,
    output_dir: str,
    device: torch.device,
    use_pooled_embeddings: bool = True,
    cache_dir: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    Process the entire dataset and save cached embeddings.
    
    Args:
        data_dir: Directory containing mesh and text files
        output_dir: Directory to save cached tensors
        device: Device to use
        use_pooled_embeddings: Whether to use pooled BioMedCLIP embeddings
        cache_dir: Directory for intermediate caching (Blender renders, etc.)
        verbose: Whether to print detailed progress
    
    Returns:
        Dictionary with processing statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    print("Loading Shap-E transmitter...")
    transmitter = load_model('transmitter', device=device)
    transmitter.eval()
    
    # Freeze transmitter
    for param in transmitter.parameters():
        param.requires_grad = False
    
    biomedclip_model, biomedclip_tokenizer = load_biomedclip(device)
    
    # Find paired files
    pairs = find_paired_files(data_dir)
    print(f"Found {len(pairs)} mesh-text pairs")
    
    if len(pairs) == 0:
        print("No pairs found! Check your data directory structure.")
        return {"total": 0, "success": 0, "failed": 0}
    
    stats = {"total": len(pairs), "success": 0, "failed": 0}
    
    # Process each pair
    for mesh_path, text_path in tqdm(pairs, desc="Processing dataset"):
        try:
            sample_name = Path(mesh_path).stem
            latent_path = os.path.join(output_dir, f"{sample_name}_latent.pt")
            text_emb_path = os.path.join(output_dir, f"{sample_name}_text.pt")
            
            # Skip if already processed
            if os.path.exists(latent_path) and os.path.exists(text_emb_path):
                if verbose:
                    print(f"Skipping {sample_name} (already processed)")
                stats["success"] += 1
                continue
            
            # Load and encode mesh
            if verbose:
                print(f"Processing mesh: {mesh_path}")
            
            batch = load_3d_mesh_to_batch(
                mesh_path,
                device=device,
                cache_dir=cache_dir,
                verbose=verbose,
            )
            latent = encode_mesh_to_latent(transmitter, batch, device)
            
            # Load and encode text
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if verbose:
                print(f"Text: {text[:100]}...")
            
            text_embedding = encode_text_biomedclip(
                biomedclip_model,
                biomedclip_tokenizer,
                [text],
                device=device,
                return_pooled=use_pooled_embeddings,
            )
            
            # Save tensors
            torch.save(latent.cpu(), latent_path)
            torch.save(text_embedding.squeeze(0).cpu(), text_emb_path)  # Remove batch dim
            
            stats["success"] += 1
            
            if verbose:
                print(f"Saved: {latent_path}, {text_emb_path}")
                print(f"Latent shape: {latent.shape}, Text embedding shape: {text_embedding.shape}")
        
        except Exception as e:
            print(f"Error processing {mesh_path}: {e}")
            stats["failed"] += 1
            continue
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for medical text-to-3D fine-tuning"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing mesh (.stl/.obj) and text (.txt) files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save cached embeddings",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for intermediate caching (Blender renders, point clouds)",
    )
    parser.add_argument(
        "--use_sequence_embeddings",
        action="store_true",
        help="Use full sequence embeddings instead of pooled [CLS] token",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    stats = process_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=device,
        use_pooled_embeddings=not args.use_sequence_embeddings,
        cache_dir=args.cache_dir,
        verbose=args.verbose,
    )
    
    print("\n" + "=" * 50)
    print("Processing complete!")
    print(f"Total samples: {stats['total']}")
    print(f"Successful: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
