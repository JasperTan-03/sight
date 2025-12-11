#!/usr/bin/env python3
"""
Data Preparation Script for Medical Text-to-3D Fine-tuning (Optimized)

This script pre-computes and caches:
1. Shap-E latents from 3D mesh files (STL/OBJ) - NO BLENDER REQUIRED
2. BioMedCLIP text embeddings from medical descriptions

OPTIMIZATIONS:
- Direct point cloud sampling from mesh (no Blender rendering)
- Parallel processing of multiple samples
- Batched text encoding for efficiency
- GPU-accelerated where possible

Usage:
    python scripts/prepare_data.py \
        --data_dir /path/to/dataset \
        --output_dir /path/to/cached \
        --device cuda \
        --num_workers 4

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
from multiprocessing import Pool, cpu_count
from functools import partial
import trimesh

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from open_clip import create_model_and_transforms, get_tokenizer
import pyvista as pv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shap_e.models.download import load_model
from shap_e.util.collections import AttrDict
from shap_e.rendering.view_data import ProjectiveCamera


def load_biomedclip(device: torch.device) -> Tuple[nn.Module, nn.Module]:
    """
    Load BioMedCLIP model and tokenizer from HuggingFace.

    Returns:
        Tuple of (model, tokenizer)
    """
    print("Loading BioMedCLIP model...")

    # Load BioMedCLIP using open_clip
    model, _, preprocess = create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    tokenizer = get_tokenizer(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
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


def render_mesh_with_pyvista(
    mesh_path: str,
    num_views: int = 1,
    image_size: int = 256,
    verbose: bool = False,
) -> List[np.ndarray]:
    """
    Render mesh views using PyVista (NO X DISPLAY REQUIRED).

    PyVista is a Python-native VTK wrapper that works without Blender/xvfb.

    Args:
        mesh_path: Path to mesh file
        num_views: Number of views to render
        image_size: Size of rendered images
        verbose: Print progress

    Returns:
        List of rendered images as numpy arrays [num_views] x [H, W, 3]
    """
    # Set offscreen rendering (no display needed!)
    pv.OFF_SCREEN = True

    if verbose:
        print(f"  Rendering {num_views} views with PyVista...")

    # Load mesh
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes) if meshes else mesh

    # Convert to PyVista
    vertices = mesh.vertices
    faces = np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces])
    pv_mesh = pv.PolyData(vertices, faces)

    # Normalize mesh
    bounds = pv_mesh.bounds
    center = [(bounds[i * 2] + bounds[i * 2 + 1]) / 2 for i in range(3)]
    scale = max([bounds[i * 2 + 1] - bounds[i * 2] for i in range(3)])
    pv_mesh.points = (pv_mesh.points - center) / scale * 2.0

    rendered_views = []

    # Create plotter
    plotter = pv.Plotter(off_screen=True, window_size=[image_size, image_size])
    plotter.add_mesh(pv_mesh, color="white", show_edges=False)

    for view_idx in range(num_views):
        # Set camera position (rotate around object)
        angle = 2 * np.pi * view_idx / max(num_views, 1)
        camera_pos = [2.0 * np.cos(angle), 2.0 * np.sin(angle), 1.0]
        plotter.camera_position = [
            camera_pos,  # Position
            (0, 0, 0),  # Focal point
            (0, 0, 1),  # View up
        ]

        # Render
        img = plotter.screenshot(return_img=True, transparent_background=False)
        rendered_views.append(img)

    plotter.close()

    return rendered_views


def sample_point_cloud_from_mesh(
    mesh_path: str,
    num_points: int = 16384,  # 2^14, same as Shap-E default
    verbose: bool = False,
) -> np.ndarray:
    """
    Directly sample a point cloud from mesh surface (NO BLENDER REQUIRED).

    This is MUCH faster than Blender rendering (seconds vs minutes).

    Supports: .stl, .obj, .ply, .off, .glb, .gltf

    Args:
        mesh_path: Path to the 3D mesh file
        num_points: Number of points to sample
        verbose: Whether to print progress

    Returns:
        Point cloud array of shape [num_points, 6] (xyz + rgb)
    """
    if verbose:
        print(f"  Loading mesh: {mesh_path}")

    # Load mesh with trimesh
    mesh = trimesh.load(mesh_path, force="mesh")

    if isinstance(mesh, trimesh.Scene):
        # Handle scene files (like some GLB files)
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if len(meshes) == 0:
            raise ValueError(f"No valid meshes found in {mesh_path}")
        mesh = trimesh.util.concatenate(meshes)

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load {mesh_path} as a triangle mesh")

    # Normalize mesh to unit cube centered at origin (same as Shap-E)
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    scale = np.max(bounds[1] - bounds[0])
    mesh.vertices = (mesh.vertices - center) / scale * 2.0

    if verbose:
        print(f"  Sampling {num_points} points from surface...")

    # Sample points uniformly on mesh surface
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)

    # Get vertex colors if available, otherwise use default gray
    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        # Interpolate colors from vertices to sampled points
        vertex_colors = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
        face_colors = vertex_colors[mesh.faces[face_indices]]
        colors = face_colors.mean(axis=1)  # Average vertex colors of each face
    else:
        # Default to gray color
        colors = np.ones((num_points, 3), dtype=np.float32) * 0.5

    # Combine points and colors: [num_points, 6] (xyz + rgb)
    point_cloud = np.concatenate([points, colors], axis=-1).astype(np.float32)

    if verbose:
        print(f"  Point cloud shape: {point_cloud.shape}")

    return point_cloud


def create_dummy_camera(device: torch.device, image_size: int = 256):
    """Create a minimal dummy camera for Shap-E encoder."""
    # Simple camera looking at origin (ensure float32 for compatibility)
    # CRITICAL: x_fov and y_fov must be numpy float32 to prevent float64 propagation
    camera = ProjectiveCamera(
        origin=np.array([0.0, 0.0, 2.0], dtype=np.float32),
        x=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        y=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        z=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        width=image_size,
        height=image_size,
        x_fov=np.float32(0.7),  # Must be np.float32, not Python float!
        y_fov=np.float32(0.7),  # Must be np.float32, not Python float!
    )
    return camera


def mesh_to_batch_with_views(
    mesh_path: str,
    device: torch.device,
    num_points: int = 16384,
    num_views: int = 1,  # Minimal views
    image_size: int = 256,  # Must match encoder's expected size
    verbose: bool = False,
) -> AttrDict:
    """
    Convert mesh to batch format for Shap-E encoder with minimal dummy views.

    The Shap-E encoder expects both point clouds AND multiview images.
    We provide minimal dummy views to satisfy this requirement.

    Args:
        mesh_path: Path to the 3D mesh file
        device: Device to use
        num_points: Number of points to sample
        num_views: Number of dummy views (default: 1 for speed)
        image_size: Size of dummy images (must be 256 to match encoder)
        verbose: Whether to print progress

    Returns:
        AttrDict with 'points' and dummy 'views', 'cameras', etc.
    """
    from PIL import Image

    # Sample point cloud directly from mesh
    point_cloud = sample_point_cloud_from_mesh(mesh_path, num_points, verbose)

    # Convert to torch tensor and move to device
    # Shape: [num_points, 6] -> [6, num_points] (channels first for Shap-E)
    points_tensor = torch.from_numpy(point_cloud).float().to(device)
    points_tensor = points_tensor.t()  # Transpose to [6, num_points]

    # Create minimal dummy views (grayscale images)
    dummy_img = Image.new("RGB", (image_size, image_size), color=(128, 128, 128))
    views = [[dummy_img for _ in range(num_views)]]  # [batch_size, num_views]

    # Create minimal dummy depth maps
    dummy_depth = np.ones((image_size, image_size), dtype=np.float32) * 2.0
    depths = [[dummy_depth for _ in range(num_views)]]

    # Create minimal dummy alphas
    dummy_alpha = np.ones((image_size, image_size), dtype=np.float32)
    view_alphas = [[dummy_alpha for _ in range(num_views)]]

    # Create minimal dummy cameras
    cameras = [[create_dummy_camera(device, image_size) for _ in range(num_views)]]

    # Create batch with all required fields
    batch = AttrDict(
        points=points_tensor.unsqueeze(0),  # [1, 6, num_points]
        views=views,
        depths=depths,
        view_alphas=view_alphas,
        cameras=cameras,
    )

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
    mesh_extensions: List[str] = [".stl", ".obj", ".ply", ".off", ".glb", ".gltf"],
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
        mesh_files.extend(data_path.glob(f"*{ext}"))
        mesh_files.extend(data_path.glob(f"*{ext.upper()}"))

    # Match with text files
    for mesh_file in mesh_files:
        stem = mesh_file.stem
        text_file = data_path / f"{stem}.txt"

        if text_file.exists():
            pairs.append((str(mesh_file), str(text_file)))
        else:
            print(f"Warning: No text file found for {mesh_file.name}")

    return pairs


def process_single_sample(
    args: Tuple[str, str, str, str, int, bool],
) -> Tuple[bool, str, Optional[str]]:
    """
    Process a single mesh-text pair (for multiprocessing).

    Returns:
        (success, sample_name, error_message)
    """
    mesh_path, text_path, output_dir, device_str, num_points, verbose = args

    device = torch.device(device_str)
    sample_name = Path(mesh_path).stem
    latent_path = os.path.join(output_dir, f"{sample_name}_latent.pt")
    text_emb_path = os.path.join(output_dir, f"{sample_name}_text.pt")

    # Skip if already processed
    if os.path.exists(latent_path) and os.path.exists(text_emb_path):
        return (True, sample_name, None)

    # Sample point cloud from mesh (NO BLENDER)
    point_cloud = sample_point_cloud_from_mesh(mesh_path, num_points, verbose)

    # Save point cloud for later batch encoding
    pc_cache_path = os.path.join(output_dir, f"{sample_name}_pc_cache.npy")
    np.save(pc_cache_path, point_cloud)

    # Load text
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # Save text for later batch encoding
    text_cache_path = os.path.join(output_dir, f"{sample_name}_text_cache.txt")
    with open(text_cache_path, "w", encoding="utf-8") as f:
        f.write(text)

    return (True, sample_name, None)


def process_dataset(
    data_dir: str,
    output_dir: str,
    device: torch.device,
    use_pooled_embeddings: bool = True,
    num_workers: int = 1,
    batch_size: int = 8,
    num_points: int = 16384,
    use_pyvista_rendering: bool = False,
    num_views: int = 1,
    image_size: int = 256,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    Process the entire dataset and save cached embeddings (OPTIMIZED).

    OPTIMIZATIONS:
    - No Blender required (direct mesh sampling OR PyVista rendering)
    - Parallel mesh processing
    - Batched encoding on GPU

    Args:
        data_dir: Directory containing mesh and text files
        output_dir: Directory to save cached tensors
        device: Device to use
        use_pooled_embeddings: Whether to use pooled BioMedCLIP embeddings
        num_workers: Number of parallel workers for mesh processing
        batch_size: Batch size for encoding
        num_points: Number of points to sample per mesh
        use_pyvista_rendering: If True, use PyVista for real multiview rendering;
                               if False, use fast dummy views (recommended for medical STLs)
        num_views: Number of views to render (only used with use_pyvista_rendering=True)
        image_size: Size of rendered/dummy images (default: 256)
        verbose: Whether to print detailed progress

    Returns:
        Dictionary with processing statistics
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find paired files
    pairs = find_paired_files(data_dir)
    print(f"Found {len(pairs)} mesh-text pairs")

    if len(pairs) == 0:
        print("No pairs found! Check your data directory structure.")
        return {"total": 0, "success": 0, "failed": 0}

    stats = {"total": len(pairs), "success": 0, "failed": 0}

    # Step 1: Process meshes in parallel (CPU-bound)
    print(
        f"\nStep 1/3: Sampling point clouds from meshes (using {num_workers} workers)..."
    )
    device_str = str(device)

    # Prepare arguments for multiprocessing
    process_args = [
        (mesh_path, text_path, output_dir, device_str, num_points, verbose)
        for mesh_path, text_path in pairs
    ]

    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(process_single_sample, process_args),
                    total=len(pairs),
                    desc="Sampling meshes",
                )
            )
    else:
        results = [
            process_single_sample(args)
            for args in tqdm(process_args, desc="Sampling meshes")
        ]

    # Collect successful samples
    successful_samples = []
    for success, sample_name, error in results:
        if success:
            successful_samples.append(sample_name)
        else:
            print(f"Error processing {sample_name}: {error}")
            stats["failed"] += 1

    if len(successful_samples) == 0:
        print("No samples successfully processed!")
        return stats

    print(f"Successfully sampled {len(successful_samples)} point clouds")

    # Step 2: Batch encode meshes with Shap-E (GPU-accelerated)
    print(f"\nStep 2/3: Encoding point clouds to latents (batch_size={batch_size})...")
    transmitter = load_model("transmitter", device=device)
    transmitter.eval()
    for param in transmitter.parameters():
        param.requires_grad = False

    from PIL import Image

    # Prepare view data based on rendering mode
    if not use_pyvista_rendering:
        # Fast mode: Create dummy views once (reuse for all samples)
        dummy_num_views = 1  # Minimal for speed
        dummy_img = Image.new("RGB", (image_size, image_size), color=(128, 128, 128))
        dummy_depth = np.ones((image_size, image_size), dtype=np.float32) * 2.0
        dummy_alpha = (
            np.ones((image_size, image_size), dtype=np.float32) * 1.0
        )  # Full alpha
        dummy_camera = create_dummy_camera(device, image_size)

    for i in tqdm(
        range(0, len(successful_samples), batch_size), desc="Encoding meshes"
    ):
        batch_samples = successful_samples[i : i + batch_size]
        actual_batch_size = len(batch_samples)

        # Load point clouds
        point_clouds = []
        mesh_paths = []
        for sample_name in batch_samples:
            pc_cache_path = os.path.join(output_dir, f"{sample_name}_pc_cache.npy")
            pc = np.load(pc_cache_path)
            point_clouds.append(torch.from_numpy(pc).float())

            # Find original mesh path for PyVista rendering
            if use_pyvista_rendering:
                for mesh_path, _ in pairs:
                    if Path(mesh_path).stem == sample_name:
                        mesh_paths.append(mesh_path)
                        break

        # Stack into batch [batch_size, 6, num_points]
        batch_pc = torch.stack([pc.t() for pc in point_clouds]).to(device)

        # Create batch with views (real or dummy)
        if use_pyvista_rendering:
            # Real rendering mode: Render views with PyVista
            all_views = []
            all_depths = []
            all_alphas = []
            all_cameras = []

            for mesh_path in mesh_paths:
                # Render views with PyVista
                rendered_imgs = render_mesh_with_pyvista(
                    mesh_path,
                    num_views=num_views,
                    image_size=image_size,
                    verbose=verbose,
                )

                # Convert numpy arrays to PIL Images
                views_pil = [Image.fromarray(img) for img in rendered_imgs]
                all_views.append(views_pil)

                # Create depth and alpha for each view (dummy for now, could be improved)
                depths = [
                    np.ones((image_size, image_size), dtype=np.float32) * 2.0
                    for _ in range(num_views)
                ]
                alphas = [
                    np.ones((image_size, image_size), dtype=np.float32)
                    for _ in range(num_views)
                ]
                cameras = [
                    create_dummy_camera(device, image_size) for _ in range(num_views)
                ]

                all_depths.append(depths)
                all_alphas.append(alphas)
                all_cameras.append(cameras)

            batch_dict = AttrDict(
                points=batch_pc,
                views=all_views,
                depths=all_depths,
                view_alphas=all_alphas,
                cameras=all_cameras,
            )
        else:
            # Fast mode: Use minimal dummy views (ensuring float32 dtype throughout)
            batch_dict = AttrDict(
                points=batch_pc,  # Already float32
                views=[
                    [dummy_img for _ in range(dummy_num_views)]
                    for _ in range(actual_batch_size)
                ],
                depths=[
                    [dummy_depth.astype(np.float32) for _ in range(dummy_num_views)]
                    for _ in range(actual_batch_size)
                ],
                view_alphas=[
                    [dummy_alpha.astype(np.float32) for _ in range(dummy_num_views)]
                    for _ in range(actual_batch_size)
                ],
                cameras=[
                    [dummy_camera for _ in range(dummy_num_views)]
                    for _ in range(actual_batch_size)
                ],
            )

        # Encode batch
        with torch.no_grad():
            latents = transmitter.encoder.encode_to_bottleneck(batch_dict)

        # Save individual latents
        for j, sample_name in enumerate(batch_samples):
            latent_path = os.path.join(output_dir, f"{sample_name}_latent.pt")
            torch.save(latents[j].cpu(), latent_path)

            # Clean up cache
            pc_cache_path = os.path.join(output_dir, f"{sample_name}_pc_cache.npy")
            if os.path.exists(pc_cache_path):
                os.remove(pc_cache_path)

    # Step 3: Batch encode texts with BioMedCLIP (GPU-accelerated)
    print(f"\nStep 3/3: Encoding texts with BioMedCLIP (batch_size={batch_size})...")
    biomedclip_model, biomedclip_tokenizer = load_biomedclip(device)

    for i in tqdm(range(0, len(successful_samples), batch_size), desc="Encoding texts"):
        batch_samples = successful_samples[i : i + batch_size]

        # Load texts
        texts = []
        for sample_name in batch_samples:
            text_cache_path = os.path.join(output_dir, f"{sample_name}_text_cache.txt")
            with open(text_cache_path, "r", encoding="utf-8") as f:
                texts.append(f.read().strip())

        # Encode batch
        text_embeddings = encode_text_biomedclip(
            biomedclip_model,
            biomedclip_tokenizer,
            texts,
            device,
            return_pooled=use_pooled_embeddings,
        )

        # Save individual embeddings
        for j, sample_name in enumerate(batch_samples):
            text_emb_path = os.path.join(output_dir, f"{sample_name}_text.pt")
            torch.save(text_embeddings[j].cpu(), text_emb_path)

            # Clean up cache
            text_cache_path = os.path.join(output_dir, f"{sample_name}_text_cache.txt")
            if os.path.exists(text_cache_path):
                os.remove(text_cache_path)

            stats["success"] += 1

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
        "--use_sequence_embeddings",
        action="store_true",
        help="Use full sequence embeddings instead of pooled [CLS] token",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=min(4, cpu_count()),
        help=f"Number of parallel workers for mesh processing (default: {min(4, cpu_count())})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for GPU encoding (default: 8)",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=16384,
        help="Number of points to sample per mesh (default: 16384)",
    )
    parser.add_argument(
        "--use_pyvista",
        action="store_true",
        help="Use PyVista for real multiview rendering instead of dummy views (slower but higher quality)",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=20,
        help="Number of views to render with PyVista (default: 20, only used with --use_pyvista)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Size of rendered images (default: 256)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Parallel workers: {args.num_workers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Points per mesh: {args.num_points}")
    if args.use_pyvista:
        print(f"Rendering mode: PyVista ({args.num_views} views)")
    else:
        print(f"Rendering mode: Dummy views (fast)")
    print("=" * 60)

    stats = process_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=device,
        use_pooled_embeddings=not args.use_sequence_embeddings,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        num_points=args.num_points,
        use_pyvista_rendering=args.use_pyvista,
        num_views=args.num_views,
        image_size=args.image_size,
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
