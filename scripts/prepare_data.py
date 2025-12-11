#!/usr/bin/env python3
"""
Data Preparation Script for Medical Text-to-3D Fine-tuning

Pre-computes and caches:
1. Shap-E latents from 3D mesh files (STL/OBJ)
2. BioMedCLIP text embeddings from medical descriptions

Optimizations:
- Direct point cloud sampling from mesh (no Blender)
- PyVista rendering with RGBA, depth, and correct camera alignment
- Parallel processing with multiprocessing
- Batched GPU encoding

Usage:
    python scripts/prepare_data.py \
        --data_dir data/raw \
        --output_dir data/processed \
        --use_pyvista \
        --device cuda

Expected data directory structure:
    data/raw/
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
import trimesh

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from open_clip import create_model_and_transforms, get_tokenizer
from PIL import Image

# Configure PyVista for headless rendering
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['VTK_SILENCE_GET_VOID_POINTER_WARNINGS'] = '1'
import pyvista as pv
pv.OFF_SCREEN = True

sys.path.insert(0, str(Path(__file__).parent.parent))

from shap_e.models.download import load_model
from shap_e.util.collections import AttrDict
from shap_e.rendering.view_data import ProjectiveCamera


def load_biomedclip(device: torch.device) -> Tuple[nn.Module, nn.Module]:
    """Load BioMedCLIP model and tokenizer from HuggingFace."""
    model, _, _ = create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    tokenizer = get_tokenizer(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    
    return model, tokenizer


def encode_text_biomedclip(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int = 256,
    return_pooled: bool = True,
) -> torch.Tensor:
    """Encode text descriptions using BioMedCLIP."""
    with torch.no_grad():
        tokens = tokenizer(texts, context_length=max_length).to(device)
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        if return_pooled:
            return text_features  # [B, 512]
        else:
            return text_features.unsqueeze(1)  # [B, 1, 512]


def render_mesh_with_pyvista(
    mesh_path: str,
    num_views: int = 20,
    image_size: int = 256,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[ProjectiveCamera]]:
    """
    Render RGBA images, depth maps, and create corresponding Shap-E cameras.
    
    Returns:
        - rendered_views: List of RGBA images [num_views] x [H, W, 4]
        - rendered_depths: List of depth maps [num_views] x [H, W]
        - shap_cameras: List of ProjectiveCamera objects matching render viewpoints
    """
    try:
        mesh = trimesh.load(mesh_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            mesh = trimesh.util.concatenate(meshes) if meshes else mesh
        
        # Normalize to unit cube with margin
        bounds = mesh.bounds
        center = (bounds[0] + bounds[1]) / 2
        scale = np.max(bounds[1] - bounds[0])
        mesh.vertices = (mesh.vertices - center) / scale * 1.8
        
        # Convert to PyVista, preserving vertex colors
        vertices = mesh.vertices
        faces = np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces])
        pv_mesh = pv.PolyData(vertices, faces)
        
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            pv_mesh['colors'] = mesh.visual.vertex_colors[:, :3]
    except Exception:
        pv_mesh = pv.read(mesh_path)
        bounds = pv_mesh.bounds
        center = [(bounds[i * 2] + bounds[i * 2 + 1]) / 2 for i in range(3)]
        scale = max([bounds[i * 2 + 1] - bounds[i * 2] for i in range(3)])
        pv_mesh.points = (pv_mesh.points - center) / scale * 1.8

    rendered_views = []
    rendered_depths = []
    shap_cameras = []

    plotter = pv.Plotter(off_screen=True, window_size=[image_size, image_size])
    plotter.set_background("black", top="black")
    
    # Preserve vertex colors if available
    if pv_mesh.n_arrays > 0:
        plotter.add_mesh(pv_mesh, rgb=True, show_edges=False)
    else:
        plotter.add_mesh(pv_mesh, color='white', show_edges=False)

    for view_idx in range(num_views):
        angle = 2 * np.pi * view_idx / max(num_views, 1)
        radius = 2.0
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 1.0
        
        cam_pos = np.array([x, y, z], dtype=np.float32)
        focal_point = np.array([0, 0, 0], dtype=np.float32)
        view_up = np.array([0, 0, 1], dtype=np.float32)

        plotter.camera_position = [cam_pos, focal_point, view_up]
        
        # Capture RGBA with transparency
        img_rgba = plotter.screenshot(return_img=True, transparent_background=True)
        rendered_views.append(img_rgba)

        # Capture depth map
        depth_img = plotter.get_image_depth(fill_value=10.0)
        rendered_depths.append(depth_img)

        # Create matching Shap-E camera (Z points backwards from target to camera)
        z_vec = cam_pos - focal_point
        z_vec = z_vec / np.linalg.norm(z_vec)
        
        x_vec = np.cross(view_up, z_vec)
        if np.linalg.norm(x_vec) < 1e-5:
            x_vec = np.array([1, 0, 0], dtype=np.float32)
        x_vec = x_vec / np.linalg.norm(x_vec)
        
        y_vec = np.cross(z_vec, x_vec)
        y_vec = y_vec / np.linalg.norm(y_vec)

        camera = ProjectiveCamera(
            origin=cam_pos,
            x=x_vec * image_size,
            y=y_vec * image_size,
            z=z_vec,
            width=image_size,
            height=image_size,
            x_fov=np.float32(0.7),
            y_fov=np.float32(0.7),
        )
        shap_cameras.append(camera)

    plotter.close()
    
    return rendered_views, rendered_depths, shap_cameras


def sample_point_cloud_from_mesh(
    mesh_path: str,
    num_points: int = 16384,
) -> np.ndarray:
    """
    Sample point cloud from mesh surface.
    
    Returns:
        Point cloud array [num_points, 6] (xyz + rgb)
    """
    mesh = trimesh.load(mesh_path, force="mesh")

    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if len(meshes) == 0:
            raise ValueError(f"No valid meshes found in {mesh_path}")
        mesh = trimesh.util.concatenate(meshes)

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load {mesh_path} as a triangle mesh")

    # Normalize to unit cube
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    scale = np.max(bounds[1] - bounds[0])
    mesh.vertices = (mesh.vertices - center) / scale * 2.0

    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)

    # Get vertex colors if available
    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        vertex_colors = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
        face_colors = vertex_colors[mesh.faces[face_indices]]
        colors = face_colors.mean(axis=1)
    else:
        colors = np.ones((num_points, 3), dtype=np.float32) * 0.5

    point_cloud = np.concatenate([points, colors], axis=-1).astype(np.float32)
    return point_cloud


def create_dummy_camera(device: torch.device, image_size: int = 256):
    """Create dummy camera for Shap-E encoder (used in fast mode)."""
    camera = ProjectiveCamera(
        origin=np.array([0.0, 0.0, 2.0], dtype=np.float32),
        x=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        y=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        z=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        width=image_size,
        height=image_size,
        x_fov=np.float32(0.7),
        y_fov=np.float32(0.7),
    )
    return camera


def find_paired_files(
    data_dir: str,
    mesh_extensions: List[str] = [".stl", ".obj", ".ply", ".off", ".glb", ".gltf"],
) -> List[Tuple[str, str]]:
    """Find paired mesh and text files in data directory."""
    data_path = Path(data_dir)
    pairs = []

    mesh_files = []
    for ext in mesh_extensions:
        mesh_files.extend(data_path.glob(f"*{ext}"))
        mesh_files.extend(data_path.glob(f"*{ext.upper()}"))

    for mesh_file in mesh_files:
        stem = mesh_file.stem
        text_file = data_path / f"{stem}.txt"

        if text_file.exists():
            pairs.append((str(mesh_file), str(text_file)))
        else:
            print(f"Warning: No text file for {mesh_file.name}")

    return pairs


def process_single_sample(
    args: Tuple[str, str, str, str, int],
) -> Tuple[str, str, Optional[str]]:
    """
    Process single mesh-text pair for multiprocessing.
    
    Returns:
        (status, sample_name, error_message)
        status: 'skip' if already complete, 'processed' if cache created, 'error' if failed
    """
    mesh_path, text_path, output_dir, device_str, num_points = args

    sample_name = Path(mesh_path).stem
    latent_path = os.path.join(output_dir, f"{sample_name}_latent.pt")
    text_emb_path = os.path.join(output_dir, f"{sample_name}_text.pt")

    # Skip if already fully processed
    if os.path.exists(latent_path) and os.path.exists(text_emb_path):
        return ("skip", sample_name, None)

    # Sample point cloud and cache
    point_cloud = sample_point_cloud_from_mesh(mesh_path, num_points)
    pc_cache_path = os.path.join(output_dir, f"{sample_name}_pc_cache.npy")
    np.save(pc_cache_path, point_cloud)

    # Load and cache text
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    
    text_cache_path = os.path.join(output_dir, f"{sample_name}_text_cache.txt")
    with open(text_cache_path, "w", encoding="utf-8") as f:
        f.write(text)

    return ("processed", sample_name, None)


def process_dataset(
    data_dir: str,
    output_dir: str,
    device: torch.device,
    use_pooled_embeddings: bool = True,
    num_workers: int = 8,
    batch_size: int = 32,
    num_points: int = 16384,
    use_pyvista_rendering: bool = False,
    num_views: int = 20,
    image_size: int = 256,
) -> Dict[str, int]:
    """
    Process entire dataset and save cached embeddings.
    
    Args:
        data_dir: Directory containing mesh and text files
        output_dir: Directory to save cached tensors
        device: Device to use (cuda/cpu)
        use_pooled_embeddings: Use pooled BioMedCLIP embeddings
        num_workers: Parallel workers for mesh processing
        batch_size: Batch size for GPU encoding
        num_points: Points to sample per mesh
        use_pyvista_rendering: Use PyVista for multiview rendering (vs dummy views)
        num_views: Views to render (only with use_pyvista_rendering=True)
        image_size: Size of rendered images
    
    Returns:
        Processing statistics
    """
    os.makedirs(output_dir, exist_ok=True)

    pairs = find_paired_files(data_dir)
    print(f"Found {len(pairs)} mesh-text pairs")

    if len(pairs) == 0:
        print("No pairs found! Check your data directory structure.")
        return {"total": 0, "success": 0, "failed": 0}

    stats = {"total": len(pairs), "success": 0, "failed": 0}

    # Step 1: Sample point clouds in parallel
    print(f"\n[1/3] Sampling point clouds ({num_workers} workers)...")
    device_str = str(device)

    process_args = [
        (mesh_path, text_path, output_dir, device_str, num_points)
        for mesh_path, text_path in pairs
    ]

    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(process_single_sample, process_args),
                    total=len(pairs),
                    desc="Sampling",
                )
            )
    else:
        results = [
            process_single_sample(args)
            for args in tqdm(process_args, desc="Sampling")
        ]

    samples_to_process = []
    samples_skipped = 0
    for status, sample_name, error in results:
        if status == "skip":
            samples_skipped += 1
            stats["success"] += 1  # Already complete
        elif status == "processed":
            samples_to_process.append(sample_name)
        else:
            print(f"Error: {sample_name}: {error}")
            stats["failed"] += 1

    if samples_skipped > 0:
        print(f"Skipped {samples_skipped} already-processed samples")

    if len(samples_to_process) == 0:
        print("All samples already processed!")
        return stats

    print(f"Sampled {len(samples_to_process)} point clouds")

    # Step 2: Encode meshes with Shap-E
    print(f"\n[2/3] Encoding to Shap-E latents (batch_size={batch_size})...")
    transmitter = load_model("transmitter", device=device)
    transmitter.eval()
    for param in transmitter.parameters():
        param.requires_grad = False

    # Prepare view data
    if not use_pyvista_rendering:
        dummy_num_views = 1
        dummy_img = Image.new("RGB", (image_size, image_size), color=(128, 128, 128))
        dummy_depth = np.ones((image_size, image_size), dtype=np.float32) * 2.0
        dummy_alpha = np.ones((image_size, image_size), dtype=np.float32)
        dummy_camera = create_dummy_camera(device, image_size)

    for i in tqdm(range(0, len(samples_to_process), batch_size), desc="Encoding"):
        batch_samples = samples_to_process[i : i + batch_size]
        actual_batch_size = len(batch_samples)

        # Load point clouds
        point_clouds = []
        mesh_paths = []
        for sample_name in batch_samples:
            pc_cache_path = os.path.join(output_dir, f"{sample_name}_pc_cache.npy")
            pc = np.load(pc_cache_path)
            point_clouds.append(torch.from_numpy(pc).float())

            if use_pyvista_rendering:
                for mesh_path, _ in pairs:
                    if Path(mesh_path).stem == sample_name:
                        mesh_paths.append(mesh_path)
                        break

        batch_pc = torch.stack([pc.t() for pc in point_clouds]).to(device)

        # Create batch with views
        if use_pyvista_rendering:
            all_views = []
            all_depths = []
            all_alphas = []
            all_cameras = []

            for mesh_path in mesh_paths:
                imgs, depths_np, cams = render_mesh_with_pyvista(
                    mesh_path, num_views, image_size
                )

                current_views_pil = []
                current_alphas = []
                current_depths = []

                for j in range(len(imgs)):
                    img_arr = imgs[j]
                    rgb = img_arr[:, :, :3]
                    alpha = img_arr[:, :, 3].astype(np.float32) / 255.0

                    current_views_pil.append(Image.fromarray(rgb))
                    current_alphas.append(alpha)

                    d = depths_np[j]
                    if d.ndim == 3:
                        d = d[:, :, 0]
                    current_depths.append(d.astype(np.float32))

                all_views.append(current_views_pil)
                all_depths.append(current_depths)
                all_alphas.append(current_alphas)
                all_cameras.append(cams)

            batch_dict = AttrDict(
                points=batch_pc,
                views=all_views,
                depths=all_depths,
                view_alphas=all_alphas,
                cameras=all_cameras,
            )
        else:
            batch_dict = AttrDict(
                points=batch_pc,
                views=[
                    [dummy_img for _ in range(dummy_num_views)]
                    for _ in range(actual_batch_size)
                ],
                depths=[
                    [dummy_depth for _ in range(dummy_num_views)]
                    for _ in range(actual_batch_size)
                ],
                view_alphas=[
                    [dummy_alpha for _ in range(dummy_num_views)]
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

        # Save latents and clean up
        for j, sample_name in enumerate(batch_samples):
            latent_path = os.path.join(output_dir, f"{sample_name}_latent.pt")
            torch.save(latents[j].cpu(), latent_path)

            pc_cache_path = os.path.join(output_dir, f"{sample_name}_pc_cache.npy")
            if os.path.exists(pc_cache_path):
                os.remove(pc_cache_path)

    # Step 3: Encode texts with BioMedCLIP
    print(f"\n[3/3] Encoding with BioMedCLIP (batch_size={batch_size})...")
    biomedclip_model, biomedclip_tokenizer = load_biomedclip(device)

    for i in tqdm(range(0, len(samples_to_process), batch_size), desc="Encoding"):
        batch_samples = samples_to_process[i : i + batch_size]

        texts = []
        for sample_name in batch_samples:
            text_cache_path = os.path.join(output_dir, f"{sample_name}_text_cache.txt")
            with open(text_cache_path, "r", encoding="utf-8") as f:
                texts.append(f.read().strip())

        text_embeddings = encode_text_biomedclip(
            biomedclip_model,
            biomedclip_tokenizer,
            texts,
            device,
            return_pooled=use_pooled_embeddings,
        )

        for j, sample_name in enumerate(batch_samples):
            text_emb_path = os.path.join(output_dir, f"{sample_name}_text.pt")
            torch.save(text_embeddings[j].cpu(), text_emb_path)

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
        default=min(16, cpu_count()),
        help=f"Parallel workers for mesh processing (default: min(16, cpu_count))",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for GPU encoding (default: 32, optimized for A100)",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=16384,
        help="Points to sample per mesh (default: 16384)",
    )
    parser.add_argument(
        "--use_pyvista",
        action="store_true",
        help="Use PyVista for multiview rendering (vs dummy views)",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=20,
        help="Views to render with PyVista (default: 20, only with --use_pyvista)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Size of rendered images (default: 256)",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"\nDevice: {device}")
    print(f"Workers: {args.num_workers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Points per mesh: {args.num_points}")
    print(f"Rendering: {'PyVista' if args.use_pyvista else 'Dummy views'} ({args.num_views} views)")
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
    )

    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"Total: {stats['total']} | Success: {stats['success']} | Failed: {stats['failed']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
