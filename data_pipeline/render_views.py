import pyvista as pv
import os
import glob
import numpy as np
import multiprocessing as mp
import argparse
from tqdm import tqdm

# --- CONFIGURATION ---
IMG_SIZE = (512, 512)
VIEWS = {
    'front':  ((0, 0, 2.5), (0, 1, 0)),
    'back':   ((0, 0, -2.5), (0, 1, 0)),
    'left':   ((-2.5, 0, 0), (0, 1, 0)),
    'right':  ((2.5, 0, 0), (0, 1, 0)),
    'top':    ((0, 2.5, 0), (0, 0, 1)),
    'bottom': ((0, -2.5, 0), (0, 0, 1))
}

def parse_args():
    parser = argparse.ArgumentParser(description="MedShapeNet Renderer")
    parser.add_argument("--data_dir", type=str, required=True, help="Root data directory containing 'raw_models'")
    parser.add_argument("--workers", type=int, default=16, help="Number of rendering processes")
    return parser.parse_args()

def render_worker(args_bundle):
    """
    Worker function. unpacking args inside to handle multiprocessing pickling easily.
    args_bundle: (file_path, output_root)
    """
    file_path, output_root = args_bundle
    
    try:
        filename = os.path.basename(file_path)
        file_id = os.path.splitext(filename)[0]
        save_folder = os.path.join(output_root, file_id)
        
        # 1. Skip if done
        if os.path.exists(save_folder) and len(glob.glob(os.path.join(save_folder, '*.png'))) == 6:
            return "SKIPPED"

        # 2. Check File Size (Sanity check for empty downloads)
        if os.path.getsize(file_path) < 1024:
            return f"BAD_SIZE: {filename}"

        os.makedirs(save_folder, exist_ok=True)

        # 3. Load Mesh
        try:
            mesh = pv.read(file_path)
        except Exception:
            return f"CORRUPT: {filename}"

        if mesh.n_points == 0:
            return f"EMPTY: {filename}"

        # 4. Center and Normalize
        center = np.array(mesh.center)
        mesh.translate(-center, inplace=True)
        if mesh.length > 0:
            mesh.scale(1.0 / mesh.length, inplace=True)

        # 5. Render (Headless OSMesa)
        plotter = pv.Plotter(off_screen=True, window_size=IMG_SIZE)
        plotter.add_mesh(mesh, color='white', smooth_shading=True, specular=0.5)
        plotter.set_background('black') 
        
        for view_name, (pos, up) in VIEWS.items():
            plotter.camera_position = [pos, (0, 0, 0), up]
            plotter.render()
            plotter.screenshot(os.path.join(save_folder, f"{view_name}.png"))
            
        plotter.close()
        plotter.deep_clean()
        return "SUCCESS"
        
    except Exception as e:
        return f"ERROR: {filename} - {e}"

def main():
    args = parse_args()
    
    input_dir = os.path.join(args.data_dir, "raw_models")
    render_dir = os.path.join(args.data_dir, "renders")

    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist. Run download script first.")
        return

    os.makedirs(render_dir, exist_ok=True)

    # 2. Find Files
    print(f"Scanning {input_dir} for models...")
    extensions = ['*.stl', '*.obj', '*.ply']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    total = len(files)
    print(f"Found {total} files. Processing with {args.workers} workers...")

    # 3. Prepare Arguments for Workers
    worker_args = [(f, render_dir) for f in files]

    # 4. Parallel Execution
    # 'spawn' is REQUIRED for PyVista/VTK multiprocessing
    ctx = mp.get_context('spawn')
    
    counts = {"SUCCESS": 0, "SKIPPED": 0, "ERROR": 0}

    with ctx.Pool(processes=args.workers) as pool:
        for res in tqdm(pool.imap_unordered(render_worker, worker_args, chunksize=5), total=total, unit="mesh"):
            if res in counts:
                counts[res] += 1
            else:
                # Handle error messages
                counts["ERROR"] += 1
                if "ERROR" in res or "BAD" in res:
                    # Optional: Write to log file here
                    pass

    print("\n" + "="*30)
    print(f" RENDER COMPLETE")
    print(f" Stats: {counts}")
    print("="*30)

if __name__ == "__main__":
    main()