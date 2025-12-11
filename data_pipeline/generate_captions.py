import os
import glob
import json
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
from vllm import LLM, SamplingParams

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"

# Distributed settings (SLURM)
RANK = int(os.environ.get("SLURM_PROCID", 0))
WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", 1))

if RANK == 0:
    print(f"World Size: {WORLD_SIZE}")


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM Caption Generator")
    parser.add_argument("--data_dir", type=str, required=True, help="Root data directory containing 'renders'")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory for checkpoint files")
    parser.add_argument("--loader_batch_size", type=int, default=10, help="How many images to load into RAM at once")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N batches")
    return parser.parse_args()


def get_all_folders(root_dir):
    """Get all render folders sorted."""
    all_folders = sorted(glob.glob(os.path.join(root_dir, "*")))
    all_folders = [f for f in all_folders if os.path.isdir(f)]
    return all_folders


def get_shard_indices(total_items, world_size, rank):
    """Split indices across ranks using numpy."""
    all_indices = np.arange(total_items)
    split_indices = np.array_split(all_indices, world_size)[rank]
    print(f"[{rank}] Processing {len(split_indices)}/{total_items} samples")
    return split_indices.tolist()


def load_checkpoint(checkpoint_path):
    """Load completed indices from checkpoint file."""
    completed = set()
    if os.path.exists(checkpoint_path):
        print(f"[{RANK}] Loading checkpoint: {checkpoint_path}")
        with open(checkpoint_path, "r") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    completed.add(record["id"])
                except json.JSONDecodeError:
                    continue
        print(f"[{RANK}] Resuming with {len(completed)} completed samples")
    return completed


def create_grid_image(folder_path):
    views_map = {
        "front": "front.png",
        "back": "back.png",
        "left": "left.png",
        "right": "right.png",
        "top": "top.png",
        "bottom": "bottom.png",
    }

    imgs = {}
    for k, v in views_map.items():
        path = os.path.join(folder_path, v)
        if os.path.exists(path):
            imgs[k] = Image.open(path).convert("RGB")
        else:
            imgs[k] = Image.new("RGB", (224, 224), (0, 0, 0))

    w, h = imgs["front"].size
    grid_img = Image.new("RGB", (w * 3, h * 2))

    grid_img.paste(imgs["front"], (0, 0))
    grid_img.paste(imgs["right"], (w, 0))
    grid_img.paste(imgs["back"], (w * 2, 0))
    grid_img.paste(imgs["left"], (0, h))
    grid_img.paste(imgs["top"], (w, h))
    grid_img.paste(imgs["bottom"], (w * 2, h))

    return grid_img


def main():
    args = parse_args()

    # Setup paths
    render_dir = os.path.join(args.data_dir, "renders")
    checkpoint_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / f"rank{RANK}.jsonl"

    # Get all folders and split across ranks
    all_folders = get_all_folders(render_dir)
    if not all_folders:
        print(f"[{RANK}] No folders found in {render_dir}")
        return

    # Get indices for this rank
    my_indices = get_shard_indices(len(all_folders), WORLD_SIZE, RANK)

    # Load checkpoint to find completed work
    completed_ids = load_checkpoint(checkpoint_path)

    # Filter remaining work
    remaining_indices = [idx for idx in my_indices if os.path.basename(all_folders[idx]) not in completed_ids]
    print(f"[{RANK}] Remaining samples: {len(remaining_indices)}")

    if len(remaining_indices) == 0:
        print(f"[{RANK}] All samples already processed!")
        return

    # Initialize vLLM
    print(f"[{RANK}] Loading vLLM model: {MODEL_ID}...")
    llm = LLM(
        model=MODEL_ID,
        tokenizer_mode="mistral",
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=8192,
        limit_mm_per_prompt={"image": 1},
    )

    # Sampling parameters
    sampling_params = SamplingParams(max_tokens=200, temperature=0.2, top_p=0.9)

    print(f"[{RANK}] Starting inference on {len(remaining_indices)} items...")

    batch_size = args.loader_batch_size
    batch_count = 0

    # Open file in append mode
    with open(checkpoint_path, "a") as f_out:
        for i in range(0, len(remaining_indices), batch_size):
            batch_indices = remaining_indices[i : i + batch_size]
            batch_folders = [all_folders[idx] for idx in batch_indices]

            # Prepare Batch Inputs
            prompts = []
            metadata = []

            for folder_path in batch_folders:
                folder_name = os.path.basename(folder_path)
                parts = folder_name.split("_", 1)
                label = parts[1] if len(parts) > 1 else folder_name

                # Create Image
                image = create_grid_image(folder_path)

                # Construct vLLM Message
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {
                                "type": "text",
                                "text": f"Analyze these 6 views of a medical object labeled '{label}'. Provide a concise anatomical description focusing on specific shape, surface texture, and visible structural irregularities. Do not use introduction sentences.",
                            },
                        ],
                    }
                ]
                prompts.append(messages)
                metadata.append({"id": folder_name, "label": label})

            # Run Batch Inference
            outputs = llm.chat(prompts, sampling_params=sampling_params)

            # Write Results
            for j, output in enumerate(outputs):
                generated_text = output.outputs[0].text.strip()
                record = {"id": metadata[j]["id"], "label": metadata[j]["label"], "description": generated_text}
                f_out.write(json.dumps(record) + "\n")

            batch_count += 1

            # Periodic checkpoint flush
            if batch_count % args.save_every == 0:
                f_out.flush()
                print(f"[{RANK}] Checkpoint: {i + len(batch_indices)}/{len(remaining_indices)} processed")

    print(f"[{RANK}] Done! Saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
