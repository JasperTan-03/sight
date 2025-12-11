import os
import glob
import json
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description="Vista Caption Generator")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory containing 'renders'")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory for output files")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (Vista can likely handle 4-8)")
    return parser.parse_args()


def get_dist_info():
    """Get SLURM rank info for data sharding."""
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    return rank, world_size


def create_grid_image(folder_path):
    """Stitches 6 views into a single 3x2 grid."""
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
    # 3 wide, 2 tall
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
    rank, world_size = get_dist_info()

    # 1. Setup Device (Simple for Vista: 1 GPU per task)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if rank == 0:
        print(f"--- Vista Inference Started ---")
        print(f"World Size: {world_size}")
        print(f"Model: {MODEL_ID} (BF16 Full Precision)")

    # 2. Load Model (Native BF16, Flash Attention)
    # We use device_map="auto" which puts the whole model on the single GPU efficiently
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # 3. Data Sharding
    render_dir = os.path.join(args.data_dir, "renders")
    all_folders = sorted([f for f in glob.glob(os.path.join(render_dir, "*")) if os.path.isdir(f)])

    if not all_folders:
        print(f"[{rank}] No folders found in {render_dir}")
        return

    # Split work deterministically
    my_indices = np.array_split(np.arange(len(all_folders)), world_size)[rank]
    my_folders = [all_folders[i] for i in my_indices]

    # 4. Checkpoint Management
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"rank{rank}.jsonl")

    completed_ids = set()
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r") as f:
            for line in f:
                try:
                    completed_ids.add(json.loads(line)["id"])
                except:
                    pass

    # Filter what's left
    todo_folders = [f for f in my_folders if os.path.basename(f) not in completed_ids]
    print(f"[{rank}] Processing {len(todo_folders)} items (skipped {len(completed_ids)})")

    # 5. Inference Loop
    # Write in append mode so we don't lose progress if it crashes
    with open(ckpt_path, "a") as f_out:
        for i in tqdm(range(0, len(todo_folders), args.batch_size), desc=f"Rank {rank}"):
            batch_paths = todo_folders[i : i + args.batch_size]

            # Prepare Batch
            texts = []
            images = []
            metadata = []

            for folder in batch_paths:
                folder_name = os.path.basename(folder)
                parts = folder_name.split("_", 1)
                label = parts[1] if len(parts) > 1 else folder_name

                # Image processing
                image = create_grid_image(folder)

                # Qwen Chat Template
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
                # Prepare text prompt
                text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                texts.append(text_prompt)
                images.append(image)
                metadata.append({"id": folder_name, "label": label})

            # Tokenize Batch
            inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
            inputs = inputs.to(model.device)

            # Generate
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=256, temperature=0.2, top_p=0.9, do_sample=True)

            # Decode (Trim input tokens to fix the "empty string" or "repetition" bug)
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            # Save Results
            for meta, desc in zip(metadata, output_texts):
                record = {"id": meta["id"], "label": meta["label"], "description": desc.strip()}
                f_out.write(json.dumps(record) + "\n")

            # Flush periodically
            f_out.flush()

    print(f"[{rank}] Finished!")


if __name__ == "__main__":
    main()
