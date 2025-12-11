import os
import glob
import json
import math
import argparse
from PIL import Image
from vllm import LLM, SamplingParams

# --- Configuration ---
# WARNING: Ensure this is a VLM (like Pixtral). Standard Mistral-Small is text-only.
# If using Pixtral, change to: "mistralai/Pixtral-12B-2409"
MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503" 

def parse_args():
    parser = argparse.ArgumentParser(description="vLLM Caption Generator")
    parser.add_argument("--data_dir", type=str, required=True, help="Root data directory containing 'renders'")
    parser.add_argument("--output_file", type=str, default="captions.jsonl", help="Base output filename")
    parser.add_argument("--num_chunks", type=int, default=1, help="Total number of chunks (nodes)")
    parser.add_argument("--chunk_idx", type=int, default=0, help="Current chunk index (0 to num_chunks-1)")
    # vLLM handles batching internally, but we use this to limit RAM usage by loading images in groups
    parser.add_argument("--loader_batch_size", type=int, default=10, help="How many images to load into RAM at once")
    return parser.parse_args()

def get_shard_files(root_dir, num_chunks, chunk_idx):
    # 1. Find all folders
    all_folders = sorted(glob.glob(os.path.join(root_dir, "*")))
    all_folders = [f for f in all_folders if os.path.isdir(f)]
    
    # 2. Shard the dataset
    total_items = len(all_folders)
    chunk_size = math.ceil(total_items / num_chunks)
    
    start_idx = chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, total_items)
    
    print(f"Node {chunk_idx}/{num_chunks} processing indices: {start_idx} to {end_idx}")
    return all_folders[start_idx:end_idx]

def create_grid_image(folder_path):
    views_map = {
        'front': 'front.png', 'back': 'back.png', 
        'left': 'left.png',   'right': 'right.png', 
        'top': 'top.png',     'bottom': 'bottom.png'
    }
    
    imgs = {}
    for k, v in views_map.items():
        path = os.path.join(folder_path, v)
        if os.path.exists(path):
            imgs[k] = Image.open(path).convert("RGB")
        else:
            imgs[k] = Image.new('RGB', (224, 224), (0, 0, 0))

    w, h = imgs['front'].size
    grid_img = Image.new('RGB', (w * 3, h * 2))
    
    grid_img.paste(imgs['front'], (0, 0))    
    grid_img.paste(imgs['right'], (w, 0))     
    grid_img.paste(imgs['back'],  (w*2, 0))   
    grid_img.paste(imgs['left'],  (0, h))     
    grid_img.paste(imgs['top'],   (w, h))     
    grid_img.paste(imgs['bottom'],(w*2, h))   
    
    return grid_img

def main():
    args = parse_args()
    
    # Setup paths
    render_dir = os.path.join(args.data_dir, "renders")
    base_name, ext = os.path.splitext(args.output_file)
    sharded_output_file = f"{base_name}_part_{args.chunk_idx}{ext}"
    
    # Get work for this node
    folders_to_process = get_shard_files(render_dir, args.num_chunks, args.chunk_idx)
    
    if not folders_to_process:
        print("No files to process for this shard.")
        return

    # Initialize vLLM
    print(f"Loading vLLM model: {MODEL_ID}...")
    llm = LLM(
        model=MODEL_ID,
        tokenizer_mode="mistral", # Ensure we use the correct tokenizer logic
        tensor_parallel_size=1,   # H100 80GB can fit this model on 1 GPU
        trust_remote_code=True,
        max_model_len=8192,       # Adjust based on image token usage
        limit_mm_per_prompt={"image": 1} # Allow 1 image per prompt
    )

    # Sampling parameters
    sampling_params = SamplingParams(
        max_tokens=200,    # CAP OUTPUT AT 200 TOKENS
        temperature=0.2,
        top_p=0.9
    )

    print(f"Starting inference on {len(folders_to_process)} items...")
    
    # Process in chunks to manage RAM (don't load 2000 images at once)
    batch_size = args.loader_batch_size
    mode = 'a' if os.path.exists(sharded_output_file) else 'w'
    
    # Open file once to append results
    with open(sharded_output_file, mode) as f_out:
        for i in range(0, len(folders_to_process), batch_size):
            batch_folders = folders_to_process[i : i + batch_size]
            
            # Prepare Batch Inputs
            prompts = []
            metadata = []
            
            for folder_path in batch_folders:
                folder_name = os.path.basename(folder_path)
                parts = folder_name.split('_', 1)
                label = parts[1] if len(parts) > 1 else folder_name
                
                # Create Image
                image = create_grid_image(folder_path)
                
                # Construct vLLM Message
                # vLLM detects the "image" type and handles the tokens automatically
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": f"Below are 6 views of a 3D medical model labeled as '{label}'. Please provide a very detailed technical description of the geometry, structure, and any visible features or abnormalities suitable for 3D modeling."}
                        ]
                    }
                ]
                prompts.append(messages)
                metadata.append({"id": folder_name, "label": label})

            # Run Batch Inference
            outputs = llm.chat(prompts, sampling_params=sampling_params)

            # Write Results
            for j, output in enumerate(outputs):
                generated_text = output.outputs[0].text.strip()
                record = {
                    "id": metadata[j]["id"],
                    "label": metadata[j]["label"],
                    "description": generated_text
                }
                f_out.write(json.dumps(record) + "\n")
            
            f_out.flush()

    print(f"Shard {args.chunk_idx} Done! Saved to {sharded_output_file}")

if __name__ == "__main__":
    main()