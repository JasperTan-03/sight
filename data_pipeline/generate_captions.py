import os
import glob
import json
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoTokenizer, AutoModelForImageTextToText, BitsAndBytesConfig

# --- Configuration ---
MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

def parse_args():
    parser = argparse.ArgumentParser(description="MedShapeNet Caption Generator")
    parser.add_argument("--data_dir", type=str, required=True, help="Root data directory containing 'renders'")
    parser.add_argument("--output_file", type=str, default="captions.jsonl", help="Output JSONL file path")
    parser.add_argument("--cache_dir", type=str, default=None, help="HuggingFace cache directory (optional)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()

class MedShapeDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # Only look for folders that actually contain images
        self.folders = sorted(glob.glob(os.path.join(root_dir, "*")))
        self.folders = [f for f in self.folders if os.path.isdir(f)]

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder_path = self.folders[idx]
        folder_name = os.path.basename(folder_path)
        
        # Extract label from filename structure '000000_label'
        parts = folder_name.split('_', 1)
        label = parts[1] if len(parts) > 1 else folder_name
        
        # Grid layout order
        views_map = {
            'front': 'front.png', 'back': 'back.png', 
            'left': 'left.png',   'right': 'right.png', 
            'top': 'top.png',     'bottom': 'bottom.png'
        }
        
        # Load all images
        imgs = {}
        for k, v in views_map.items():
            path = os.path.join(folder_path, v)
            if os.path.exists(path):
                imgs[k] = Image.open(path).convert("RGB")
            else:
                imgs[k] = Image.new('RGB', (224, 224), (0, 0, 0))

        # Create 3x2 Grid
        # Row 1: Left, Front, Right
        # Row 2: Top, Back, Bottom
        # (You can adjust this layout logic as per your preference, 
        #  kept consistent with your original logic roughly)
        w, h = imgs['front'].size
        grid_img = Image.new('RGB', (w * 3, h * 2))
        
        # Adjusted layout for better logical flow if needed, 
        # but sticking to your indexing for consistency:
        grid_img.paste(imgs['front'], (0, 0))    
        grid_img.paste(imgs['right'], (w, 0))     
        grid_img.paste(imgs['back'],  (w*2, 0))   
        grid_img.paste(imgs['left'],  (0, h))     
        grid_img.paste(imgs['top'],   (w, h))     
        grid_img.paste(imgs['bottom'],(w*2, h))   

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Below are 6 views of a 3D medical model labeled as '{label}'. Please provide a very detailed technical description of the geometry, structure, and any visible features or abnormalities suitable for 3D modeling. The following is an example of a description: This 3D model presents a watertight, organic reniform volume, elongated vertically and compressed front-to-back to form a flattened, lenticular profile. The geometry is defined by a convex lateral curve and a deeply concave medial border that recesses into a pronounced renal hilum, exhibiting a smooth, cavernous saddle-point topology. Morphologically asymmetrical, the mesh features a tapered, narrow superior pole and a broader, blunt inferior pole, while the overall surface texture displays the low-frequency undulations and soft 'lumpiness' characteristic of a smoothed, segmented medical scan, avoiding perfectly geometric curvature in favor of natural biological irregularity."}
                ]
            }
        ]

        return {
            "image": grid_img,
            "conversation": conversation,
            "folder_name": folder_name,
            "label": label
        }

def collate_fn(batch, processor):
    images = [item['image'] for item in batch]
    conversations = [item['conversation'] for item in batch]
    
    texts = [processor.apply_chat_template(conv, add_generation_prompt=True) for conv in conversations]
    
    # Process inputs (returns float32 by default)
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    
    return inputs, [x['folder_name'] for x in batch], [x['label'] for x in batch]

def main():
    args = parse_args()
    
    # Set Cache if provided (Critical for TACC scratch)
    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        
    render_dir = os.path.join(args.data_dir, "renders")
    if not os.path.exists(render_dir):
        print(f"Error: Render directory {render_dir} not found.")
        return

    print(f"Loading model: {MODEL_ID}...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, fix_mistral_regex=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer = tokenizer

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    dataset = MedShapeDataset(render_dir)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers,
        collate_fn=lambda x: collate_fn(x, processor)
    )

    print(f"Starting inference on {len(dataset)} items...")
    
    # Use append mode if file exists, or write new
    mode = 'a' if os.path.exists(args.output_file) else 'w'

    with open(args.output_file, mode) as f_out:
        for batch_idx, (inputs, folder_names, labels) in tqdm(enumerate(loader), total=len(loader)):
            
            # Move inputs to device and cast pixel_values to bf16
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs, 
                    max_new_tokens=300, 
                    do_sample=True, 
                    temperature=0.2,
                    top_p=0.9
                )

            generated_texts = processor.batch_decode(output_ids, skip_special_tokens=True)

            for i, full_text in enumerate(generated_texts):
                # Robustly split content
                if "[/INST]" in full_text:
                    description = full_text.split("[/INST]")[-1].strip()
                else:
                    description = full_text

                record = {
                    "id": folder_names[i],
                    "label": labels[i],
                    "description": description
                }
                f_out.write(json.dumps(record) + "\n")
            
            f_out.flush()

    print(f"Done! Saved to {args.output_file}")

if __name__ == "__main__":
    main()