import os
import re
import random
import argparse
import urllib.parse
import requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="MedShapeNet Downloader & Sampler")
    parser.add_argument("--input_file", type=str, default="MedShapeNetDataset.txt", help="Path to master URL list")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory to save downloaded models")
    parser.add_argument("--target_percentage", type=float, default=0.20, help="Fraction of dataset to download (0.0 - 1.0)")
    parser.add_argument("--workers", type=int, default=16, help="Number of download threads")
    return parser.parse_args()

def get_filename_from_url(url):
    """Extracts the actual filename (e.g., '000000_tumoredbrain.stl') from the URL."""
    parsed = urllib.parse.urlparse(url.strip())
    query_params = urllib.parse.parse_qs(parsed.query)
    if 'files' in query_params:
        return query_params['files'][0]
    return parsed.path.split('/')[-1]

def get_class_from_filename(filename):
    """Extracts cleaner class names from the filename."""
    # 1. Remove extension
    name_clean = filename.split('.')[0]
    # 2. Remove Patient/Scan IDs (Prefixes)
    name_clean = re.sub(r'^(?:s\d+|\d+)_', '', name_clean)
    # 3. Remove Laterality (Suffixes)
    name_clean = re.sub(r'_(?:left|right|l|r)$', '', name_clean, flags=re.IGNORECASE)
    return name_clean

def solve_for_cap(counts, target_total):
    """Finds the integer 'cap' such that sum(min(n, cap)) ~= target_total."""
    low = 0
    high = max(counts)
    best_cap = high
    
    while low <= high:
        mid = (low + high) // 2
        current_sum = sum(min(n, mid) for n in counts)
        if current_sum < target_total:
            low = mid + 1
        else:
            best_cap = mid
            high = mid - 1
    return best_cap

def download_file(url, save_path):
    """Downloads a single file."""
    if os.path.exists(save_path):
        return "SKIPPED" # Simple caching
    try:
        r = requests.get(url.strip(), stream=True)
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return "SUCCESS"
    except Exception:
        return "ERROR"

def main():
    args = parse_args()
    
    # 1. Setup Directories
    raw_dir = os.path.join(args.data_dir, "raw_models")
    os.makedirs(raw_dir, exist_ok=True)
    
    print(f"Reading {args.input_file}...")
    class_map = defaultdict(list)
    
    # 2. Read and Classify
    try:
        with open(args.input_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                fname = get_filename_from_url(line)
                cls_name = get_class_from_filename(fname)
                
                # Store tuple of (url, filename)
                class_map[cls_name].append((line, fname))
    except FileNotFoundError:
        print(f"Error: Could not find {args.input_file}")
        return

    total_files = sum(len(v) for v in class_map.values())
    print(f"Found {total_files} files across {len(class_map)} classes.")
    
    # 3. Calculate Sampling
    target_count = int(total_files * args.target_percentage)
    counts = [len(v) for v in class_map.values()]
    cap = solve_for_cap(counts, target_count)
    
    print(f"Targeting {target_count} files (Cap: {cap} per class).")

    # 4. Select Files
    download_queue = []
    random.seed(42)
    
    for cls_name, items in class_map.items():
        if len(items) <= cap:
            download_queue.extend(items)
        else:
            download_queue.extend(random.sample(items, cap))
            
    random.shuffle(download_queue)
    print(f"Queued {len(download_queue)} files for download.")

    # 5. Execute Downloads
    # Using ThreadPoolExecutor because downloads are I/O bound
    print(f"Downloading to {raw_dir} with {args.workers} workers...")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for url, fname in download_queue:
            save_path = os.path.join(raw_dir, fname)
            futures.append(executor.submit(download_file, url, save_path))
        
        # Track progress
        results = {"SUCCESS": 0, "SKIPPED": 0, "ERROR": 0}
        for f in tqdm(futures, total=len(futures), unit="file"):
            res = f.result()
            results[res] += 1

    print("\nDownload Complete.")
    print(results)

if __name__ == "__main__":
    main()