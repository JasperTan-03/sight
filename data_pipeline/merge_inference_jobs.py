import glob
import json
from pathlib import Path


def merge_results(checkpoint_dir="checkpoints", output="captions_final.jsonl"):
    """Merge all rank checkpoint files into a single output file."""
    pattern = str(Path(checkpoint_dir) / "rank*.jsonl")
    files = sorted(glob.glob(pattern))
    print(f"Merging {len(files)} files from {checkpoint_dir}...")

    seen_ids = set()
    total_records = 0

    with open(output, "w") as outfile:
        for fname in files:
            file_count = 0
            with open(fname, "r") as infile:
                for line in infile:
                    try:
                        record = json.loads(line.strip())
                        # Deduplicate by ID
                        if record["id"] not in seen_ids:
                            seen_ids.add(record["id"])
                            outfile.write(json.dumps(record) + "\n")
                            file_count += 1
                    except json.JSONDecodeError:
                        continue
            print(f"  - Loaded {file_count} records from {Path(fname).name}")
            total_records += file_count

    print(f"Done! Merged {total_records} records into {output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge checkpoint files")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory containing rank*.jsonl files")
    parser.add_argument("--output", type=str, default="captions_final.jsonl", help="Output file path")
    args = parser.parse_args()

    merge_results(args.checkpoint_dir, args.output)
