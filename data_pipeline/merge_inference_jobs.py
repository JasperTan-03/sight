import glob

def merge_results(pattern="captions_part_*.jsonl", output="captions_final.jsonl"):
    files = sorted(glob.glob(pattern))
    print(f"Merging {len(files)} files...")
    
    with open(output, 'w') as outfile:
        for fname in files:
            with open(fname, 'r') as infile:
                outfile.write(infile.read())
                
    print(f"Done! All data in {output}")

if __name__ == "__main__":
    merge_results()