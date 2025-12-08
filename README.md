# SIGHT (Shap-E Improved via Generative Health Text)
SIGHT enhances medical text-to-3D generation by fine-tuning Shap-E with VLM-synthesized captions. By replacing sparse labels with dense anatomical descriptions, SIGHT produces higher-fidelity 3D organs and precise pathological structures compared to standard baselines.

## Setup

### For TACC Lonestar6

**Recommended: Use mamba (faster and avoids solver issues)**

1. Load necessary modules:
```bash
module load cuda/12.0    # or cuda/11.4, cuda/12.8 depending on your needs
module load gcc          # optional but recommended
```

2. Create environment using mamba:
```bash
mamba env create -f environment.yml
conda activate sight
```

**Alternative: Manual setup (if mamba env create has issues)**
```bash
module load cuda/12.0
module load gcc
mamba create -n sight python=3.10
conda activate sight
pip install -e .
```

**Quick setup script:**
```bash
chmod +x setup_tacc.sh
./setup_tacc.sh
```

### For Local Machines

```bash
conda env create -f environment.yml
conda activate sight
```

Or if conda has solver issues:
```bash
conda create -n sight python=3.10
conda activate sight
pip install -e .
```