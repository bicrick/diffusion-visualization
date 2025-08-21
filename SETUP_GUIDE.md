# Quick Setup Guide

This guide will get you running with FLUX.1-schnell asset generation quickly.

## Step 1: Environment Setup ✅ 

Already completed! You have:
- Python virtual environment: `venv/`
- All dependencies installed
- FLUX.1-schnell configured

## Step 2: Quick Test

Before generating full assets, let's test that everything works:

```bash
# Activate environment
source venv/bin/activate

# Run quick test (this will download the model)
python scripts/quick_test.py
```

**⚠️ Important:** The first run downloads ~24GB of model files. This takes 10-30 minutes depending on your internet speed.

## Step 3: Generate Assets

Once the quick test works:

```bash
# Single prompt test
python scripts/generate_assets.py --prompt "A cat holding a sign that says hello world"

# Full preset generation
python scripts/generate_assets.py
```

## What happens during download:

The model consists of several large files:
- **Transformer**: ~15GB (main diffusion model)  
- **Text Encoders**: ~5GB (understands prompts)
- **VAE**: ~200MB (converts to/from images)
- **Tokenizers**: ~10MB (processes text)

Files are cached in `~/.cache/huggingface/hub/` so you only download once.

## Expected Timeline:

1. **Download**: 10-30 minutes (first time only)
2. **Quick Test**: 30 seconds  
3. **Single Asset Generation**: 2-3 minutes
4. **Full Asset Set (3 prompts)**: 5-8 minutes

## Troubleshooting:

### Slow Download
```bash
# Check available space
df -h

# Resume interrupted download
python scripts/quick_test.py  # Just run again
```

### Memory Issues
```bash
# Reduce image size for testing
python scripts/generate_assets.py --prompt "test" --width 512 --height 512
```

### CUDA Out of Memory
The script automatically enables CPU offloading, but if you still get errors:
```bash
# Force CPU-only mode (slower but uses less VRAM)
export CUDA_VISIBLE_DEVICES=""
python scripts/quick_test.py
```

## Next Steps:

After successful asset generation:
1. ✅ Verify images in `public/data/` directory
2. 🌐 Set up the web application (React + Three.js)
3. 🎨 Build the 3D visualization interface

## File Structure After Setup:

```
diffusion-visualization/
├── venv/                          # Python environment
├── scripts/
│   ├── generate_assets.py         # Main generator
│   ├── quick_test.py              # Test script
│   └── README.md                  # Detailed docs
├── public/data/                   # Generated assets
│   ├── test_prompt/
│   │   ├── steps/                 # step_000.png → step_050.png
│   │   ├── metadata.json          # Generation info
│   │   └── final.png              # Final result
│   └── all_prompts.json          # Combined metadata
└── test_output/                   # Quick test results
```

Ready to proceed? Run `python scripts/quick_test.py` to get started! 🚀
