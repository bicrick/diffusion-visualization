# Asset Generation Scripts

This directory contains scripts for generating visualization assets from diffusion models.

## Setup

1. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Hugging Face Authentication** (for gated models like FLUX.1-dev)
   ```bash
   huggingface-cli login
   ```
   Or set environment variable:
   ```bash
   export HF_TOKEN="your_token_here"
   ```

## Usage

### Quick Test Generation

Generate assets for a single prompt (good for testing):

```bash
python scripts/generate_assets.py --prompt "A cat holding a sign that says hello world"
```

### Full Asset Generation

Generate assets for multiple preset prompts:

```bash
python scripts/generate_assets.py
```

### Custom Configuration

```bash
python scripts/generate_assets.py \
  --model "black-forest-labs/FLUX.1-dev" \
  --output "public/data" \
  --steps 50
```

## Output Structure

The script generates the following structure:

```
public/data/
├── all_prompts.json                 # Combined metadata
├── cat_hello_world/
│   ├── metadata.json               # Prompt-specific metadata  
│   ├── final.png                   # Final generated image
│   └── steps/
│       ├── step_000.png           # Initial noise
│       ├── step_005.png           # Early denoising
│       ├── step_010.png           # ...
│       └── step_050.png           # Near-final result
└── abstract_landscape/
    └── ... (same structure)
```

## Generated Assets

### Images
- **step_XXX.png**: Denoising progression images
- **final.png**: Final generated image  
- Format: PNG, 1024x1024 by default

### Metadata (JSON)
- Prompt text and ID
- Generation parameters (steps, guidance, seed)
- Per-step information (timestep, progress)
- Model information

## Default Prompts

The script includes these preset prompts for educational variety:

1. **"A cat holding a sign that says hello world"**
   - Simple, clear subject matter
   - Good for showing basic diffusion progression

2. **"A surreal landscape with floating geometric shapes and vibrant colors"**
   - Complex composition
   - Shows how abstract concepts emerge

3. **"Portrait of a woman with flowing hair, digital art style"**  
   - Human subjects with fine details
   - Demonstrates feature refinement

## Customization

### Adding New Prompts

Edit the `prompts` list in `generate_assets.py`:

```python
prompts = [
    {
        "id": "your_prompt_id",
        "text": "Your prompt text here", 
        "params": {
            "seed": 789,
            "guidance_scale": 4.0,
            "num_inference_steps": 40
        }
    }
]
```

### Model Parameters

- **num_inference_steps**: More steps = smoother progression, larger files
- **guidance_scale**: Higher = closer prompt following
- **seed**: Fixed seed for reproducible results
- **width/height**: Image dimensions (1024x1024 recommended)

## Performance Notes

- **GPU Memory**: FLUX.1-dev requires ~12GB VRAM
- **CPU Offloading**: Enabled by default to reduce VRAM usage
- **Generation Time**: ~30-60 seconds per prompt (50 steps)
- **Storage**: ~50MB per prompt (50 steps + final image)

## Troubleshooting

### CUDA Out of Memory
```bash
# Enable CPU offloading (already enabled by default)
# Or reduce image dimensions:
python scripts/generate_assets.py --width 512 --height 512
```

### Slow Generation
```bash
# Reduce inference steps:
python scripts/generate_assets.py --steps 25
```

### Model Access Issues
```bash
# Make sure you have access to FLUX.1-dev on Hugging Face
# Check your authentication:
huggingface-cli whoami
```

## Next Steps

After generating assets:

1. **Start Web Application**
   ```bash
   npm run dev
   ```

2. **Verify Assets**: Check that images and JSON files are properly generated

3. **Test Loading**: Ensure the web app can load and display the generated assets

The generated assets will be automatically discovered by the web application and made available for interactive visualization.
