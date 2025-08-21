#!/usr/bin/env python3
"""
Simple script to generate images from prompts using FLUX.1-schnell
"""

import torch
from diffusers import FluxPipeline
from pathlib import Path
import argparse

def generate_image(prompt, output_dir="output", filename=None):
    """Generate a single image from a prompt"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"🎨 Generating: '{prompt}'")
    
    try:
        # Load the model (should be cached from download_model.py)
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", 
            torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()  # Save VRAM
        
        # Generate image
        image = pipe(
            prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=4,  # FLUX.1-schnell works well with 4 steps
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(42)
        ).images[0]
        
        # Save image
        if filename is None:
            filename = prompt.replace(" ", "_").replace(",", "")[:50] + ".png"
        
        save_path = output_path / filename
        image.save(save_path)
        
        print(f"✅ Saved: {save_path}")
        return str(save_path)
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate images with FLUX.1-schnell")
    parser.add_argument("prompt", help="Text prompt for image generation")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--filename", help="Output filename (optional)")
    
    args = parser.parse_args()
    
    result = generate_image(args.prompt, args.output, args.filename)
    exit(0 if result else 1)

if __name__ == "__main__":
    main()
