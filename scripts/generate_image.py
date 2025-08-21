#!/usr/bin/env python3
"""
Generate single images from prompts using the configured diffusion model
"""

import torch
from diffusers import FluxPipeline, StableDiffusion3Pipeline, DiffusionPipeline
from pathlib import Path
import argparse
from config import get_current_model_config, get_model_config, list_available_models

def generate_image(prompt, output_dir="output", filename=None, model_name=None):
    """Generate a single image from a prompt"""
    
    # Get model configuration
    if model_name:
        model_config = get_model_config(model_name)
    else:
        model_config = get_current_model_config()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"🎨 Generating: '{prompt}'")
    print(f"📦 Using: {model_config['model_id']}")
    
    try:
        # Load the appropriate pipeline
        pipeline_class = globals()[model_config['pipeline_class']]
        torch_dtype = getattr(torch, model_config['torch_dtype'])
        
        pipe = pipeline_class.from_pretrained(
            model_config['model_id'],
            torch_dtype=torch_dtype
        )
        pipe.enable_model_cpu_offload()  # Save VRAM
        
        # Prepare generation parameters
        generation_kwargs = {
            "prompt": prompt,
            "height": 1024,
            "width": 1024,
            "guidance_scale": model_config['guidance_scale'],
            "num_inference_steps": model_config['optimal_steps'],
            "generator": torch.Generator("cpu").manual_seed(42)
        }
        
        # Add model-specific parameters
        if 'max_sequence_length' in model_config:
            generation_kwargs['max_sequence_length'] = model_config['max_sequence_length']
        
        # Generate image
        image = pipe(**generation_kwargs).images[0]
        
        # Save image
        if filename is None:
            filename = prompt.replace(" ", "_").replace(",", "")[:50] + ".png"
        
        save_path = output_path / filename
        image.save(save_path)
        
        print(f"✅ Saved: {save_path}")
        print(f"🎯 Used {model_config['optimal_steps']} inference steps")
        return str(save_path)
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        print("💡 Make sure the model is downloaded first:")
        print(f"   python scripts/download_model.py --model {model_name or 'current'}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate images with diffusion models")
    parser.add_argument("prompt", nargs='?', help="Text prompt for image generation")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--filename", help="Output filename (optional)")
    parser.add_argument("--model", choices=list_available_models(), 
                       help="Model to use (default: from config)")
    parser.add_argument("--list-models", action="store_true", 
                       help="List available models and exit")
    
    args = parser.parse_args()
    
    if args.list_models:
        from config import MODEL_CONFIG
        print("🔧 Available Models:")
        print("=" * 40)
        current_model = MODEL_CONFIG["current_model"]
        
        for name, config in MODEL_CONFIG["models"].items():
            current_marker = " 👉 [CURRENT]" if name == current_model else ""
            print(f"📦 {name}{current_marker}")
            print(f"   {config['description']}")
            print(f"   Steps: {config['optimal_steps']}")
            print()
        
        print("Usage examples:")
        print(f"  python scripts/generate_image.py 'a cat in space'")
        print(f"  python scripts/generate_image.py 'a robot' --model stable-diffusion-3.5-medium")
        return
    
    if not args.prompt:
        parser.error("Prompt is required when not using --list-models")
    
    result = generate_image(args.prompt, args.output, args.filename, args.model)
    exit(0 if result else 1)

if __name__ == "__main__":
    main()
