#!/usr/bin/env python3
"""
Download diffusion models using the configuration system
"""

import torch
import argparse
from diffusers import FluxPipeline, StableDiffusion3Pipeline, DiffusionPipeline
from config import get_current_model_config, get_model_config, list_available_models

def download_model(model_name=None):
    """Download the specified model (or current model from config)"""
    
    # Get model configuration
    if model_name:
        model_config = get_model_config(model_name)
    else:
        model_config = get_current_model_config()
    
    print(f"🚀 Downloading {model_config['model_id']}...")
    print(f"📝 {model_config['description']}")
    
    # Estimate download size based on model
    if "flux" in model_config['model_id'].lower():
        size_estimate = "~24GB"
    elif "stable-diffusion-3" in model_config['model_id'].lower():
        size_estimate = "~8GB"
    elif "stable-diffusion-xl" in model_config['model_id'].lower():
        size_estimate = "~7GB"
    else:
        size_estimate = "~5-10GB"
    
    print(f"⚠️  This will download {size_estimate} of files (one-time only)")
    print("📍 Files will be cached in: ~/.cache/huggingface/hub/")
    print("")
    
    try:
        # Get the appropriate pipeline class
        pipeline_class = globals()[model_config['pipeline_class']]
        torch_dtype = getattr(torch, model_config['torch_dtype'])
        
        print("🌐 Starting download...")
        
        # This downloads and caches the model
        pipe = pipeline_class.from_pretrained(
            model_config['model_id'],
            torch_dtype=torch_dtype
        )
        
        print("✅ Model downloaded successfully!")
        print(f"📦 Model: {model_config['model_id']}")
        print(f"🎯 Optimal steps: {model_config['optimal_steps']}")
        print(f"📍 Cached in: ~/.cache/huggingface/hub/")
        print("")
        print("🚀 Ready to generate images!")
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("")
        print("💡 Troubleshooting:")
        print("- Check your internet connection")
        print("- Ensure you have enough disk space")
        print("- For gated models, run: huggingface-cli login")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download diffusion models")
    parser.add_argument("--model", choices=list_available_models(), 
                       help="Model to download (default: from config)")
    parser.add_argument("--list-models", action="store_true", 
                       help="List available models and exit")
    
    args = parser.parse_args()
    
    if args.list_models:
        from config import MODEL_CONFIG
        print("🔧 Available Models to Download:")
        print("=" * 50)
        current_model = MODEL_CONFIG["current_model"]
        
        for name, config in MODEL_CONFIG["models"].items():
            current_marker = " 👉 [CURRENT]" if name == current_model else ""
            print(f"📦 {name}{current_marker}")
            print(f"   {config['description']}")
            print(f"   Model: {config['model_id']}")
            print(f"   Optimal steps: {config['optimal_steps']}")
            
            # Size estimates
            if "flux" in config['model_id'].lower():
                size = "~24GB"
            elif "stable-diffusion-3" in config['model_id'].lower():
                size = "~8GB"  
            elif "stable-diffusion-xl" in config['model_id'].lower():
                size = "~7GB"
            else:
                size = "~5-10GB"
            print(f"   Download size: {size}")
            print()
        
        print("Usage examples:")
        print(f"  python scripts/download_model.py                    # Download current model ({current_model})")
        print(f"  python scripts/download_model.py --model flux-schnell")
        print(f"  python scripts/download_model.py --model stable-diffusion-3.5-medium")
        return
    
    success = download_model(args.model)
    exit(0 if success else 1)

if __name__ == "__main__":
    main()
