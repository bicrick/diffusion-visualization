#!/usr/bin/env python3
"""
Simple script to download FLUX.1-schnell model
"""

import torch
from diffusers import FluxPipeline

def download_model():
    """Download FLUX.1-schnell model to cache"""
    print("🚀 Downloading FLUX.1-schnell model...")
    print("⚠️  This will download ~24GB of files (one-time only)")
    
    try:
        # This downloads and caches the model
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", 
            torch_dtype=torch.bfloat16
        )
        print("✅ Model downloaded successfully!")
        print("📍 Cached in: ~/.cache/huggingface/hub/")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

if __name__ == "__main__":
    success = download_model()
    exit(0 if success else 1)
