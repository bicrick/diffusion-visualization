#!/usr/bin/env python3
"""
Quick test script for FLUX.1-schnell to verify everything works
"""

import torch
from diffusers import FluxPipeline
from pathlib import Path
import os

def quick_test():
    """Quick test of FLUX.1-schnell with minimal setup"""
    
    print("🚀 FLUX.1-schnell Quick Test")
    print("=" * 50)
    
    # Create simple output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"📁 Output directory: {output_dir.absolute()}")
    
    try:
        print("\n🌐 Loading FLUX.1-schnell...")
        print("⚠️  Note: First run will download ~24GB of model files")
        print("   This may take 10-30 minutes depending on your internet speed")
        
        # Load the pipeline
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", 
            torch_dtype=torch.bfloat16
        )
        
        # Enable CPU offloading to save VRAM
        pipe.enable_model_cpu_offload()
        
        print("✅ Model loaded successfully!")
        
        # Simple test prompt
        prompt = "A cute robot holding a sign that says 'Hello AI!'"
        
        print(f"\n🎨 Generating image with prompt: '{prompt}'")
        print("⏱️  This should take ~10-20 seconds...")
        
        # Generate image
        image = pipe(
            prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=4,  # FLUX.1-schnell works well with fewer steps
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(42)
        ).images[0]
        
        # Save the image
        output_path = output_dir / "test_image.png"
        image.save(output_path)
        
        print(f"✅ Image saved to: {output_path.absolute()}")
        print("\n🎉 Success! FLUX.1-schnell is working correctly.")
        print("\nNext steps:")
        print("1. Run the full asset generation: python scripts/generate_assets.py")
        print("2. Check the generated image quality")
        print("3. Proceed with web application setup")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("- Ensure you have enough disk space (~25GB)")
        print("- Check your internet connection")
        print("- Try running again (downloads can resume)")
        return False

if __name__ == "__main__":
    success = quick_test()
    exit(0 if success else 1)
