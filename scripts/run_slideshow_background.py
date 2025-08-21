#!/usr/bin/env python3
"""
Wrapper script to run slideshow generation in background with caffeinate
"""

import subprocess
import sys
import argparse
from pathlib import Path

def run_caffeinated_slideshow(prompt, steps=20, output="slideshow_output"):
    """Run slideshow generation with caffeinate to prevent sleep"""
    
    print(f"🚀 Starting background slideshow generation...")
    print(f"💡 Your Mac will stay awake during generation")
    print(f"🔒 You can lock your screen - the process will continue")
    print(f"📱 Prompt: '{prompt}'")
    print(f"⏱️  Steps: {steps}")
    print(f"📁 Output: {output}")
    print("")
    
    # Build the command
    script_path = Path(__file__).parent / "generate_image_slideshow.py"
    
    cmd = [
        "caffeinate",     # Prevent system sleep
        "-i",             # Prevent idle sleep
        "-d",             # Prevent display sleep  
        "python", 
        str(script_path),
        prompt,
        "--steps", str(steps),
        "--output", output
    ]
    
    print("🔧 Running command:")
    print(" ".join(cmd))
    print("")
    print("⚠️  Press Ctrl+C to cancel if needed")
    print("📊 Watch for progress updates below...")
    print("=" * 60)
    
    try:
        # Run the command
        result = subprocess.run(cmd, check=True)
        
        print("=" * 60)
        print("🎉 Slideshow generation completed successfully!")
        print(f"📁 Check your results in: {output}/")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Generation failed with exit code {e.returncode}")
        return False
        
    except KeyboardInterrupt:
        print("\n🛑 Generation cancelled by user")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run slideshow generation in background")
    parser.add_argument("prompt", help="Text prompt for image generation")
    parser.add_argument("--steps", type=int, default=20, help="Number of steps to capture")
    parser.add_argument("--output", default="slideshow_output", help="Output directory")
    
    args = parser.parse_args()
    
    success = run_caffeinated_slideshow(args.prompt, args.steps, args.output)
    exit(0 if success else 1)

if __name__ == "__main__":
    main()
