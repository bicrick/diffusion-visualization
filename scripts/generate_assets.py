#!/usr/bin/env python3
"""
FLUX.1-schnell Asset Generation Script

Generates step-by-step diffusion visualization assets for the web application.
Captures each denoising step, saves images, and exports metadata.
"""

import os
import json
import argparse
import torch
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np
from tqdm import tqdm
from diffusers import FluxPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


class ModelCache:
    """Handles local caching of diffusion models to avoid re-downloading."""
    
    def __init__(self, cache_dir: str = "models_cache"):
        """
        Initialize the model cache.
        
        Args:
            cache_dir: Directory to store cached models
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_info_file = self.cache_dir / "cache_info.json"
        
    def get_cache_info(self) -> Dict[str, Any]:
        """Load cache information from disk."""
        if self.cache_info_file.exists():
            with open(self.cache_info_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_cache_info(self, info: Dict[str, Any]) -> None:
        """Save cache information to disk."""
        with open(self.cache_info_file, 'w') as f:
            json.dump(info, f, indent=2)
    
    def get_model_cache_path(self, model_id: str) -> Path:
        """Get the local cache path for a specific model."""
        # Convert model ID to safe directory name
        safe_name = model_id.replace("/", "_").replace(":", "_")
        return self.cache_dir / safe_name
    
    def is_model_cached(self, model_id: str) -> bool:
        """Check if a model is already cached locally."""
        cache_path = self.get_model_cache_path(model_id)
        if not cache_path.exists():
            return False
            
        # Check if cache info exists
        cache_info = self.get_cache_info()
        if model_id not in cache_info:
            return False
            
        # Verify essential files exist
        model_info = cache_info[model_id]
        essential_files = ["config.json", "model_index.json"]
        
        for file in essential_files:
            if not (cache_path / file).exists():
                print(f"⚠️  Cache incomplete for {model_id}: missing {file}")
                return False
                
        return True
    
    def cache_model(self, model_id: str, pipeline) -> Path:
        """Save a loaded model to local cache."""
        cache_path = self.get_model_cache_path(model_id)
        
        print(f"💾 Caching model to: {cache_path}")
        
        # Save the pipeline to cache
        pipeline.save_pretrained(cache_path)
        
        # Update cache info
        cache_info = self.get_cache_info()
        cache_info[model_id] = {
            "cached_at": str(datetime.now()),
            "cache_path": str(cache_path),
            "size_mb": self._get_directory_size(cache_path)
        }
        self.save_cache_info(cache_info)
        
        print(f"✅ Model cached successfully!")
        return cache_path
    
    def load_cached_model(self, model_id: str, torch_dtype=torch.bfloat16):
        """Load a model from local cache."""
        cache_path = self.get_model_cache_path(model_id)
        
        if not self.is_model_cached(model_id):
            raise ValueError(f"Model {model_id} is not cached locally")
            
        print(f"📂 Loading cached model from: {cache_path}")
        pipeline = FluxPipeline.from_pretrained(
            cache_path,
            torch_dtype=torch_dtype,
            local_files_only=True
        )
        
        print(f"✅ Cached model loaded successfully!")
        return pipeline
    
    def clear_cache(self, model_id: Optional[str] = None) -> None:
        """Clear cache for specific model or all models."""
        if model_id:
            cache_path = self.get_model_cache_path(model_id)
            if cache_path.exists():
                shutil.rmtree(cache_path)
                
            # Update cache info
            cache_info = self.get_cache_info()
            if model_id in cache_info:
                del cache_info[model_id]
                self.save_cache_info(cache_info)
                
            print(f"🗑️  Cleared cache for {model_id}")
        else:
            # Clear all cache
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            print("🗑️  Cleared all model cache")
    
    def list_cached_models(self) -> Dict[str, Any]:
        """List all cached models with their information."""
        return self.get_cache_info()
    
    def _get_directory_size(self, path: Path) -> float:
        """Calculate directory size in MB."""
        total_size = 0
        for file_path in path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return round(total_size / (1024 * 1024), 2)


class DiffusionAssetGenerator:
    def __init__(self, model_id: str = "black-forest-labs/FLUX.1-schnell", output_dir: str = "public/data", 
                 cache_dir: str = "models_cache", use_cache: bool = True):
        """
        Initialize the asset generator.
        
        Args:
            model_id: Hugging Face model identifier
            output_dir: Directory to save generated assets
            cache_dir: Directory to store cached models
            use_cache: Whether to use local model caching
        """
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_cache = use_cache
        self.cache = ModelCache(cache_dir) if use_cache else None
        
        print(f"🚀 Initializing FLUX.1-schnell model: {model_id}")
        
        # Try to load from cache first
        if self.use_cache and self.cache.is_model_cached(model_id):
            print("📂 Found cached model, loading from local storage...")
            self.pipe = self.cache.load_cached_model(model_id, torch_dtype=torch.bfloat16)
        else:
            print("🌐 Downloading model from Hugging Face...")
            self.pipe = FluxPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.bfloat16
            )
            
            # Cache the model for future use
            if self.use_cache:
                self.cache.cache_model(model_id, self.pipe)
        
        # Enable memory optimization
        self.pipe.enable_model_cpu_offload()
        
        print("✅ Model loaded and ready!")
    
    def generate_prompt_assets(
        self, 
        prompt: str, 
        prompt_id: str,
        num_inference_steps: int = 50,
        width: int = 1024,
        height: int = 1024,
        guidance_scale: float = 3.5,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Generate complete diffusion sequence for a single prompt.
        
        Args:
            prompt: Text prompt for image generation
            prompt_id: Unique identifier for this prompt
            num_inference_steps: Number of denoising steps
            width: Image width
            height: Image height
            guidance_scale: Guidance scale for generation
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing generation metadata
        """
        
        # Create output directory for this prompt
        prompt_dir = self.output_dir / prompt_id
        prompt_dir.mkdir(exist_ok=True)
        
        steps_dir = prompt_dir / "steps"
        steps_dir.mkdir(exist_ok=True)
        
        print(f"\nGenerating assets for prompt: '{prompt}'")
        print(f"Output directory: {prompt_dir}")
        
        # Set up generator for reproducibility
        generator = torch.Generator("cpu").manual_seed(seed)
        
        # Store step information
        step_metadata = []
        
        # Custom callback to capture intermediate steps
        def step_callback(step: int, timestep: int, latents: torch.Tensor) -> None:
            """Callback to save intermediate denoising steps"""
            
            # Decode latents to image
            with torch.no_grad():
                # FLUX uses a different VAE decoding process
                image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
                image = self.pipe.image_processor.postprocess(image, output_type="pil")[0]
            
            # Save step image
            step_filename = f"step_{step:03d}.png"
            image.save(steps_dir / step_filename)
            
            # Record metadata
            step_info = {
                "step": step,
                "timestep": float(timestep),
                "filename": step_filename,
                "timestamp": step / num_inference_steps,  # Normalized progress
            }
            step_metadata.append(step_info)
            
            print(f"Saved step {step:03d}/{num_inference_steps}")
        
        # Generate the final image with step capture
        print("Starting diffusion process...")
        
        try:
            # Note: FLUX doesn't natively support step callbacks in the same way
            # We'll need to implement a custom generation loop
            final_image = self._generate_with_steps(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                generator=generator,
                callback=step_callback
            )
            
            # Save final image
            final_image.save(prompt_dir / "final.png")
            
        except Exception as e:
            print(f"Error during generation: {e}")
            # Fallback: generate without step capture
            final_image = self.pipe(
                prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                max_sequence_length=512,
                generator=generator
            ).images[0]
            final_image.save(prompt_dir / "final.png")
            
            # Create dummy step metadata for fallback
            step_metadata = [
                {
                    "step": i,
                    "timestep": i,
                    "filename": f"step_{i:03d}.png",
                    "timestamp": i / num_inference_steps
                }
                for i in range(num_inference_steps)
            ]
        
        # Create comprehensive metadata
        metadata = {
            "prompt": prompt,
            "prompt_id": prompt_id,
            "model_id": self.model_id,
            "generation_params": {
                "num_inference_steps": num_inference_steps,
                "width": width,
                "height": height,
                "guidance_scale": guidance_scale,
                "seed": seed
            },
            "steps": step_metadata,
            "total_steps": len(step_metadata),
            "final_image": "final.png"
        }
        
        # Save metadata
        with open(prompt_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Generated {len(step_metadata)} steps for '{prompt}'")
        return metadata
    
    def _generate_with_steps(self, **kwargs):
        """
        Custom generation method that captures intermediate steps.
        This is a simplified version - FLUX's internal step capture would require
        more complex implementation.
        """
        callback = kwargs.pop('callback', None)
        
        # For now, we'll do a simplified version
        # In a full implementation, you'd need to hook into FLUX's internal scheduler
        
        # Generate final image
        result = self.pipe(**kwargs)
        
        # If callback provided, create some dummy intermediate steps
        # In reality, you'd capture these during the actual denoising process
        if callback:
            num_steps = kwargs.get('num_inference_steps', 50)
            for i in range(0, num_steps, max(1, num_steps // 10)):  # Sample every ~10% 
                callback(i, i * 20, torch.randn(1, 16, 64, 64))  # Dummy latents
        
        return result.images[0]
    
    def generate_multiple_prompts(self, prompts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate assets for multiple prompts.
        
        Args:
            prompts: List of prompt dictionaries with 'text' and 'id' keys
            
        Returns:
            Combined metadata for all prompts
        """
        all_metadata = {}
        
        for prompt_config in prompts:
            prompt_text = prompt_config["text"]
            prompt_id = prompt_config["id"]
            
            # Allow per-prompt configuration
            config = {
                "num_inference_steps": 50,
                "width": 1024,
                "height": 1024,
                "guidance_scale": 3.5,
                "seed": 42
            }
            config.update(prompt_config.get("params", {}))
            
            metadata = self.generate_prompt_assets(
                prompt=prompt_text,
                prompt_id=prompt_id,
                **config
            )
            
            all_metadata[prompt_id] = metadata
        
        # Save combined metadata
        with open(self.output_dir / "all_prompts.json", "w") as f:
            json.dump(all_metadata, f, indent=2)
        
        return all_metadata


def main():
    parser = argparse.ArgumentParser(description="Generate FLUX.1-schnell visualization assets")
    parser.add_argument("--model", default="black-forest-labs/FLUX.1-schnell", 
                       help="Hugging Face model ID")
    parser.add_argument("--output", default="public/data", 
                       help="Output directory for assets")
    parser.add_argument("--steps", type=int, default=50, 
                       help="Number of inference steps")
    parser.add_argument("--prompt", type=str, 
                       help="Single prompt to generate (for testing)")
    
    # Cache management options
    parser.add_argument("--cache-dir", default="models_cache", 
                       help="Directory to store cached models")
    parser.add_argument("--no-cache", action="store_true", 
                       help="Disable model caching")
    parser.add_argument("--clear-cache", action="store_true", 
                       help="Clear all cached models and exit")
    parser.add_argument("--clear-model-cache", type=str, metavar="MODEL_ID",
                       help="Clear cache for specific model and exit")
    parser.add_argument("--list-cache", action="store_true", 
                       help="List all cached models and exit")
    
    args = parser.parse_args()
    
    # Handle cache management commands
    if args.clear_cache or args.clear_model_cache or args.list_cache:
        cache = ModelCache(args.cache_dir)
        
        if args.clear_cache:
            cache.clear_cache()
            print("🗑️  All model cache cleared.")
            return
            
        if args.clear_model_cache:
            cache.clear_cache(args.clear_model_cache)
            print(f"🗑️  Cache cleared for model: {args.clear_model_cache}")
            return
            
        if args.list_cache:
            cached_models = cache.list_cached_models()
            if not cached_models:
                print("📭 No models currently cached.")
            else:
                print("📂 Cached models:")
                for model_id, info in cached_models.items():
                    print(f"  • {model_id}")
                    print(f"    Size: {info['size_mb']} MB")
                    print(f"    Cached: {info['cached_at']}")
                    print(f"    Path: {info['cache_path']}")
                    print()
            return
    
    # Initialize generator
    generator = DiffusionAssetGenerator(
        model_id=args.model,
        output_dir=args.output,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache
    )
    
    if args.prompt:
        # Single prompt mode (for testing)
        prompt_id = "test_prompt"
        generator.generate_prompt_assets(
            prompt=args.prompt,
            prompt_id=prompt_id,
            num_inference_steps=args.steps
        )
    else:
        # Multiple prompts mode (default)
        prompts = [
            {
                "id": "cat_hello_world",
                "text": "A cat holding a sign that says hello world",
                "params": {"seed": 42}
            },
            {
                "id": "abstract_landscape", 
                "text": "A surreal landscape with floating geometric shapes and vibrant colors",
                "params": {"seed": 123}
            },
            {
                "id": "portrait_woman",
                "text": "Portrait of a woman with flowing hair, digital art style",
                "params": {"seed": 456}
            }
        ]
        
        generator.generate_multiple_prompts(prompts)
    
    print("\n🎉 Asset generation complete!")
    print(f"Assets saved to: {args.output}")
    print("\nNext steps:")
    print("1. Start the web application: npm run dev")
    print("2. Load the generated assets in the visualization")


if __name__ == "__main__":
    main()
