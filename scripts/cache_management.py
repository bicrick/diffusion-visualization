#!/usr/bin/env python3
"""
Model Cache Management Utility

Standalone script for managing cached diffusion models.
"""

import argparse
import sys
from pathlib import Path

# Add the scripts directory to the path so we can import our classes
sys.path.insert(0, str(Path(__file__).parent))

from generate_assets import ModelCache


def main():
    parser = argparse.ArgumentParser(description="Manage cached diffusion models")
    parser.add_argument("--cache-dir", default="models_cache", 
                       help="Directory where models are cached")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all cached models")
    
    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear cached models")
    clear_parser.add_argument("model_id", nargs="?", 
                             help="Specific model to clear (optional)")
    clear_parser.add_argument("--all", action="store_true", 
                             help="Clear all cached models")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show cache directory info")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Pre-download and cache a model")
    download_parser.add_argument("model_id", help="Hugging Face model ID to download")
    download_parser.add_argument("--force", action="store_true", 
                                help="Re-download even if already cached")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cache = ModelCache(args.cache_dir)
    
    if args.command == "list":
        cached_models = cache.list_cached_models()
        if not cached_models:
            print("📭 No models currently cached.")
        else:
            print("📂 Cached models:")
            total_size = 0
            for model_id, info in cached_models.items():
                print(f"  • {model_id}")
                print(f"    Size: {info['size_mb']} MB")
                print(f"    Cached: {info['cached_at']}")
                print(f"    Path: {info['cache_path']}")
                total_size += info['size_mb']
                print()
            print(f"Total cache size: {total_size:.1f} MB")
    
    elif args.command == "clear":
        if args.all:
            cache.clear_cache()
            print("🗑️  All model cache cleared.")
        elif args.model_id:
            cache.clear_cache(args.model_id)
            print(f"🗑️  Cache cleared for model: {args.model_id}")
        else:
            print("❌ Specify --all or provide a model_id to clear")
    
    elif args.command == "info":
        cache_dir = Path(args.cache_dir)
        if cache_dir.exists():
            cached_models = cache.list_cached_models()
            print(f"📂 Cache directory: {cache_dir.absolute()}")
            print(f"📊 Number of cached models: {len(cached_models)}")
            
            if cached_models:
                total_size = sum(info['size_mb'] for info in cached_models.values())
                print(f"💾 Total cache size: {total_size:.1f} MB")
            else:
                print("💾 Cache is empty")
        else:
            print(f"❌ Cache directory does not exist: {cache_dir}")
    
    elif args.command == "download":
        if cache.is_model_cached(args.model_id) and not args.force:
            print(f"✅ Model {args.model_id} is already cached.")
            print("Use --force to re-download.")
        else:
            print(f"🌐 Downloading and caching model: {args.model_id}")
            try:
                from diffusers import FluxPipeline
                import torch
                
                pipeline = FluxPipeline.from_pretrained(
                    args.model_id,
                    torch_dtype=torch.bfloat16
                )
                
                cache_path = cache.cache_model(args.model_id, pipeline)
                print(f"✅ Model cached successfully at: {cache_path}")
                
            except Exception as e:
                print(f"❌ Failed to download model: {e}")


if __name__ == "__main__":
    main()
