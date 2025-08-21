#!/usr/bin/env python3
"""
Configuration file for diffusion model settings
"""

# Model Configuration
MODEL_CONFIG = {
    # Current model to use
    "current_model": "flux-dev",
    
    # Available models
    "models": {
        "flux-dev": {
            "model_id": "black-forest-labs/FLUX.1-dev",
            "pipeline_class": "FluxPipeline",
            "optimal_steps": 50,
            "max_steps": 100,
            "guidance_scale": 3.5,
            "max_sequence_length": 512,
            "torch_dtype": "bfloat16",
            "description": "FLUX.1-dev, high quality with excellent step progression"
        },
        
        "flux-schnell": {
            "model_id": "black-forest-labs/FLUX.1-schnell",
            "pipeline_class": "FluxPipeline",
            "optimal_steps": 4,
            "max_steps": 20,
            "guidance_scale": 3.5,
            "max_sequence_length": 256,
            "torch_dtype": "bfloat16",
            "description": "Fast FLUX model, 4-20 steps, limited noise progression"
        },
        
        "stable-diffusion-3.5-medium": {
            "model_id": "stabilityai/stable-diffusion-3.5-medium",
            "pipeline_class": "StableDiffusion3Pipeline",
            "optimal_steps": 28,
            "max_steps": 50,
            "guidance_scale": 4.5,
            "max_sequence_length": 77,
            "torch_dtype": "float16",
            "description": "SD3.5 Medium, excellent step progression from noise"
        },
        
        "stable-diffusion-xl": {
            "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "pipeline_class": "DiffusionPipeline",
            "optimal_steps": 30,
            "max_steps": 50,
            "guidance_scale": 7.5,
            "max_sequence_length": 77,
            "torch_dtype": "float16",
            "description": "SDXL Base, great for educational diffusion progression"
        }
    }
}

# Slideshow Configuration
SLIDESHOW_CONFIG = {
    "default_steps": 30,
    "image_size": 1024,
    "seed": 42,
    "save_intermediate": True,
    "create_html_viewer": True
}

def get_current_model_config():
    """Get configuration for the currently selected model"""
    current = MODEL_CONFIG["current_model"]
    if current not in MODEL_CONFIG["models"]:
        raise ValueError(f"Model '{current}' not found in config")
    return MODEL_CONFIG["models"][current]

def get_model_config(model_name):
    """Get configuration for a specific model"""
    if model_name not in MODEL_CONFIG["models"]:
        raise ValueError(f"Model '{model_name}' not found in config")
    return MODEL_CONFIG["models"][model_name]

def list_available_models():
    """List all available models"""
    return list(MODEL_CONFIG["models"].keys())

def set_current_model(model_name):
    """Set the current model to use"""
    if model_name not in MODEL_CONFIG["models"]:
        raise ValueError(f"Model '{model_name}' not available. Choose from: {list_available_models()}")
    MODEL_CONFIG["current_model"] = model_name
    print(f"✅ Switched to model: {model_name}")

if __name__ == "__main__":
    # Print current configuration
    print("🔧 Diffusion Model Configuration")
    print("=" * 40)
    print(f"Current model: {MODEL_CONFIG['current_model']}")
    print()
    
    print("Available models:")
    for name, config in MODEL_CONFIG["models"].items():
        current = "👉 " if name == MODEL_CONFIG["current_model"] else "   "
        print(f"{current}{name}")
        print(f"     {config['description']}")
        print(f"     Model: {config['model_id']}")
        print(f"     Steps: {config['optimal_steps']} (max {config['max_steps']})")
        print()
