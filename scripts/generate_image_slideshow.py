#!/usr/bin/env python3
"""
Generate a slideshow of diffusion steps showing how an image emerges from noise
"""

import torch
from diffusers import FluxPipeline, StableDiffusion3Pipeline, DiffusionPipeline
from pathlib import Path
import argparse
import json
from datetime import datetime
import re
from config import get_current_model_config, get_model_config, list_available_models

class DiffusionSlideshow:
    def __init__(self, num_steps=20, model_name=None):
        """Initialize the slideshow generator"""
        self.num_steps = num_steps
        self.step_images = []
        self.step_metadata = []
        
        # Get model configuration
        if model_name:
            self.model_config = get_model_config(model_name)
        else:
            self.model_config = get_current_model_config()
        
        print(f"🎬 Loading {self.model_config['model_id']} for {num_steps}-step slideshow...")
        print(f"📝 {self.model_config['description']}")
        
        # Load the appropriate pipeline
        pipeline_class = globals()[self.model_config['pipeline_class']]
        torch_dtype = getattr(torch, self.model_config['torch_dtype'])
        
        self.pipe = pipeline_class.from_pretrained(
            self.model_config['model_id'],
            torch_dtype=torch_dtype
        )
        self.pipe.enable_model_cpu_offload()
        
    def step_callback(self, step, timestep, latents):
        """Callback function to capture each diffusion step"""
        print(f"  📸 Capturing step {step + 1}/{self.num_steps}")
        
        # Decode latents to image
        with torch.no_grad():
            # Scale latents for VAE
            latents_scaled = latents / self.pipe.vae.config.scaling_factor
            
            # Decode to image
            image = self.pipe.vae.decode(latents_scaled, return_dict=False)[0]
            image = self.pipe.image_processor.postprocess(image, output_type="pil")[0]
            
            # Store the image and metadata
            self.step_images.append(image)
            self.step_metadata.append({
                "step": step + 1,
                "timestep": float(timestep),
                "progress": (step + 1) / self.num_steps,
                "noise_level": float(timestep) / 1000.0  # Approximate noise level
            })
    
    def generate_slideshow(self, prompt, output_dir="slideshow_output"):
        """Generate a complete slideshow from prompt"""
        
        # Create output directory
        prompt_safe = re.sub(r'[^\w\s-]', '', prompt).strip()
        prompt_safe = re.sub(r'[-\s]+', '_', prompt_safe)[:50]
        
        output_path = Path(output_dir) / f"prompt_{prompt_safe}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🎨 Generating slideshow for: '{prompt}'")
        print(f"📁 Output: {output_path}")
        
        # Reset storage
        self.step_images = []
        self.step_metadata = []
        
        # Generate with step capture
        print(f"🚀 Starting {self.num_steps}-step generation...")
        
        # Note: FLUX doesn't directly support step callbacks, so we'll modify the approach
        # We'll run multiple generations with different numbers of steps to simulate progression
        
        generator = torch.Generator("cpu").manual_seed(42)
        
        # Generate images with increasing steps to show progression
        # Use model-specific step progression
        optimal_steps = self.model_config['optimal_steps']
        max_steps = min(self.num_steps, self.model_config['max_steps'])
        
        if max_steps <= 10:
            step_counts = [1, 2, 3, 4, 6, 8, max_steps]
        elif max_steps <= 20:
            step_counts = [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, max_steps]
        else:
            step_counts = [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 20, 25, 30, optimal_steps, max_steps]
        
        # Remove duplicates and sort
        step_counts = sorted(list(set([s for s in step_counts if s <= max_steps])))
        
        for i, steps in enumerate(step_counts):
            if steps > self.num_steps:
                break
                
            print(f"  🎬 Generating step {i + 1}/{len(step_counts)} (using {steps} inference steps)")
            
            # Generate image with specific number of steps using model config
            generation_kwargs = {
                "prompt": prompt,
                "height": 1024,
                "width": 1024,
                "guidance_scale": self.model_config['guidance_scale'],
                "num_inference_steps": steps,
                "generator": torch.Generator("cpu").manual_seed(42)  # Same seed for consistency
            }
            
            # Add model-specific parameters
            if 'max_sequence_length' in self.model_config:
                generation_kwargs['max_sequence_length'] = self.model_config['max_sequence_length']
            
            image = self.pipe(**generation_kwargs).images[0]
            
            # Save image immediately
            filename = f"step_{i + 1:03d}.png"
            image.save(output_path / filename)
            print(f"    ✅ Saved {filename}")
            
            # Store the image and metadata
            self.step_images.append(image)
            self.step_metadata.append({
                "step": i + 1,
                "inference_steps": steps,
                "progress": (i + 1) / len(step_counts),
                "description": f"Generated with {steps} inference steps"
            })
            
            # Save incremental metadata
            temp_metadata = {
                "prompt": prompt,
                "model": "black-forest-labs/FLUX.1-schnell",
                "total_steps_planned": len(step_counts),
                "steps_completed": i + 1,
                "created_at": datetime.now().isoformat(),
                "steps": self.step_metadata.copy()
            }
            
            with open(output_path / "metadata_progress.json", "w") as f:
                json.dump(temp_metadata, f, indent=2)
            
            print(f"    📊 Progress: {i + 1}/{len(step_counts)} steps completed")
        
        # Step images already saved iteratively above
        print(f"💾 All {len(self.step_images)} step images saved during generation")
        
        # Save final high-quality image
        print("🎯 Generating final high-quality image...")
        final_kwargs = {
            "prompt": prompt,
            "height": 1024,
            "width": 1024,
            "guidance_scale": self.model_config['guidance_scale'],
            "num_inference_steps": self.model_config['optimal_steps'],
            "generator": torch.Generator("cpu").manual_seed(42)
        }
        
        if 'max_sequence_length' in self.model_config:
            final_kwargs['max_sequence_length'] = self.model_config['max_sequence_length']
        
        final_image = self.pipe(**final_kwargs).images[0]
        
        final_image.save(output_path / "final.png")
        print("  ✅ Saved final.png")
        
        # Save metadata
        slideshow_metadata = {
            "prompt": prompt,
            "model": self.model_config['model_id'],
            "model_config": self.model_config,
            "total_steps": len(self.step_images),
            "created_at": datetime.now().isoformat(),
            "steps": self.step_metadata
        }
        
        with open(output_path / "metadata.json", "w") as f:
            json.dump(slideshow_metadata, f, indent=2)
        
        print(f"✅ Slideshow complete! {len(self.step_images)} images saved to {output_path}")
        
        # Create simple HTML viewer
        self.create_html_viewer(output_path, prompt, slideshow_metadata)
        
        return output_path
    
    def create_html_viewer(self, output_path, prompt, metadata):
        """Create a simple HTML slideshow viewer"""
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diffusion Slideshow: {prompt}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f0f0f0;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .slideshow-container {{
            position: relative;
            max-width: 800px;
            margin: auto;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .slide {{
            display: none;
            text-align: center;
            padding: 20px;
        }}
        .slide.active {{
            display: block;
        }}
        .slide img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .controls {{
            text-align: center;
            margin: 20px 0;
        }}
        .btn {{
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 0 5px;
            border-radius: 5px;
            cursor: pointer;
        }}
        .btn:hover {{
            background: #0056b3;
        }}
        .progress {{
            width: 100%;
            height: 10px;
            background: #e0e0e0;
            border-radius: 5px;
            overflow: hidden;
            margin: 20px 0;
        }}
        .progress-bar {{
            height: 100%;
            background: #007bff;
            transition: width 0.3s ease;
        }}
        .step-info {{
            margin: 15px 0;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎬 Diffusion Slideshow</h1>
        <h2>"{prompt}"</h2>
        <p>Watch how FLUX.1-schnell builds this image step by step</p>
    </div>
    
    <div class="slideshow-container">
"""
        
        # Add slides
        for i, step in enumerate(metadata['steps']):
            active_class = "active" if i == 0 else ""
            html_content += f"""
        <div class="slide {active_class}">
            <img src="step_{i + 1:03d}.png" alt="Step {i + 1}">
            <div class="step-info">
                <strong>Step {i + 1}/{len(metadata['steps'])}</strong><br>
                {step.get('description', f"Inference steps: {step.get('inference_steps', 'N/A')}")}
            </div>
        </div>"""
        
        # Add final image
        html_content += f"""
        <div class="slide">
            <img src="final.png" alt="Final Result">
            <div class="step-info">
                <strong>Final Result</strong><br>
                High-quality final generation
            </div>
        </div>
    </div>
    
    <div class="progress">
        <div class="progress-bar" id="progressBar"></div>
    </div>
    
    <div class="controls">
        <button class="btn" onclick="previousSlide()">⏮️ Previous</button>
        <button class="btn" onclick="toggleAutoplay()" id="playBtn">▶️ Play</button>
        <button class="btn" onclick="nextSlide()">⏭️ Next</button>
    </div>
    
    <script>
        let currentSlide = 0;
        let autoplay = false;
        let autoplayInterval;
        const totalSlides = {len(metadata['steps']) + 1};
        
        function showSlide(n) {{
            const slides = document.querySelectorAll('.slide');
            if (n >= totalSlides) currentSlide = 0;
            if (n < 0) currentSlide = totalSlides - 1;
            
            slides.forEach(slide => slide.classList.remove('active'));
            slides[currentSlide].classList.add('active');
            
            // Update progress bar
            const progress = (currentSlide / (totalSlides - 1)) * 100;
            document.getElementById('progressBar').style.width = progress + '%';
        }}
        
        function nextSlide() {{
            currentSlide++;
            showSlide(currentSlide);
        }}
        
        function previousSlide() {{
            currentSlide--;
            showSlide(currentSlide);
        }}
        
        function toggleAutoplay() {{
            const btn = document.getElementById('playBtn');
            if (autoplay) {{
                clearInterval(autoplayInterval);
                btn.textContent = '▶️ Play';
                autoplay = false;
            }} else {{
                autoplayInterval = setInterval(nextSlide, 1500);
                btn.textContent = '⏸️ Pause';
                autoplay = true;
            }}
        }}
        
        // Keyboard controls
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'ArrowLeft') previousSlide();
            if (e.key === 'ArrowRight') nextSlide();
            if (e.key === ' ') {{
                e.preventDefault();
                toggleAutoplay();
            }}
        }});
    </script>
</body>
</html>"""
        
        # Save HTML file
        with open(output_path / "slideshow.html", "w") as f:
            f.write(html_content)
        
        print(f"🌐 Created slideshow viewer: {output_path}/slideshow.html")
        print("   Open in browser to view the animated progression!")

def main():
    parser = argparse.ArgumentParser(description="Generate diffusion step slideshow")
    parser.add_argument("prompt", nargs='?', help="Text prompt for image generation")
    parser.add_argument("--steps", type=int, default=30, help="Number of steps to capture")
    parser.add_argument("--output", default="slideshow_output", help="Output directory")
    parser.add_argument("--model", choices=list_available_models(), 
                       help="Model to use (default: from config)")
    parser.add_argument("--list-models", action="store_true", 
                       help="List available models and exit")
    
    args = parser.parse_args()
    
    if args.list_models:
        from config import MODEL_CONFIG
        print("🔧 Available Models:")
        print("=" * 40)
        for name, config in MODEL_CONFIG["models"].items():
            print(f"📦 {name}")
            print(f"   {config['description']}")
            print(f"   Model: {config['model_id']}")
            print(f"   Optimal steps: {config['optimal_steps']}")
            print()
        return
    
    if not args.prompt:
        parser.error("Prompt is required when not using --list-models")
    
    # Generate slideshow
    slideshow = DiffusionSlideshow(num_steps=args.steps, model_name=args.model)
    output_path = slideshow.generate_slideshow(args.prompt, args.output)
    
    print(f"\n🎉 Slideshow complete!")
    print(f"📁 Files saved to: {output_path}")
    print(f"🌐 View slideshow: open {output_path}/slideshow.html")

if __name__ == "__main__":
    main()
