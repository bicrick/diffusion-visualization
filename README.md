# Diffusion Visualization

An interactive 3D visualization application that helps users understand how diffusion models generate images through an intuitive, visually-driven educational experience.

## Vision

This project aims to demystify the complex process of diffusion models by creating an engaging visual journey that shows:
- How images emerge from noise through iterative denoising
- The neural network architecture working in real-time 
- The mathematical concepts behind diffusion without heavy equations
- The "sculpting from chaos" process that transforms random noise into coherent images

## Educational Goals

**Target Understanding:**
- "It's not creating something from nothing - it's finding something that was always potentially there"
- "The noise contains all possibilities, and the model selects one path"
- "Each part of the neural network has a specific job in this organized process"

**Visual Metaphors:**
- **Sculpting from Chaos**: Watch noise get carved away to reveal hidden images
- **Archaeological Dig**: Excavate images buried under layers of noise
- **Orchestra Conducting**: See how different network layers conduct chaos into harmony

## Technical Approach

### Pre-Rendered Asset Strategy

Rather than real-time model inference, this application uses pre-computed assets for optimal performance and educational control:

- **Preset Prompts**: Curated set of prompts with complete diffusion sequences
- **Step-by-Step Images**: Each denoising step saved as static images (20-50 steps per prompt)
- **Attention Maps**: Pre-computed attention visualizations for each step
- **Architecture Data**: Network structure and layer information as JSON

### Tech Stack

**Frontend:**
- **Three.js** - 3D neural architecture visualization
- **React** - UI components and state management  
- **TypeScript** - Enhanced development experience
- **Framer Motion** - Smooth step transitions and animations
- **Tailwind CSS** - Rapid UI styling
- **Vite** - Fast development and optimized builds

**Data Pipeline (Offline):**
- **Python + PyTorch/Diffusers** - Generate diffusion sequences
- **Hugging Face Models** - Source for pre-trained diffusion models
- **JSON + Images** - Export web-friendly visualization data

### Project Structure

```
diffusion-visualization/
├── src/
│   ├── components/
│   │   ├── DiffusionViewer.tsx     # Main visualization component
│   │   ├── ArchitectureView.tsx    # 3D neural network renderer
│   │   ├── StepController.tsx      # Timeline and controls
│   │   └── EducationalPanels.tsx   # Contextual explanations
│   ├── data/
│   │   └── prompts/
│   │       ├── prompt1/
│   │       │   ├── metadata.json   # Step info, timing, concepts
│   │       │   ├── steps/          # step_000.png -> step_050.png
│   │       │   └── attention/      # attention maps per step
│   │       └── prompt2/...
│   ├── models/
│   │   └── architecture.json       # Network structure definition
│   └── utils/
│       ├── threeHelpers.ts         # 3D rendering utilities
│       └── dataLoaders.ts          # Asset loading and caching
├── scripts/
│   └── generate_assets.py          # Offline asset generation
└── public/
    └── assets/                     # Static visualization assets
```

## Asset Generation Pipeline

### 1. Model Selection
- Choose diffusion model from Hugging Face
- Consider models with good attention visualization capabilities
- Prioritize educational value over state-of-the-art performance

### 2. Prompt Curation  
- Select diverse, visually interesting prompts
- Balance simple concepts (dog, house) with complex scenes
- Ensure good progression from noise to final image

### 3. Data Export
```python
# Example workflow
for step in diffusion_steps:
    - Save denoised image at current step
    - Extract and visualize attention maps  
    - Record metadata (concepts detected, confidence levels)
    - Export 3D-ready architecture activations
```

### 4. Web Optimization
- Compress images for web delivery
- Generate progressive loading sequences
- Create thumbnail previews for quick browsing

## Key Features (MVP)

### Interactive 3D Architecture
- **Network Visualization**: 3D representation of U-Net layers, attention mechanisms
- **Data Flow**: Animated particles showing information flow through layers
- **Layer Inspection**: Click any layer to understand its role
- **Skip Connections**: Visual bridges showing how information is preserved

### Step-by-Step Progression
- **Timeline Scrubber**: Smoothly navigate through all diffusion steps
- **Split Views**: Compare noisy input vs. clean output
- **Attention Overlay**: Toggle attention heatmaps on/off
- **Concept Emergence**: Track when objects/features become recognizable

### Educational Context
- **Progressive Disclosure**: Information revealed as users explore
- **Visual Analogies**: Connect abstract concepts to familiar ideas  
- **Interactive Discoveries**: "Notice anything interesting about step 15?" prompts
- **Architecture Spotlights**: Highlight specific network components in action

## Future Enhancements

- **Multi-Model Comparison**: Compare different diffusion architectures
- **Custom Prompts**: Allow users to input new prompts (with real-time inference)
- **Interactive Training**: Show how the model learned these patterns
- **Performance Metrics**: Visualize computational cost and efficiency
- **Mobile Optimization**: Touch-friendly 3D interactions

## Getting Started

1. **Clone and Install**
   ```bash
   git clone <repo-url>
   cd diffusion-visualization
   npm install
   ```

2. **Set up Python Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Generate Assets** (with automatic model caching)
   ```bash
   # First run downloads and caches the model (~10-15GB for FLUX.1-dev)
   python scripts/generate_assets.py
   
   # Subsequent runs use the cached model (loads in seconds!)
   python scripts/generate_assets.py --prompt "A beautiful sunset over mountains"
   ```

4. **Development**
   ```bash
   npm run dev
   ```

## Model Caching System

The asset generator now includes an intelligent caching system that stores models locally to avoid repeated downloads:

### Features
- **Automatic caching**: Models are cached on first download
- **Fast loading**: Cached models load in seconds vs. minutes
- **Cache validation**: Ensures cached models are complete and valid
- **Space management**: Track cache size and clear when needed

### Usage Examples

```bash
# Generate with default settings (caches model automatically)
python scripts/generate_assets.py

# Use different model (will cache this model too)
python scripts/generate_assets.py --model "stabilityai/stable-diffusion-xl-base-1.0"

# Disable caching (always download fresh)
python scripts/generate_assets.py --no-cache

# Custom cache directory
python scripts/generate_assets.py --cache-dir "./my_models"
```

### Cache Management

```bash
# List all cached models with sizes
python scripts/cache_management.py list

# Clear all cached models
python scripts/cache_management.py clear --all

# Clear specific model
python scripts/cache_management.py clear "black-forest-labs/FLUX.1-dev"

# Pre-download a model for later use
python scripts/cache_management.py download "stabilityai/stable-diffusion-xl-base-1.0"

# Show cache directory info
python scripts/cache_management.py info
```

### CLI Options for Asset Generation

```bash
python scripts/generate_assets.py --help

# Key caching options:
--cache-dir DIR          # Set cache directory (default: models_cache)
--no-cache              # Disable caching completely
--clear-cache           # Clear all cache and exit
--clear-model-cache ID  # Clear specific model cache and exit
--list-cache           # List cached models and exit
```

## Contributing

This project welcomes contributions in:
- **Educational Content**: Better explanations and visual metaphors
- **Visualization Ideas**: New ways to represent complex concepts
- **Performance**: Optimizations for smoother 3D rendering
- **Accessibility**: Making the content accessible to diverse learners

## License

MIT License - See LICENSE file for details

---

*Making AI understandable through beautiful, interactive visualization.*
