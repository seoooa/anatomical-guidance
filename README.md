# Context-Aware 3D Medical Image Segmentation with Anatomical Guidance
Accurate 3D medical image segmentation presents a significant challenge due to indistinct boundaries, anatomical ambiguity, and limited supervision. To address this, we introduce an anatomy-guidance mechanism that conditions segmentation networks on signed distance maps (SDMs) generated from pre-trained multi-organ segmentation models. These maps encode long-range spatial context, serving as soft anatomical priors. This guidance, integrated via a lightweight FiLM-based module, enhances network performance without requiring additional annotations or altering the core architecture. Our method substantially improves both segmentation accuracy and structural consistency across diverse datasets and network backbones. Notably, it achieves significant reductions in boundary errors (e.g., up to 55\% in HD95) and topological inconsistencies while maintaining computational efficiency.

## Project Structure

The project is organized as follows:
- `data/`: Directory for storing datasets
- `nbs/`: Directory for Jupyter Notebook files
- `script/`: Directory for experiment and test scripts
- `src/`: Directory for source code
  - `archs/`: Code related to model architectures
  - `data/`: Code for data processing
  - `losses/`: Code for loss functions
  - `metrics/`: Code for evaluation metrics
  - `models/`: Code for model implementations
  - `utils/`: Directory for utility functions
- `script/`: Scripts for running the demo

## Installation

### Prerequisites
- Python 3.10
- uv (for dependency management and virtual environment)

### Setup

#### Option 1: Using uv (Recommended)

1. Install uv if you don't have it already:
   ```
   pip install uv
   ```

2. Clone the repository:
   ```
   git clone
   cd anatomical-guidance
   ```

3. Create a virtual environment and install dependencies using uv:
   ```
   uv venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate     # Windows
   ```

4. Install dependencies:
   ```
   uv pip install -r requirements.txt
   ```

5. Install PyTorch with CUDA support:
   ```
   uv pip install torch==2.1.1+cu121 torchvision==0.16.1+cu121 torchaudio==2.1.1+cu121 --index-url https://download.pytorch.org/whl/cu121
   ```