# GES-Explorer (ACM TOG 2025)

Educational web visualizer for the research paper **"When Gaussian Meets Surfel" (GES)**.

## 🛠 Setup Instructions

Follow these steps exactly to ensure the CUDA kernels and Nerfstudio dependencies are correctly linked.

### 1. Create the Environment
Create a local conda environment inside the project folder:
```powershell
conda env create --prefix ./.env -f environment.yml
conda activate ./.env
```

### 2. Install Local Packages (Editable Mode)
To ensure the custom logic in `external_code` is used, we must install the local versions as editable packages.
```powershell
# Remove the official nerfstudio to avoid conflicts
pip uninstall nerfstudio -y

# Install local versions
pip install -e external_code/nerfstudio
pip install -e external_code/gsplat
```

### 3. Data Preparation
Unzip your datasets into the root directory. For synthetic data, ensure it follows this structure:
`nerf_synthetic/chair/transforms_train.json`

---

## 🚀 Training

To train the model on the synthetic "Chair" dataset:

```powershell
python src/main.py blender-data --data nerf_synthetic/chair
```

### Key Parameters:
*   **0-10k Iterations:** Surfel optimization and densification.
*   **10k Mark:** Surfel discard (opacity < 0.8) and seed point generation.
*   **20k Mark:** Surfel solidification (opacity=255) and Gaussian spawning.
*   **20k-30k Iterations:** Joint refinement of Gaussians and Surfel colors.

---

## 🌐 Visualization

Once training starts, open the provided Viser link (usually `http://localhost:7007`) in your browser to watch the bi-scale representation evolve.
