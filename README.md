# GES-Explorer (ACM TOG 2025)

An interactive, browser-based educational web visualizer and offline processing framework designed to explain the core mechanisms of the research paper: **"When Gaussian Meets Surfel: Ultra-fast High-fidelity Radiance Field Rendering"** (GES) by Ye et al. (ACM TOG 2025).

This project is part of a Computer Graphics and Visualization (CGV) seminar and follows a **"Compute Offline, Replay Online"** architecture. It splits the workload into:

1. **Offline Training & Preprocessing (Python/PyTorch):** Joint surfel-Gaussian optimization built on top of Nerfstudio, checkpoint dumping, and custom preprocessing tools to package optimized/quantized scenes.
2. **Online Interactive Visualizer (TypeScript/Three.js/GLSL):** A web-based tool that renders surfels and Gaussians from a single combined `scene.ply`. A custom GLSL shader draws surfels as flat opaque discs that write depth, so the hardware z-buffer occludes the Gaussians behind them — a real-time take on the paper's *surfel-first* occlusion. A separate **Artifact Comparison** tab then uses interactive 2D/3D diagrams to demonstrate the full sorting-free compositing, tile-seam, depth-offset ($\delta$), and specular-limitation concepts.

---

## 📂 Project Structure

```
CGV_Seminars/
├── src/                        # Custom Offline Python/PyTorch Codebase (Graded)
│   ├── main.py                 # Nerfstudio training entry point
│   ├── config.py               # Model and pipeline configurations
│   ├── ges_model.py            # Custom GES model implementation (surfel-gaussian logic)
│   ├── ges_strategy.py         # Custom densification/pruning strategies
│   └── training_schedule.py    # Training hyperparameters and schedule overrides
├── tools/                      # Offline preprocessing & diagnostic tools
│   ├── pack_for_web.py         # Formats/quantizes raw PyTorch checkpoints into combined PLY assets
├── web/                        # Online Interactive Visualizer (TypeScript/GLSL)
│   ├── components/
│   │   ├── viewer/             # Main 3D GES renderer & scene loaders
│   │   └── comparison/         # Artifact comparison tab, interactive 2D/3D card modules
│   │       ├── diagrams/       # Custom canvas diagrams (orbit, tiles, depth-leakage)
│   │       └── viewers/        # Multi-viewer side-by-side Three.js instances
│   ├── index.html              # Web UI entry point with KaTeX integration
│   ├── main.ts                 # Web app bootstrapping logic
│   └── style.css               # Modern dark-mode UI styles
├── external_code/              # External dependencies
│   ├── nerfstudio/             # Local clone of Nerfstudio (v1.1.5 tag)
│   ├── gsplat/                 # Local clone of gsplat (v1.5.3 tag)
│   └── render/                 # Custom local rendering package
├── environment.yml             # Conda environment definition for PyTorch/CUDA
├── requirements.txt            # Pip dependencies for preprocessing tools
```

---

## 🛠 Offline Pipeline Setup & Training (Python/PyTorch)

Follow these steps to configure the local environment, run the optimization, and prepare assets for the web visualizer.

### 1. Create the Environment

Create a local Conda environment containing PyTorch, CUDA, and build essentials (e.g., `ninja` compiler):

```powershell
conda env create --prefix ./.env -f environment.yml
conda activate ./.env
```

### 2. Clone & Install Local Packages (Editable Mode)

Since the `external_code/nerfstudio` and `external_code/gsplat` codebases are not committed to the Git repository, you must clone their specific target tags into the `external_code` directory first, and then install them in editable mode alongside the custom `render` package:

```powershell
# 1. Clone the specific tags into external_code/
git clone --branch v1.1.5 https://github.com/nerfstudio-project/nerfstudio.git external_code/nerfstudio
git clone --branch v1.5.3 --recursive https://github.com/nerfstudio-project/gsplat.git external_code/gsplat

# 2. Remove any pre-installed/official nerfstudio package to avoid import conflicts
pip uninstall nerfstudio -y

# 3. Install all packages in editable mode (-e)
pip install -e external_code/nerfstudio
pip install -e external_code/gsplat
pip install -e external_code/render
```

### 3. Data Preparation

Unzip your dataset inside the root directory. For blender/synthetic datasets, ensure they follow this structure:
`nerf_synthetic/chair/transforms_train.json`

### 4. Run Training

> <span style="color: red; font-weight: bold;"> [!CAUTION]
> A CUDA-compatible GPU is strictly required for training.</span> The optimization kernels (such as gsplat and custom nerfstudio modules) rely on CUDA and **will not run on CPU**.

To run optimization on the synthetic "Chair" dataset:

```powershell
python src/main.py --data <path_to_dataset>
```

* **0-10k Iterations:** Joint surfel-Gaussian optimization and densification.
* **10k Mark:** Surfel discard (low opacity) and seeding step.
* **18k-20k Mark:** Transition phase ramping up disc opacity $\tau$ from $30 \rightarrow 90 \rightarrow 255$.
* **20k-30k Iterations:** Joint refinement of Gaussians and Surfel colors.

### 5. Packaging Checkpoints for the Web

Use the custom exporter to package raw PyTorch weights into optimized, web-ready PLY files:

```powershell
python tools/pack_for_web.py
```

This saves combined PLY files (mapping primitives with a `prim_type` attribute: `0` for Gaussians, `1` for Surfels) into the `web_assets` directory.

---

## 🌐 Online Visualizer Setup & Running (Vite/TypeScript)

The web visualizer reads the exported PLY models and renders them using high-performance custom GLSL shaders.

### 1. Install Dependencies

Navigate to the `web` folder and install Node.js dependencies:

```powershell
cd web
npm install
```

### 2. Run the Development Server

Start Vite's development server locally:

```powershell
npm run dev
```

Open the provided localhost address (typically `http://localhost:5173`) in your browser.

> [!NOTE]
> No large scene is bundled with the repository. The **GES Web Viewer** tab starts empty — click **"Choose scene.ply"** in the control panel to load a scene you exported in step 5 (or any combined GES PLY). The **Artifact Comparison** tab works without any scene.

### 3. Build for Production

To bundle the web app into a static production build:

```powershell
npm run build
```

This outputs compiled assets into `web/dist/`, which can be served as a static webpage.

---

## 💡 Key Educational Modules in GES-Explorer

`GES-Explorer` is structured into two main tabs to make paper concepts interactive and tangible:

### 1. GES Web Viewer (Tab 1)

* **Interactive 3D Inspector:** Load any exported `scene.ply` and toggle **Surfels Only** / **Gaussians Only** / **both** to inspect the bi-scale representation. Blender-style navigation (left-drag orbit, right-drag pan, scroll zoom).
* **Custom GLSL Rendering:** Surfels are drawn as flat opaque discs (custom shader, Eq. 6-7) that write depth and occlude the Gaussians behind them via the hardware z-buffer — a real-time realization of the paper's surfel-first occlusion. (The full additive Eq. 2-5 compositing is illustrated interactively in Tab 2.)

### 2. Artifact Comparison (Tab 2)

Provides visual and math-backed proof of how GES addresses 3DGS rendering flaws:

* **Card 0: The Two-Pass Render Flow:** Interactive overview schematic showing the rasterization of surfels (Pass 1) to produce a depth/color map, followed by additive Gaussian splatting (Pass 2) with a depth-offset test ($\delta$).
* **Card ▶: Live 3D Viewport Orbit:** Side-by-side interactive orbit comparison. Watch 3DGS colors pop as splat depth rankings swap, while GES remains smooth. It lists live depths and culls splats behind surfels (marked with a red ✗ on the GES side).
* **Card 1: Camera Orbit & Compositing:** Interactive camera orbit slider demonstrating why 3DGS suffers from depth sorting popping artifacts while GES's order-independent math maintains view-consistent rendering.
* **Card 2: Tile Boundaries:** Interactive slider illustrating tile-border artifacts. See how 3DGS causes hard seams when tiles drop boundary splats, whereas GES renders seam-free images.
* **Card 3: Depth Test & Color Leaking:** Interactive depth-offset ($\delta$) slider demonstrating that sorting-free methods leak colors of background primitives through foreground geometry, and how GES's depth-test culls them cleanly.
* **Card 4: Specular Reflections (Limitations):** Demonstrates a core trade-off: 3DGS can fake mirror reflections using semi-transparent underground floaters. GES's opaque surfel surface blocks underground light rays, revealing why capturing specular reflections is a limitation of the surfel-first pass.

---

## 📦 External Resources

All graded, custom code lives in `src/`, `tools/`, and `web/`. Everything below is third-party
and is isolated under `external_code/` (for imported source) or pulled in as a package manager
dependency. **Nothing in `external_code/` is graded.**

### Imported source code (`external_code/`)

| Resource                                                                                                          | Version | Used for                                                                                                                                                                                                              |
| ----------------------------------------------------------------------------------------------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Nerfstudio** — [github.com/nerfstudio-project/nerfstudio](https://github.com/nerfstudio-project/nerfstudio) | v1.1.5  | Training framework. Our GES model (`src/ges_model.py`) subclasses its `Splatfacto` model and reuses its trainer, data parsers, and camera pipeline. Cloned + installed editable (see setup).                      |
| **gsplat** — [github.com/nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat)             | v1.5.3  | Differentiable CUDA rasterizer; backend for both the surfel (2DGS) and Gaussian (3DGS) passes during optimization. Cloned + installed editable.                                                                       |
| **render** (`external_code/render`)                                                                       | local   | Our custom C++/CUDA surfel-rasterizer extension. It is*our* code but is placed under `external_code/` purely for isolation per the submission rules. due to relatively small num of chanegs to the original code. |

### Python libraries (offline pipeline)

PyTorch + torchvision (autograd/CUDA backend), **pytorch-msssim** (SSIM term in the loss),
**torchmetrics** (PSNR/LPIPS evaluation), **plyfile** (read/write the exported PLY in
`tools/pack_for_web.py`), plus NumPy, SciPy, tqdm, and zstandard. See `requirements.txt` /
`environment.yml`.

### Web libraries (`web/`, via npm — see `web/package.json`)

| Resource                                                                                                                       | Version      | Used for                                                               |
| ------------------------------------------------------------------------------------------------------------------------------ | ------------ | ---------------------------------------------------------------------- |
| **three.js** — [threejs.org](https://threejs.org)                                                                          | ^0.184.0     | WebGL scene graph, camera, and `OrbitControls` for all 3D views.     |
| **@mkkellogg/gaussian-splats-3d** — [github.com/mkkellogg/GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) | ^0.4.7       | Splat-PLY parsing and `DropInViewer` rendering of surfels/Gaussians. |
| **Vite** — [vitejs.dev](https://vitejs.dev)                                                                                | ^8.0.13      | Dev server and production bundler.                                     |
| **KaTeX** — [katex.org](https://katex.org)                                                                                 | 0.16.9 (CDN) | Renders the equations shown in the UI (loaded in `index.html`).      |

### Data

The datasets used in this project can be accessed at the following links:
* **Lounge:** [Google Drive Folder](https://drive.google.com/drive/folders/17NIX-caohju88_jiy14DKbviBHZGXLPf?usp=drive_link)
* **Truck (T&T):** [Hugging Face Dataset (tandt/truck)](https://huggingface.co/datasets/alexmkwizu/gaussian_training_datasets/tree/main/tandt/truck)
* **Garden (Mip-NeRF 360):** [Hugging Face Dataset (3DGS_datasets)](https://huggingface.co/datasets/ZaY19/3DGS_datasets/tree/main)

---

## 🤖 AI Assistance

In the interest of transparency:

* The **web UI code and visual design** : the control panel, layout/CSS, and the Artifact-Comparison diagram cards , were produced **largely with AI assistance**. While the underlying logic, algorithms are written by me, couldn't  separate them considering they are very tangled.
* The **core rendering and algorithm logic** was authored by me: the offline GES training pipeline (surfel/Gaussian optimization, densification/pruning strategy, the custom CUDA surfel rasterizer) and the renderer's shading/depth handling, in both train and the web view.
* AI assistance was also used to write the **code comments** and to **clean up** the `src/ python code and web/ ts code` .

---

## 📚 Paper & References

* **Paper:** Keyang Ye, Tianjia Shao, Kun Zhou. *When Gaussian Meets Surfel: Ultra-fast High-fidelity Radiance Field Rendering.* ACM TOG 2025.
* **Authors' reference code:** [https://github.com/YessionCC/GES](https://github.com/YessionCC/GES) — consulted for the algorithm; not vendored into this submission.
