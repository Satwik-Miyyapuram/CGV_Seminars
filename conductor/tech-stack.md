# Technology Stack: GES-Explorer

This document defines the core technologies used to implement GES-Explorer.

## 1. Core Frameworks
*   **nerfstudio:** The overarching framework providing the training loop, dataloaders, evaluation metrics, and the web viewer (`viser`).
*   **gsplat:** Provides the highly optimized CUDA kernels for rasterizing both 2D Gaussians (Surfels) and 3D Gaussians.
*   **PyTorch:** The deep learning library underlying the entire optimization process.

## 2. Programming Languages
*   **Python (3.x):** Used for all custom logic, including the custom Nerfstudio `Model` and `Pipeline`.
*   **CUDA (C++):** (Via `gsplat`) Used implicitly for fast rendering. We will avoid writing custom CUDA unless absolutely necessary for the GES sorting logic.

## 3. Presentation / Viewer
*   **viser (via nerfstudio):** A Python-driven React frontend used for visualizing the training progress and exploring the final trained model interactively. The presentation will run live locally.