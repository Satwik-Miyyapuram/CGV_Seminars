# Project Overview: CGV Seminar — When Gaussian Meets Surfel (GES)

This project is part of a Computer Graphics and Visualization (CGV) seminar. It centers around the research paper: **"When Gaussian Meets Surfel: Ultra-fast High-fidelity Radiance Field Rendering"** (GES) by Ye et al. (ACM TOG 2025).

The primary goals are:
1.  **Technical Implementation:** Developing **`GES-Explorer`**, an interactive, browser-based educational tool to visualize and explain the core mechanisms of the GES paper (e.g., bi-scale representation, two-pass rendering, $\tau$ modulation).
2.  **Presentation:** Delivering a 10-minute presentation supported by the visualizer and a slide deck.

## Technical Architecture

The project follows a **"Compute Offline, Replay Online"** architecture:
-   **Offline (Python/PyTorch):** Uses the authors' original optimizer (located in `external_code/`) to generate training checkpoints and failure-case scenes.
-   **Preprocessing:** Custom Python tools (`tools/`) process raw checkpoints into optimized, quantized binary files for the web.
-   **Online (TypeScript/three.js/GLSL):** A high-performance web-based visualizer that streams precomputed data and renders it using custom shaders.

## Key Modules in GES-Explorer

-   **Module A (Representation Inspector):** Inspect surfels and Gaussians.
-   **Module B (The Two-Pass Pipeline):** Visualizes the rendering steps (Rasterization -> Splatting -> Compositing).
-   **Module C (Order-Independence Proof):** Demonstrates that sorting is unnecessary in GES.
-   **Module D (Popping vs. Consistency):** Side-by-side comparison with 3DGS to show view consistency.
-   **Module E (Parameter Explorers):** Interactive sliders for $\tau$ and $\delta$, plus an optimization replay scrubber.
-   **Module F (Limitations Gallery):** Explicitly surfaces failure modes and motivations for paper extensions.

## Building and Running

### Development Requirements
-   **Node.js & Vite:** For the browser-based visualizer.
-   **Python 3.x with PyTorch:** For the offline optimization driver and asset generation.

### Key Commands
-   **TODO:** Identify exact npm/pip install and run commands.
-   `npm run dev`: (Inferred) Start the Vite dev server for the visualizer.
-   `python tools/run_optim.py`: (Inferred) Run the optimization driver.

## Project Structure

-   `paper.pdf` / `paper.txt`: The core research paper.
-   `implementation_plan.md`: Technical roadmap for `GES-Explorer`.
-   `slides_plan.md`: Slide-by-slide plan for the seminar presentation.
-   `Implementation.pdf`: Official course requirements and grading criteria.
-   `tools/`: (Planned) Python scripts for asset generation and optimization.
-   `src/`: (Planned) TypeScript and GLSL source code for the visualizer.
-   `external_code/`: Third-party libraries (three.js) and the authors' optimizer.

## Development Conventions

-   **Educational Focus:** The code is not a production renderer but a teaching tool. Clarity and "open-box" visibility are prioritized.
-   **Strict Separation:** Graded custom code (visualizer, drivers) must be clearly distinguished from the authors' code in `external_code/`.
-   **Performance:** Precomputation is used to ensure the browser tool remains responsive during live demonstrations.
