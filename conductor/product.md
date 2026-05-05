# Initial Concept

The goal is to build **GES-Explorer**, an interactive, browser-based educational tool to support the presentation of the research paper "When Gaussian Meets Surfel: Ultra-fast High-fidelity Radiance Field Rendering" (GES). It will visualize and explain the core mechanisms of the paper that are hard to grasp from text alone, including:
1. The two-pass, sorting-free renderer.
2. The opacity-modulating parameter τ.
3. The depth offset δ.
4. Popping artifacts in 3DGS vs. GES's view consistency.
5. The coarse-to-fine optimization process (replayed from precomputed checkpoints).
6. The limitations of GES.

The architecture relies on computing expensive operations (optimization, reference renders) offline in Python and replaying them efficiently in the browser using TypeScript, three.js, and custom WebGL2 shaders.

# Product Guide

## Target Audience
The primary audience for **GES-Explorer** consists of peers and course grading staff. The tool is specifically designed to support a live seminar presentation and clearly demonstrate the student's mastery of the material and technical implementation.

## Educational Goal
The overriding pedagogical objective is "Intuition over Math." While the tool will touch upon equations like the two-pass compositing formula, the emphasis is on providing tangible, visual demonstrations of complex concepts—making them instantly understandable without requiring the audience to parse dense formulas during a 10-minute presentation.

## Key Interactive Features
To achieve its educational goals, GES-Explorer will feature:
*   **Live Parameter Sweeps:** Interactive sliders allowing the user to manipulate the opacity-modulating parameter τ and the depth offset δ, instantly visualizing their effects on the rendering.
*   **Pipeline Visualization:** A step-by-step UI breaking down the sorting-free rendering passes, from surfel rasterization to Gaussian splatting and final composition.
*   **Optimization Replay:** A timeline scrubber that lets the presenter and users watch the model train smoothly from precomputed checkpoints, illustrating the coarse-to-fine process.

## Data Handling & Architecture
To ensure the tool remains fast, responsive, and reliable during a live talk, it will rely **strictly on precomputed data**. All heavy lifting—such as the full GES optimization run, tile-based 3DGS reference rendering, and multi-view re-renders—will be computed offline. The browser's role is strictly to load, stream, and render these artifacts interactively.