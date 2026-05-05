# Track Specification: Implement offline optimization driver and core two-pass WebGL renderer

## 1. Overview
This track forms the foundation of GES-Explorer by implementing the offline data generation pipeline and the online rendering framework. 

## 2. Requirements
*   **Offline Driver:** Wrap the authors' PyTorch optimizer to output binary checkpoint files at specific iterations. 
*   **Data Packing:** Implement a packing script that quantizes spherical harmonics (float16) and compresses the output with zstd to fit the 200 MiB project limit.
*   **Browser Loader:** Build a mechanism in TypeScript to stream, decompress, and decode the binary payload into GPU buffers.
*   **WebGL Renderer:** Use three.js and custom GLSL shaders to implement a two-pass render pipeline (Surfel rasterization with Z-buffer -> Gaussian splatting with depth test and additive accumulation).

## 3. Out of Scope
*   Advanced UI controls, interactive sliders for tau/delta, and the limitations gallery (these will be separate tracks).