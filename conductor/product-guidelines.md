# Product Guidelines: GES-Explorer

These guidelines ensure that **GES-Explorer** remains a high-quality, performant, and reliable educational tool suited for live presentations and peer evaluation.

## 1. User Experience (UX) Principles
*   **Performance is Paramount:** The visualizer must maintain a stable framerate. All heavy operations during rendering must be optimized via CUDA (using `gsplat`).
*   **Content Over Chrome:** The 3D viewport and visual comparisons are the core product. UI elements in the `viser` viewer should be clean and not obscure the rendering.
*   **Instant Feedback:** Sliders for parameters like $\tau$ and $\delta$ must update the rendering in real-time.

## 2. Educational Design
*   **Accessible Mathematics:** When possible, expose key pipeline variables (like the split between surfel and gaussian rendering) via the viewer GUI.
*   **Side-by-Side Clarity:** Utilize Nerfstudio's rendering capabilities to allow toggling between the 2DGS pass, 3DGS pass, and the combined GES pass.

## 3. Code Style & Architecture
*   **Nerfstudio Integration:** We are building a custom `Model` and `Pipeline` within the `nerfstudio` ecosystem. Custom logic should be encapsulated in a dedicated `src/` module and registered with `nerfstudio`.
*   **gsplat Dependency:** We rely on `gsplat` for the core rasterization kernels of both 2D Gaussians (Surfels) and 3D Gaussians. Our custom code will handle the mixing and order-independent logic defined in the GES paper.
*   **Live Backend:** The presentation will rely on running the `nerfstudio` viewer locally, utilizing the Python backend and CUDA to render and stream the UI to the browser.