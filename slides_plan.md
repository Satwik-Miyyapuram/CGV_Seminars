# Presentation Plan: When Gaussian Meets Surfel

Based on the paper "When Gaussian Meets Surfel", the course organization guidelines, and the presentation tips, here is a detailed, slide-by-slide plan for a 10-minute presentation. 

## Core Presentation Guidelines Addressed:
1.  **Timing (10 min + 3 min Q&A):** The plan is scoped to 11 slides, allowing roughly 1 minute per slide to fit the 10-minute limit.
2.  **Content Ratio:** Follows the "20% Motivation, 80% Key Idea" rule.
3.  **The "One Thing":** Visibly structured around the core concept of a **bi-scale representation (surfels + Gaussians)** enabling **sorting-free rendering**.
4.  **No Dry Related Work:** Prior work (3DGS) is integrated directly into the motivation/problem statement to build the story.
5.  **Equation Intuition:** The rendering equation is included but the focus is on the *intuition* behind it, not just the math.
6.  **Visuals over Text:** Slides are planned with minimal bullet points, emphasizing the use of figures from the paper.

---

## Slide-by-Slide Outline

### Slide 1: Title Slide
*   **Title:** When Gaussian Meets Surfel: Ultra-fast High-fidelity Radiance Field Rendering
*   **Visual:** The teaser image (Fig. 1) showing the quality and speed of GES compared to other methods.
*   **Notes:** Welcome the audience and state the title.

### Slide 2: Motivation - The Goal
*   **Content:** Synthesizing novel views of 3D scenes from 2D images in real-time.
*   **Why it matters:** Crucial for immersive VR/AR experiences and interactive 3D applications.
*   **Visual:** A short, clear example of Novel View Synthesis.

### Slide 3: Motivation - Limits of Current Practice (3DGS)
*   **Content:** The current state-of-the-art is 3D Gaussian Splatting (3DGS). However, 3DGS relies on alpha-blending, which requires depth-sorting millions of Gaussians.
*   **The Problem:** Approximating this sort (tile-based sorting) leads to "popping artifacts"—patchy colors that flicker when the camera moves.
*   **Visual:** Highlight the popping artifact (cropping from Fig. 12 comparing 3DGS popping vs GES smooth views).

### Slide 4: The Key Idea (The "One Thing")
*   **Content:** **Gaussian-enhanced Surfels (GES)**. If you forget everything else, just remember this: GES uses a **bi-scale representation** to achieve **completely sorting-free rendering**.
*   **Visual:** High-level conceptual diagram of Surfels + Gaussians.

### Slide 5: The Bi-Scale Representation (Part 1: Coarse)
*   **Content:** **2D Opaque Surfels**. These represent the coarse-scale geometry and base appearance of the scene.
*   **Visual:** Show Fig. 2 (left) - just the surfel rendering. Emphasize that it looks like the basic structure but lacks fine details.

### Slide 6: The Bi-Scale Representation (Part 2: Fine)
*   **Content:** **3D Gaussians**. These surround the surfels and supplement the high-frequency details and complex textures.
*   **Visual:** Show Fig. 2 (middle) - just the Gaussian rendering, then transition to Fig. 2 (right) - the full GES rendering.

### Slide 7: Sorting-Free Rendering Pipeline
*   **Content:** How does it avoid sorting?
    1.  **Pass 1:** Rasterize opaque surfels (standard z-buffer, fast). Output: Depth Map & Color Map.
    2.  **Pass 2:** Splat 3D Gaussians. If a Gaussian is behind the surfel depth map, ignore it. Otherwise, accumulate its color.
*   **Visual:** Fig. 3 (Rendering Pipeline).
*   **Equation Intuition:** Briefly show `Final Image = Surfel Color + Accumulated Gaussian Color`. Explain that because surfels block what's behind them, we don't need to sort the whole scene!

### Slide 8: Coarse-to-Fine Optimization (The Challenge)
*   **Content:** How do we train opaque surfels? Standard gradients don't work well on hard, opaque edges.
*   **The Solution:** Start with translucent surfels (like 2D Gaussians) and use an "opacity modulating parameter" to gradually solidify them from the center outward until they are fully opaque discs.
*   **Visual:** Fig. 4 showing the shape of opacity changing as it optimizes.

### Slide 9: Results - Artifact-Free View Consistency
*   **Content:** Does it solve the popping problem? Yes.
*   **Visual:** Fig. 12 comparing the view consistency of GES against 3DGS and SpeedySplat. 

### Slide 10: Results - Speed and Quality
*   **Content:** GES achieves 675 FPS at 1080p, while the "Speedy-GES" extension hits 1135 FPS.
*   **Visual:** Quick summary table or chart (from Table 1 & Table 3) showing FPS vs. PSNR. Emphasize that it matches SOTA quality but is significantly faster.

### Slide 11: Conclusion
*   **Content:** GES successfully marries point-based surface rendering (surfels) with volumetric rendering (Gaussians). 
*   **Takeaway:** Sorting-free rendering is a viable and powerful path forward for real-time radiance fields.
*   **End:** "Thank you. Questions?"
