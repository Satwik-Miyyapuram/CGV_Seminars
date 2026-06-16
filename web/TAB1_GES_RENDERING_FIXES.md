# Tab 1 (GES Web Viewer) — Rendering fixes to match the paper

> **Status: NOT YET APPLIED.** These changes are deferred until the Tab 1 training pipeline
> is finalized (PLY layout / depth conventions may still change). This file is the spec to
> apply later. Tab 2 (Artifact Comparison) fixes have already been applied.

Reference: *"When Gaussian Meets Surfel: Ultra-fast High-fidelity Radiance Field Rendering"*
(`paper.txt`), Section 3 (Eq. 2–7) and Section 5 (Implementation Details).

## What already matches the paper (do not change)

The two-pass structure in `components/viewer/SceneManager.ts` is faithful:

- **Pass 1** renders surfel color; **Pass 2** writes surfel depth to the z-buffer;
  **Pass 3** splats Gaussians with depth testing + additive blending; the composite
  shader normalizes by accumulated weight.
- Composite `texel.rgb / max(texel.a, ε)` then `mix(bg, fg, α)` implements
  `I = (w_S·C_S + Σ c_iα_i) / (w_S + Σ α_i)` — **Eq. 5**.
- Gaussian additive accumulation (`blendSrc=SrcAlpha`, `blendDst=One` for RGB;
  `One/One` for alpha) yields `Σ c_iα_i` and `Σ α_i` — **Eq. 2–3**, order-independent
  (sorting-free), matching the paper's core claim.
- Surfel flat-disc opacity (`if (g < 1/255) discard; opacity = vColor.a;` in
  `SurfelLoader.configureMaterial`) approximates the opaque disc of **Eq. 6** (opacity
  constant inside the ~3.33σ boundary).

## Deviations to fix

### 1. δ depth offset — make it a true view-space offset (Eq. 2–3)

**Current:** `SurfelLoader.configureMaterial` sets `material.polygonOffsetUnits = delta`
(≈ lines 92–94) and `setDelta` updates the same (≈ lines 149–153). The UI slider
(`#deltaSlider`, range −10…50) therefore drives GPU **polygon-offset units** — hardware
depth-buffer units scaled by an implementation-defined factor, **not** world/view depth,
and **not** per-Gaussian. The slider value does not correspond to the paper's δ.

**Target:** The depth test is `𝟙(d_i < D_S + δ)` where `d_i` and `D_S` are **view-space
depths**. Apply δ in the **Gaussian** fragment comparison, not as polygon offset:

- Add a `uDelta` uniform (shared, updated by `setDelta`) to the **Gaussian** material
  (`GaussianLoader`).
- Sample the surfel depth map `D_S` and compare in view-space depth:
  pass `d_i` (Gaussian center view-space depth, see #2) as a varying, and keep the
  fragment only if `d_i < D_S + δ`.
  - Easiest hardware route: keep using `gl_FragDepth` for the depth test, but bias the
    **surfel** depth write by a δ converted from view-space to NDC/window-depth once per
    frame (depth is nonlinear, so convert using the camera near/far, do not add δ directly
    to window-space depth). Cleaner route: do the depth test manually in the Gaussian
    shader against a sampled linear-depth surfel texture.
- Remove `polygonOffset` / `polygonOffsetUnits` usage for δ once the above is in place
  (polygon offset may still be used separately for z-fighting if needed, but not as δ).

**Higher-fidelity target (optional):** the paper uses an **adaptive per-Gaussian**
`δ_i = 5 · (Σ_{k} s_{i,k}) / d_scale` style offset tied to the Gaussian's scale (Sec. 3,
Eq. after Eq. 9 / Implementation Details), so δ varies with geometric granularity. A single
global view-space δ slider is the minimum correct step; per-Gaussian δ_i is the full match.

### 2. Depth test by Gaussian CENTER depth, not per-fragment

**Current:** `GaussianLoader` writes `gl_FragDepth = (ndcDepth + 1)/2` from per-fragment
`gl_Position.z / gl_Position.w` (≈ lines 95, 108–112). For the `@mkkellogg/gaussian-splats-3d`
splat quads the extent is added in screen space, so per-fragment z ≈ the center's clip z —
close, but not exactly the paper's `d_i` (the **center** depth).

**Target:** Pass the splat **center** view-space depth as a flat varying (compute it from
the center clip position in the vertex shader, before the screen-space quad expansion) and
use that single value for the whole splat in the depth comparison. This makes the test
exactly `𝟙(d_i < D_S + δ)` with `d_i` = center depth, matching **Eq. 2–3**.

### 3. Surfel color pass — opaque z-buffer (frontmost wins), fixed w_S = 1

**Current:**
- `SceneManager.animate` Pass 1 renders surfels with `depthWrite=false` then a separate
  depth-only Pass 2 (`depthWrite=true, colorWrite=false`) (≈ lines 164–192).
- `SurfelLoader` blends surfels premultiplied "over" (`blendSrc=One`,
  `blendDst=OneMinusSrcAlpha`, CPU-sorted) (≈ lines 84–90).
- The composite divides by the **accumulated surfel alpha**, i.e. it treats the surfel
  weight as the blended α rather than the paper's fixed `w_S = 1`.

Per the paper (Sec. 3 + Sec. 5), the post-optimization surfels are **fully opaque** and
rendered with a **z-buffer (frontmost surfel wins)**; the surfel color map `C_S` then has a
fixed weight `w_S = 1` in Eq. 5. For opaque surfels (α ≈ 1) the current approximation is
close, but translucent/edge surfels blend incorrectly and `w_S` drifts from 1.

**Target:**
- Render the surfel **color** pass with `depthTest=true, depthWrite=true` and opaque
  (non-additive) blending so the nearest surfel's color wins per pixel (true z-buffer).
  This can fold Pass 1 and Pass 2 into a single opaque pass that writes both color and
  depth.
- In the composite (`SceneManager.setupCompositing`, ≈ lines 82–103), use a **fixed
  `w_S = 1`** for pixels covered by a surfel rather than the accumulated surfel α. Track
  surfel coverage (e.g. a coverage flag / α channel from the opaque pass) so background
  pixels keep `w_S = 0`.
- Keep the Gaussian pass additive and depth-tested against this surfel depth (+ δ).

### 4. τ / Eq. 6–7 opacity-modulation UI — leave inert (correct)

The opacity-modulating parameter τ and the translucent-surfel Eq. 6–7 path are **training-time**
constructs (used to make surfel geometry differentiable during optimization). For
**post-optimization rendering** surfels are fully opaque, so the disabled/no-op τ controls
(`#opacityCapSlider`, `setOpacityCap`, `toggleBiScale`, `setUseBiScale`) are correctly inert.
No change needed beyond optionally hiding the dead UI.

## Files to touch (when applying)

- `components/viewer/SurfelLoader.ts` — δ uniform plumbing; opaque color pass; remove
  polygon-offset-as-δ; (optional) surfel coverage channel.
- `components/viewer/GaussianLoader.ts` — `uDelta` uniform; center-depth varying; manual /
  biased depth comparison.
- `components/viewer/SceneManager.ts` — single opaque surfel pass; fixed `w_S = 1` in the
  composite shader.
- `components/viewer/UIController.ts` / `index.html` — repoint the δ slider to the new
  uniform; (optional) remove dead τ controls.

## Verification (when applied)

1. `cd web && npm run dev`, open **GES Web Viewer**, load surfel + Gaussian PLYs + config.
2. Sweep the δ slider: at δ = 0 some near-surface Gaussians get truncated (color
   discontinuities, per paper Fig. 16); as δ grows, occluded Gaussians begin to leak —
   the transition should be smooth and δ should read in scene/view units.
3. Toggle "Enable Gaussian Depth Culling": with culling on, Gaussians behind opaque surfels
   are removed (no color leaking); off → SortFreeGS-style leaking.
4. Orbit the camera: result stays view-consistent (no popping) since Gaussian blending is
   additive/order-independent.
5. `npm run build` (or `tsc`) — no type errors.
