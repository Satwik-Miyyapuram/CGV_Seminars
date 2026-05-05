# Implementation Plan — CGV Seminar

**Paper.** Ye, Shao, Zhou. *When Gaussian Meets Surfel: Ultra-fast High-fidelity Radiance Field Rendering* (GES). ACM TOG 44(4), Article 113, Aug 2025. doi:10.1145/3730925.

**Course.** CS4705 / DSAIT4215 — Implementation component (40% of grade; ≈56 h budget).

---

## 1. Goal and constraints

Per `Implementation.pdf`, the deliverable is **not** a re-implementation of the paper. It must be an **educational tool** that supports the paper presentation, demonstrates substantial *custom* code, and ships as `code.zip` + 1-page PDF report + ≤3 min mp4.

I will build **`GES-Explorer`**, an interactive in-browser visualizer that opens up the parts of the paper that are genuinely hard to grasp from text alone:

1. The **two-pass, sorting-free renderer** (paper §3, Eqs. 2–5).
2. The **opacity-modulating parameter τ** that turns a translucent 2D Gaussian into an opaque disc (paper §4.1, Eqs. 6–7, Fig. 4).
3. The **depth offset δ** that decides how Gaussians get truncated by surfels (paper §5).
4. The **popping artifacts in 3DGS** vs. GES's order-independent weighted sum (paper §1, §2.4, Fig. 1, Fig. 12).
5. The **coarse-to-fine optimization process itself**, replayed from saved checkpoints so the audience can scrub through 30 K iterations of training (paper §4).
6. The **limitations** of GES, made empirically visible — what the paper *implicitly* admits by motivating Mip-/Speedy-/Compact-/2D-GES (paper §6).

This satisfies LO3 (a tool that further explains the discussed technique) and feeds LO4 (the report critically assesses what the tool reveals — including what GES still gets wrong).

---

## 2. Architectural principle: **compute once offline, replay forever in the browser**

The tool must stay snappy during a live talk and during grading. Anything expensive — full GES optimization, tile-based 3DGS reference renders, multi-view re-renders for limitations — is computed **once, ahead of time**, on my workstation, and the result is **serialized to disk**. The browser tool only *loads and renders* these precomputed artifacts; it never optimizes anything heavy at runtime.

This decision drives several other choices below (asset format, repo layout, snapshot cadence). It also lets the tool ship as a static site that runs offline from a USB stick if needed.

### 2.1 What gets precomputed

| Artifact | How produced | Used by |
|---|---|---|
| `scene.json` | Custom Python authoring tool: Poisson-disk surfel sampling on a mesh + heuristic Gaussian seeding | Modules A, B, C |
| `optim_run/iter_*.bin` | One full GES optimization run on a small scene using the **authors' code** in `external_code/`, instrumented to dump checkpoints. Resumable. | Module E (replay), Module F (limitations) |
| `popping_orbit/*.png` + `gaussian_ranks.json` | Pre-rendered camera orbit through 3DGS-style sorted blend, Hou-et-al. no-depth-test, and GES; per-frame tile-sort rank metadata | Module D |
| `tau_alt_runs/{aggressive, slow, paper}/iter_*.bin` | Three additional optimization runs with deliberately bad τ schedules | Module F |
| `failure_scenes/{thin_plate, mirror, sparse_gauss}/*.bin` | Hand-authored or quickly-fit scenes designed to expose specific GES limitations | Module F |

### 2.2 Checkpoint format and cadence

Each checkpoint is a single binary file `iter_<N>.bin` with a tiny JSON sidecar `iter_<N>.json` for fast browsing.

```
iter_05000.bin    // packed float32: surfel{p,r,s,SH,τ}, gaussian{p,σ,r,s,SH}
iter_05000.json   // {iter, τ_global, num_surfels, num_gaussians, loss_l1, loss_dssim, t_elapsed_s}
```

Cadence is non-uniform — dense around milestones the paper highlights (§4.1: "10K-th, 18K-th, 19K-th, 20K-th iteration"), sparse elsewhere:

- Every 500 iters in [0, 10 K] (densification phase)
- Every 100 iters in [9 800, 10 200] (the surfel-discard event, very visual)
- Every 100 iters in [17 800, 20 200] (τ ramp 30 → 60 → 90 → 255)
- Every 1 000 iters in [20 K, 30 K] (joint Gaussian-surfel refinement)

This gives ≈ 70 checkpoints. Per-checkpoint payload is dominated by Gaussians at later iterations — for a small scene with ~50 K Gaussians, ~3 MB per checkpoint, ~210 MB total raw. With float16 quantization for SH and zstd compression on the binary, this fits well under 50 MB total — comfortable inside the 200 MiB submission budget. The browser streams them lazily.

### 2.3 Resumable offline computation

The optimization driver (`tools/run_optim.py`, custom code wrapping the authors' optimizer) implements:

- **Checkpoint after every snapshot** — not just at exit. If a run crashes at iter 12 437, the next launch reads the highest-numbered `iter_*.bin` and resumes from there. No iteration is ever redone.
- **`--resume` and `--from-iter N` flags** for explicit control.
- **Atomic writes** (`tmp` + `os.replace`) so an interrupted write never leaves a half-file.
- **A `manifest.json` per run** logging: dataset, hyperparameters, git SHA of authors' code, total wall-time per phase, GPU model. Reproducibility for the report.
- **Hash-keyed cache** for sub-results that are stable across runs (e.g., SfM init), so re-running with a tweaked τ schedule re-uses the same init.

This means I can iterate on the visualizer for days without ever waiting on a 2-hour optimization run again, and a power outage on hour 1.5 of a 2-hour run loses at most one checkpoint interval.

### 2.4 Browser side: lazy streaming + LRU

The viewer never holds all checkpoints in memory. Module E's timeline scrubber:

- Fetches a checkpoint on demand via `fetch()`.
- Decodes (decompress → typed-array view → upload to GPU buffers).
- Keeps an LRU cache of ~5 decoded checkpoints to make scrubbing fluid.
- Pre-fetches the next checkpoint in the playback direction.

This is custom code and fully testable.

---

## 3. Why a browser tool? (technical choice)

**Stack: TypeScript + three.js + custom GLSL shaders + Vite for the viewer; Python + PyTorch for the offline optimizer driver.**

Justifications:

- **One URL, no install.** Critical for an educational tool used live during a presentation and reviewed by a grader. Static `dist/` ships in the ZIP.
- **Custom code is unambiguous.** All GLSL shaders for both render passes, the checkpoint loader, the UI, the asset generator, and the optimization-driver wrapper are written by me. three.js is used only as a thin scene-graph + WebGL2 wrapper and is isolated under `external_code/`. The authors' optimizer lives under `external_code/` and is not graded; my driver and instrumentation around it *is* custom and is graded.
- **WebGL2 has what we need.** MRT for separating numerator/denominator buffers (Eq. 5), custom blend modes for additive accumulation (Eq. 2/3), depth-pre-pass + sampler for the depth test against the surfel depth map.
- **Mobile-friendly.** The paper itself emphasizes that GES does not need compute shaders; mirroring that constraint is on-message.

---

## 4. Scope — in vs. out

| In scope | Out of scope |
|---|---|
| Interactive viewer rendering precomputed GES scenes | Live optimization in the browser |
| The two-pass pipeline (surfel rasterization → Gaussian splatting with depth test, Eqs. 1–5) | A faithful CUDA tile-based splatting reproduction at production speed |
| Live τ explorer (Eqs. 6–7, Fig. 4) | Mip-/Speedy-/Compact-/2D-GES retraining |
| Live δ explorer (paper §5, last paragraph) | Reproducing the paper's full benchmark tables |
| Side-by-side popping comparison driven by precomputed orbit renders | Real-time 3DGS rasterization for arbitrary scenes |
| **Optimization replay**: scrub through 30 K iterations from checkpoints | Authoring new datasets |
| **Limitations gallery**: failure-case scenes with side-by-side commentary | A general-purpose GES viewer for arbitrary `.ply` files |

---

## 5. Module-by-module design

The tool ships as a single web app with six linked modules accessible from a left-side nav. Each module corresponds to a specific paper section/figure.

### Module A — Representation Inspector (paper §3, Fig. 3)

3D viewer with a precomputed scene. Click any primitive → side panel shows raw parameters and Eq. 1 evaluation in the current view. Toggles: surfels only, Gaussians only, both (reproduces Fig. 2). SH-degree selector (0–3) for surfels and Gaussians independently.

### Module B — The Two-Pass Pipeline (paper §3, Eqs. 2–5)

Stepper UI with four frames: Pass 1 (surfel rasterization), Pass 1.5 (depth-map post-process applying δ), Pass 2 (Gaussian splatting, additive blend into MRT: numerator = Σ α·c, denominator = Σ α), Composite (Eq. 5: `C = (γ·C_S + C_G) / (γ + W_G)`, γ = 1). Each frame shows the relevant intermediate buffer. Pixel-hover formula readout substitutes actual pixel values into Eqs. 2, 3, 5.

### Module C — Order-Independence Proof (paper §3, last paragraph)

A "shuffle Gaussian draw order" button re-renders Pass 2 with a permuted order; the pixel-difference image against the previous render is shown beside it (≈ machine-epsilon, only floating-point reordering). A "force sorted by depth" toggle confirms visual identity, making Eq. 5's permutation-invariance falsifiable in real time.

### Module D — Popping vs. View-Consistency (paper §1, §2.4, Fig. 12)

Three viewports playing the **precomputed** orbit:

1. **3DGS-style** tile-based center-depth sort + sorted α-blending.
2. **Hou-et-al-style** sorting-free weighted sum without depth test (color leakage).
3. **GES** (sorting-free + depth test).

The orbit was generated offline so the demo is always smooth on any laptop. Per-frame overlays highlight Gaussians whose tile-sort rank just changed (this is what causes 3DGS to pop). "Step 1 frame" mode lets a presenter pause exactly on the popping frame.

### Module E — Parameter Explorers + **Optimization Replay**

This module gets the most pedagogical depth.

- **τ explorer (Eqs. 6–7, Fig. 4).** Slider τ ∈ [0.1, 255], log scale. Side-by-side: 2D heatmap of α(x,y) for current τ, and a 1-D radial overlay at τ = 0.1, 30, 90, 255 (live-rendered Fig. 4). A small **toy 1-D optimization sandbox** (custom NumPy code, runs in the browser via Pyodide *or* pre-baked frames if Pyodide is too heavy) plots the loss `L(p) = (render(p) − target)²` and its gradient as a surfel translates along x, demonstrating that at τ = 255 the gradient is zero almost everywhere with a step at the edge (paper §4.1's non-differentiability claim) while at τ = 0.1 it is smooth.

- **δ explorer (paper §5, last paragraph).** Slider δ-multiplier on top of `δ_i = √(5 · Σ s²_{i,j}/d)`. Sweep shows leakage at large δ vs. truncation at small δ. Reference at the paper's recommended δ pinned for comparison.

- **Optimization replay.** Timeline scrubber across the 70-checkpoint sequence. Plays back the actual GES training. UI elements:
  - Loss curve (L1 + D-SSIM) with a vertical playhead.
  - Surfel count and Gaussian count over time, with annotations at the milestone iterations the paper highlights.
  - Live render of the scene at the scrubbed iteration.
  - Toggle: tint surfels by their current τ value (red → blue as τ ramps up). Makes the τ schedule visible as a propagating wave through the surfel set.
  - Side panel: per-iteration JSON sidecar (loss, num primitives, elapsed time).
  - "Jump to milestone" buttons: iter 10 K (surfel discard event), iter 18/19/20 K (τ ramp), iter 20 K → 30 K (Gaussian densification).

This replay is the single most pedagogically valuable feature: it turns paper §4 from prose into a movie.

### Module F — Limitations Gallery (new)

The paper presents GES as a near-uniformly winning method. Pedagogically that's a half-truth — the existence of Mip-/Speedy-/Compact-/2D-GES is itself an admission of basic-3D-GES limitations. This module surfaces those honestly.

Each item is a side-by-side: "what GES does" / "what fails / what the extension fixes" / one-paragraph annotation citing the paper.

1. **Aliasing without Mip-GES (motivates §6.1).** Render the same scene at native and 4× downsampled resolutions; sparse Gaussians produce visible flicker / dilation artifacts in the basic 3D-GES.
2. **Storage cost without Compact-GES (motivates §6.3).** Show the scene size in MB for the basic representation vs. the paper's reported 20× compression — quantitative bar chart, no false PSNR claims.
3. **Geometry discontinuities at surfel intersections (motivates §6.4 / 2D-GES).** A close-up of a corner where two surfels intersect; the depth/normal discontinuity is rendered as a heatmap. Toggle to a 2D-Gaussian replacement (using a checkpoint from a 2D-GES alt run) shows the smoothing.
4. **τ-schedule failure mode.** Replay the **aggressive τ schedule** alt run from `tau_alt_runs/aggressive/`: surfels jump to opaque before they have settled, optimization plateaus at high loss. Demonstrates *why* the paper's specific schedule (10 K / 18 K / 19 K / 20 K) is not arbitrary.
5. **δ has no globally-good value.** A scene with both very thin foreground (truncation-prone) and a glossy back wall (leakage-prone). Slider sweep shows there is no δ that fixes both.
6. **View extrapolation.** Render from camera positions deliberately outside the training-camera convex hull. Quality degrades; the report can connect this to the surfel-only coarse pass falling back to its baked SH at unseen angles.

Items 4–6 use the optimization-replay infrastructure from Module E; items 1–3 are precomputed image pairs with overlays. So Module F is mostly *new content*, not new infrastructure — keeps cost manageable.

This directly hits LO4: critical assessment.

---

## 6. Asset and snapshot generation (offline, Python)

`tools/build_scene.py` — custom: Poisson-disk surfel sampling on a low-poly mesh + heuristic Gaussian seeding at high-curvature regions. Exports `scene.json`.

`tools/run_optim.py` — custom driver wrapping the authors' optimizer (in `external_code/ges-author-code/`). Adds: checkpoint cadence schedule from §2.2, atomic writes, manifest logging, `--resume` / `--from-iter`, `--tau-schedule {paper, aggressive, slow}` for the alt runs Module F needs.

`tools/render_orbit.py` — custom: takes a checkpoint, renders a smooth orbit through three modes (3DGS tile-sort, Hou-style, GES) into PNGs, and dumps per-frame Gaussian tile-sort ranks for Module D's overlays.

`tools/pack_for_web.py` — custom: takes the raw `iter_*.bin` checkpoints, applies float16 SH quantization + zstd compression, and emits a `manifest.web.json` index the browser can stream.

All four scripts are checkpoint-aware: each can be killed and resumed.

---

## 7. Repository / submission layout (per Implementation.pdf §1)

```
ges-explorer/
├── README.md                 # setup + run command + external resources list
├── package.json              # custom
├── vite.config.ts            # custom
├── src/                      # CUSTOM CODE (graded)
│   ├── main.ts
│   ├── modules/{A,B,C,D,E,F}/
│   ├── shaders/              # all GLSL written by me
│   ├── ges/                  # SH eval, projection, accumulation, Eq. 5 composite
│   ├── checkpoints/          # streaming loader, LRU cache, decoder
│   └── ui/                   # HUDs, sliders, formula overlays, timeline scrubber
├── tools/                    # CUSTOM CODE (graded)
│   ├── build_scene.py
│   ├── run_optim.py
│   ├── render_orbit.py
│   └── pack_for_web.py
├── data/                     # PRECOMPUTED ARTIFACTS (small subset shipped)
│   ├── scenes/scene.json
│   ├── optim_run/            # ~30 MB after compression
│   ├── tau_alt_runs/{aggressive,slow}/   # decimated, ~5 MB each
│   ├── popping_orbit/        # PNGs + ranks.json, ~20 MB
│   └── failure_scenes/
├── external_code/            # NOT graded; isolated per §1
│   ├── three.js/             # vendored, with LICENSE
│   └── ges-author-code/      # vendored authors' optimizer, with LICENSE + citation
├── report.pdf                # 1 page text + 1 page figures
└── video.mp4                 # ≤ 3 min, H.264
```

For the BS submission, only a *minimal* `data/` subset is shipped (one canonical scene, decimated checkpoints — every 5th — and a single short popping orbit). The README gives a `download_full_data.sh` link to the full dataset on a public bucket so a grader can opt in. Total ZIP target: ≤ 150 MiB.

---

## 8. Phases and time budget

Total budget 56 h. Plan uses 54 h, leaving 2 h micro-slack; the two big risk pads (WebGL OIT, Module D popping) absorb most of any overrun internally.

| # | Phase | Hours | Output |
|---|---|---:|---|
| 0 | Setup, paper deep-read, equation transcription, repo + Vite + `external_code/` skeleton | 4 | repo skeleton, notes.md |
| 1 | Asset generation (`build_scene.py`) → `scene.json` | 4 | toy scene |
| 2 | **Optimization driver `run_optim.py`** with checkpoint cadence, atomic writes, resume, manifest. Vendor + smoke-test authors' code. | 5 | resumable driver |
| 3 | **Run the optimization once for ~2 h on real data** while implementing Module A in parallel. Babysit the run; if it fails, resume. Produce paper, aggressive, and slow τ checkpoints in sequence. | 5 (overlaps with Module A) | `optim_run/` + `tau_alt_runs/` |
| 4 | Module A (representation inspector) + Module B Pass 1 (surfel rasterizer + depth pre-pass) | 6 | A done, B half |
| 5 | Module B Pass 2 (Gaussian splatter, MRT additive blend, composite per Eq. 5) | 7 | B done |
| 6 | `pack_for_web.py` + browser-side checkpoint loader / LRU / streaming | 4 | replay infra |
| 7 | Module C (order-independence proof) | 2 | C done |
| 8 | `render_orbit.py` + Module D (popping side-by-side, animated overlay) | 5 | D done |
| 9 | Module E τ explorer + 1-D toy optimization sandbox | 3 | partial E |
| 10 | Module E optimization-replay timeline scrubber | 3 | E done |
| 11 | Module E δ explorer | 2 | E done |
| 12 | Module F (limitations gallery; reuses replay infra) | 5 | F done |
| 13 | Polish, equation overlays, in-app captions, perf pass, decimation for ZIP | 2 | shippable |
| 14 | Report (1 page text + 1 page figures) and ≤ 3-min video | 5 | report.pdf, video.mp4 |
|   | **Total** | **54** | |

Critical path notes:

- Phase 3 (the offline optimization run) overlaps with Phase 4 (Module A coding) because the run is unattended after kickoff. This is the single biggest schedule efficiency.
- Phase 6 (checkpoint loader) is on the critical path for Modules E-replay and F; if it slips, those modules slip with it.
- Phase 12 (Limitations) reuses Phase 6 infrastructure, so its 5 h is all *content*, not infrastructure.

---

## 9. How a presenter uses this during the 25-min talk

Each module URL is a permalink. Concrete script:

- Slide on bi-scale representation → `#/A` → toggle surfel-only / Gaussian-only / both. Mirrors Fig. 2.
- Slide on rendering pipeline → `#/B` → walk Pass 1 → 1.5 → 2 → Composite. Hover a pixel; formula overlay reads out Eq. 5 with that pixel's numbers.
- Slide claiming "sorting-free" → `#/C` → click "shuffle"; show diff ≈ 0.
- Slide on popping → `#/D` → play orbit. Pause on the popping frame.
- Slide on optimizing opaque surfels → `#/E/tau` → sweep τ, then switch to the 1-D loss sandbox.
- Slide on the depth offset → `#/E/delta`.
- Slide on coarse-to-fine training → `#/E/replay` → scrub from iter 0 to 30 K, paying special attention to the iter-10 K surfel-discard event and the τ ramp at 18–20 K.
- Honest closing slide on limitations → `#/F` → cycle through aliasing / storage / discontinuities / bad-τ failure / no-good-δ scene / view extrapolation.

---

## 10. Educational value (mapped to the paper)

| Concept that is hard from the paper alone | What the tool does | Paper anchor |
|---|---|---|
| Why GES looks like *one* image instead of *two* | Module B animates the composite from its parts | §3, Eq. 5 |
| Why "sorting-free" is OK | Module C: empirical permutation test | §3 last ¶ |
| Why 3DGS pops at all | Module D: animated camera + per-frame rank-flip overlay | §1, §2.4, Fig. 12 |
| Why a sorting-free predecessor (Hou et al.) leaks color | Module D middle viewport | §2.4 |
| Why optimizing an opaque surfel is hard, and why τ fixes it | Module E τ + 1-D sandbox | §4.1, Eqs. 6–7, Fig. 4 |
| Why δ matters (and isn't a free parameter) | Module E δ | §5 last ¶ |
| What 30 K iterations of GES training actually look like | Module E replay timeline | §4 |
| Why the paper needs Mip-/Speedy-/Compact-/2D-GES | Module F gallery | §6 |
| Where GES's τ schedule is fragile | Module F item 4 (aggressive-τ alt run) | §4.1 |

The report (1 page text + 1 page figures) writes up exactly this table plus a "what we found" critical assessment, addressing LO4.

---

## 11. Risks and mitigations

- **WebGL2 order-independent accumulation is fiddly.** *Mitigation:* prototype MRT additive blend in Phase 5; CPU-side reference splatter for sanity checks; budget pad inside Phase 5.
- **Authors' code may not run cleanly on my GPU/CUDA.** *Mitigation:* Phase 2 includes a smoke test before committing to a long run; if hopeless, fall back to a smaller dataset or a published checkpoint, and skip the alt-τ runs (Module F items 4 still illustrative via cherry-picked early checkpoints).
- **Snapshot storage exceeds 200 MiB.** *Mitigation:* aggressive float16 SH quantization + zstd; ship every-5th checkpoint in the ZIP and full set via download script.
- **Long optimization run loses progress.** *Mitigation:* the resumable checkpoint loop in §2.3 means any crash costs at most one cadence interval (≤ 1000 iters ≈ a few minutes).
- **3DGS-style tile sort in WebGL is non-trivial.** *Mitigation:* Module D uses **precomputed PNG orbits**, not live 3DGS rasterization. Tile-sort happens once offline in Python.
- **Time overrun.** Hard rank by educational value: B > E-replay > E-τ > F > D > C > A > E-δ. Trim from the bottom; the 2 h slack + risk pads inside phases handle most.
- **Pyodide too heavy for the τ sandbox.** *Mitigation:* fall back to pre-baking ~50 evaluation frames of `(L, dL/dp)` vs. p for τ ∈ {0.1, 1, 30, 90, 255} as static JSON; a custom JS plotter renders them. Still custom code, no compute-in-browser.

---

## 12. What I will *not* claim in the report

To stay honest (per Implementation.pdf "minimal functional requirements"):

- The tool does not produce photoreal reconstructions; it visualizes precomputed ones.
- The 3DGS comparison in Module D uses a CPU-side tile sort — mechanism-faithful, performance-not-faithful.
- Numbers in the report are illustrative, not benchmark replications. Quantitative claims (e.g., "20× compression motivates Compact-GES") are taken from the paper and cited, not re-measured.
- The optimization replay uses **the authors' optimizer** (in `external_code/`) — what is custom is the driver, the checkpointing infrastructure, the alt-τ schedules, the packing pipeline, and the entire browser-side replay UI. The report makes this split explicit.
