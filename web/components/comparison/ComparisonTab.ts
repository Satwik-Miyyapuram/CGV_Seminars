import "./comparison.css";
import { ComparisonManager } from "./ComparisonManager";
import { card, slider, statusBox, diagramCanvas, CardSpec } from "./markup";

/**
 * ComparisonTab owns the "Artifact Comparison" tab (Tab B). Its job is small:
 *
 *   1. inject the tab's markup into #comparison-container,
 *   2. typeset the maths with KaTeX,
 *   3. once the tab is actually visible, hand off to ComparisonManager, which
 *      constructs the canvas diagrams and the three.js viewers against the ids
 *      this markup creates.
 *
 * The *content* (the narrative cards) is declared as data in `cards()` below and
 * rendered through the pure builders in `markup.ts`, so this file stays readable
 * and the per-card HTML lives in one place each.
 */
export class ComparisonTab {
    private manager: ComparisonManager | null = null;

    /** Inject the tab markup, typeset the maths, and build the diagrams/viewers once visible. */
    constructor() {
        const container = document.getElementById("comparison-container");
        if (!container) {
            console.error("[ComparisonTab] #comparison-container not found");
            return;
        }
        container.innerHTML = this.markup();
        this.renderMath(container);

        // Build the diagrams only once the tab is actually visible (non-zero size). Several
        // diagrams compute sizes from the canvas and would throw if constructed at zero width
        // (which would abort the whole init), so we defer until the container has a real size.
        if (container.clientWidth > 0) {
            this.init();
        } else {
            const ro = new ResizeObserver(() => {
                if (container.clientWidth > 0 && !this.manager) {
                    this.init();
                    ro.disconnect();
                }
            });
            ro.observe(container);
        }
    }

    /** Build the diagrams and 3D viewers against the injected markup. */
    private init() {
        this.manager = new ComparisonManager();
    }

    /** Forward a resize to the diagrams/viewers so their canvases re-measure. */
    public handleResize() {
        this.manager?.handleResize();
    }

    /** Re-run KaTeX over freshly-injected markup (the global auto-render already ran on load). */
    private renderMath(root: HTMLElement) {
        const render = () => {
            const fn = (window as any).renderMathInElement;
            if (typeof fn === "function") {
                fn(root, {
                    delimiters: [
                        { left: "$$", right: "$$", display: true },
                        { left: "$", right: "$", display: false },
                    ],
                });
                return true;
            }
            return false;
        };
        // KaTeX's contrib script is loaded with `defer`; retry briefly if not ready yet.
        if (!render()) {
            let tries = 0;
            const id = setInterval(() => {
                if (render() || ++tries > 20) clearInterval(id);
            }, 100);
        }
    }

    // ---- Page structure --------------------------------------------------

    private markup(): string {
        return `
        <div class="cmp-root">
            ${this.header()}
            ${card(this.introCard())}
            <div class="cmp-stack">
                ${this.cards().map(card).join("\n")}
            </div>
        </div>`;
    }

    /** Build the tab header: title, subtitle, and the colour legend. */
    private header(): string {
        const chip = (color: string, label: string) =>
            `<span class="legend-chip"><span class="legend-dot" style="background:var(${color})"></span>${label}</span>`;
        return `
        <div class="cmp-header">
            <div>
                <h1 class="cmp-title">GES vs 3DGS — Artifact Explorer</h1>
                <div class="cmp-sub">Why Gaussian-enhanced Surfels render fast, view-consistent images — and where 3DGS breaks.</div>
            </div>
            <div class="cmp-legend">
                ${chip("--cmp-3dgs", "3DGS")}
                ${chip("--cmp-ges", "GES")}
                ${chip("--cmp-surfel", "Surfel (opaque)")}
                ${chip("--cmp-gauss", "Gaussian (detail)")}
            </div>
        </div>`;
    }

    // ---- Cards (declarative content) ------------------------------------

    /** Card 0 — the two-pass overview schematic. */
    private introCard(): CardSpec {
        return {
            num: "0",
            cardClass: "intro",
            title: "How GES renders — two passes, no sorting",
            take: `GES draws an <b style="color:var(--cmp-surfel)">opaque surfel</b> surface first (giving a colour map and a depth map), then <b style="color:var(--cmp-gauss)">adds Gaussian</b> detail on top with a depth test — and <b>never sorts</b>. That's what makes it fast and pop-free.`,
            body: this.introFlow(),
            math: `Final image combines the surfel colour map and the normalized Gaussian sum:
                    <div class="row">$I = \\dfrac{w_S\\,C_S + \\sum_i \\mathbb{1}(d_i < D_S + \\delta)\\,c_i\\alpha_i}{w_S + \\sum_i \\mathbb{1}(d_i < D_S + \\delta)\\,\\alpha_i}$</div>
                    with fixed surfel weight $w_S = 1$. The Gaussian sum is order-independent, so no sorting is needed.`,
        };
    }

    /** The bespoke INPUT → INTERMEDIATE → OUTPUT flow schematic shown in the intro card. */
    private introFlow(): string {
        return `
            <div class="flow">
                <div class="flow-stage">
                    <div class="flow-k">INPUT</div>
                    <div class="flow-box surfel">2D Surfels</div>
                    <div class="flow-box gauss">3D Gaussians</div>
                </div>
                <div class="flow-arrow">→<div class="flow-op">Pass 1 — rasterize + z-buffer</div></div>
                <div class="flow-stage">
                    <div class="flow-k">INTERMEDIATE</div>
                    <div class="flow-box">Colour map <span class="dim">C&#8347;</span></div>
                    <div class="flow-box">Depth map <span class="dim">D&#8347;</span></div>
                </div>
                <div class="flow-arrow">→<div class="flow-op">Pass 2 — splat · depth-test (δ) · add</div></div>
                <div class="flow-stage">
                    <div class="flow-k">INTERMEDIATE</div>
                    <div class="flow-box">Σ cᵢαᵢ &nbsp;·&nbsp; Σ αᵢ</div>
                </div>
                <div class="flow-arrow">→<div class="flow-op">normalize (Eq. 5)</div></div>
                <div class="flow-stage">
                    <div class="flow-k">OUTPUT</div>
                    <div class="flow-box out">Final image</div>
                </div>
            </div>`;
    }

    /** Cards 1–4 plus the live demo, in display order. */
    private cards(): CardSpec[] {
        return [
            // Live 3D side-by-side (SideBySide3DViewer) — first card in the column.
            {
                num: "▶",
                numClass: "live",
                title: "Live 3D — the two-pass renderer",
                take: `Orbit the splats. The cyan <b style="color:var(--cmp-surfel)">surfel</b> writes depth and <b>blocks</b> Gaussians behind it (GES, right). 3DGS (left) re-sorts every frame, so its colours <span class="red">pop</span> as you move.`,
                flow: [
                    [{ kind: "in", label: "input", value: "orbit camera" }],
                    [{ kind: "mid", label: "intermediate", value: "surfel depth" }],
                    [{ kind: "out", label: "output", value: "composited frame" }],
                ],
                body: this.liveViewports(),
                caption: `The debug readout lists each splat's view-space depth and the current draw order. When 3DGS's order flips, its blend jumps (popping). On the GES side a ✗ marks Gaussians the surfel's depth test culls.`,
            },

            // ① Popping under view change (RayCompositingDiagram).
            {
                num: "1",
                title: `Orbit the camera → 3DGS "pops"`,
                take: `3DGS sorts splats by depth, then alpha-blends. As you orbit, that global order <b>flips all at once</b> and the colour jumps — the <span class="red">popping</span> artifact. GES just <b>adds</b> the splats up, so the result never jumps.`,
                flow: [
                    [{ kind: "in", label: "input", value: "camera angle" }],
                    [
                        { kind: "out", label: "output", value: "3DGS strip" },
                        { kind: "out", label: "output", value: "GES strip" },
                    ],
                ],
                body:
                    slider({
                        id: "orderSlider",
                        label: "Drag to orbit the camera",
                        valueId: "orderValue",
                        valueText: "0°",
                        min: "-45", max: "45", step: "1", value: "0",
                    }) +
                    diagramCanvas("ray-canvas"),
                caption: `Top = a top-down view of the camera orbiting the splats. Bottom strips = the rendered pixels: watch the <span class="red">3DGS</span> strip's colours snap as the sort order flips, while the <span class="green">GES</span> strip stays smooth.`,
                math: `<div class="row"><span class="lbl red">3DGS</span> $C = \\sum_i c_i\\alpha_i \\prod_{j<i}(1-\\alpha_j)$ &nbsp;— order-dependent (front-to-back)</div>
                    <div class="row"><span class="lbl green">GES</span> $C = \\dfrac{\\sum_i c_i\\alpha_i}{\\sum_i \\alpha_i}$ &nbsp;— a sum, so order doesn't matter</div>`,
            },

            // ② Why it pops — tile membership (TileBoundaryDiagram).
            {
                num: "2",
                title: "Why it happens: tiles drop splats",
                take: `3DGS chops the screen into <b>tiles</b>; each tile only draws the blobs assigned to it. Slide blob <b style="color:var(--cmp-surfel)">B</b> across a tile border: when a tile drops it, the two tiles disagree and a hard <span class="red">seam</span> appears. GES has no tiles → no seam.`,
                flow: [
                    [{ kind: "in", label: "input", value: "blob B position" }],
                    [
                        { kind: "out", label: "output ①", value: "3DGS tiled" },
                        { kind: "out", label: "output ②", value: "GES no-tiles" },
                    ],
                ],
                body:
                    slider({
                        id: "tileSplatSlider",
                        label: "Slide blob B across the tile border",
                        valueId: "tileSplatValue",
                        valueText: "in Tile 1",
                        min: "0", max: "100", step: "1", value: "80",
                    }) +
                    statusBox("tileStatus") +
                    diagramCanvas("tile-canvas"),
                caption: `Each row shows the same two blobs. Top = 3DGS (two tiles); the result bar gets a <span class="red">hard seam</span> where a tile skips B. Bottom = GES (one pass); the same blobs stay <span class="green">smooth</span>.`,
            },

            // ③ Depth test → no colour leaking (ColorLeakingDiagram).
            {
                num: "3",
                title: "Depth test → no colour leaking",
                take: `A bright Gaussian hidden <b>behind</b> the surface shouldn't be visible. GES culls it with a depth test against the surfel depth (with a small offset <b>δ</b>); sort-free methods blend it anyway and <span class="red">leak</span> its colour through.`,
                flow: [
                    [{ kind: "in", label: "input", value: "depth offset δ" }],
                    [
                        { kind: "out", label: "output", value: "SortFreeGS (leaks)" },
                        { kind: "out", label: "output", value: "GES (culled)" },
                    ],
                ],
                body:
                    slider({
                        id: "compDeltaSlider",
                        label: "Depth-test offset δ",
                        valueId: "compDeltaValue",
                        valueText: "0.5",
                        min: "0.0", max: "1.5", step: "0.05", value: "0.5",
                    }) +
                    diagramCanvas("leaking-canvas"),
                caption: `A cyan <b style="color:var(--cmp-surfel)">surfel</b> sits in front of a red Gaussian. Left = SortFreeGS always shows the red <span class="red">leak</span>. Right = GES culls the red when it's deeper than the surfel + δ (<span class="green">clean</span>). Push δ too high and even GES starts to leak.`,
                math: `The indicator $\\mathbb{1}(d_i < D_S + \\delta)$ keeps a Gaussian only if its depth $d_i$ is in front of the surfel depth $D_S$ (plus offset):
                    <div class="row">$C = \\dfrac{w_S C_S + \\sum_i \\mathbb{1}(d_i < D_S + \\delta)\\,c_i\\alpha_i}{w_S + \\sum_i \\mathbb{1}(d_i < D_S + \\delta)\\,\\alpha_i + \\epsilon}$</div>`,
            },

            // ④ Limitation — specular reflections (Reflection3DViewer).
            {
                num: "4",
                title: "Honest limitation: specular reflections",
                take: `Both methods light the scene with a <b style="color:var(--cmp-gauss)">3D Gaussian</b> light source. 3DGS represents the mirror as a <b style="color:var(--cmp-gauss)">flat translucent Gaussian</b> and fakes the reflection with a second Gaussian placed <b>underground</b> — the mirror image of the light, seen through the translucent surface. GES's mirror is an <b style="color:var(--cmp-surfel)">opaque surfel</b> (a flat, opaque Gaussian), which blocks any underground floater. So GES must fake the reflection with a <b>view-dependent Gaussian highlight</b> sitting directly on top of the surfel.`,
                flow: [
                    [{ kind: "in", label: "input", value: "camera position" }],
                    [
                        { kind: "out", label: "output", value: "3DGS (underground floater)" },
                        { kind: "out", label: "output", value: "GES (view-dependent highlight)" },
                    ],
                ],
                body: this.reflectionContainer() + this.reflectionWipe(),
                caption: `Orbit the scene. 3DGS (left) renders the reflection as a static underground Gaussian — the true mirror image of the light — visible through the translucent mirror, so it stays view-stable. GES (right) can't do that: the opaque surfel blocks the floater, so the reflection is a Gaussian on top of the surface whose opacity and colour shift with the camera (Fresnel: stronger and warmer at grazing angles). Below: the truck rendered along one smooth camera path — drag the divider to wipe between <b>3DGS</b> (left) and <b>GES</b> (right), and watch how each handles the view-dependent specular highlight on the windshield as the camera orbits.`,
            },
        ];
    }

    // ---- Bespoke card bodies (verbatim DOM the viewers attach to) --------

    /** Live side-by-side viewports (SideBySide3DViewer attaches to these ids). */
    private liveViewports(): string {
        return `
            <div class="sbs-viewport-container">
                <div class="sbs-viewport" id="viewport-3dgs">
                    <div class="viewport-label" style="color:var(--cmp-3dgs); border-color:var(--cmp-3dgs);">3DGS · alpha blend (sorted)</div>
                    <div id="debug-3dgs" style="position:absolute; bottom:10px; left:10px; font-family:monospace; font-size:10px; color:#fff; background:rgba(0,0,0,0.8); padding:4px 6px; border-radius:4px; z-index:2; pointer-events:none;"></div>
                </div>
                <div class="sbs-viewport" id="viewport-ges">
                    <div class="viewport-label" style="color:var(--cmp-ges); border-color:var(--cmp-ges);">GES · two-pass depth test</div>
                    <div id="debug-ges" style="position:absolute; bottom:10px; left:10px; font-family:monospace; font-size:10px; color:#fff; background:rgba(0,0,0,0.8); padding:4px 6px; border-radius:4px; z-index:2; pointer-events:none;"></div>
                </div>
            </div>`;
    }

    /** Reflection viewer mount (Reflection3DViewer attaches to #reflection-container). */
    private reflectionContainer(): string {
        return `
            <div id="reflection-container" style="position:relative; border-radius:8px; overflow:hidden; border:1px solid rgba(255,255,255,0.05); background:#0d0c14;">
                <div style="position:absolute; left:10px; top:10px; color:var(--cmp-3dgs); font-size:11px; font-weight:bold; background:rgba(0,0,0,0.5); padding:4px 6px; border-radius:4px; z-index:10;">3DGS — alpha blend</div>
                <div style="position:absolute; right:10px; top:10px; color:var(--cmp-ges); font-size:11px; font-weight:bold; background:rgba(0,0,0,0.5); padding:4px 6px; border-radius:4px; z-index:10;">GES — opaque surfel</div>
            </div>`;
    }

    /** 3DGS-vs-GES video wipe (VideoWipeViewer attaches to #wipe-container). */
    private reflectionWipe(): string {
        return `
            <div id="wipe-container" style="margin-top:14px;">
                <video id="wipe-3dgs" src="/truck_3dgs.mp4" muted loop playsinline></video>
                <div class="wipe-clip"><video id="wipe-ges" src="/truck_ges.mp4" muted loop playsinline></video></div>
                <div class="wipe-handle"></div>
                <span class="wipe-label l3dgs">3DGS</span>
                <span class="wipe-label lges">GES</span>
            </div>`;
    }
}
