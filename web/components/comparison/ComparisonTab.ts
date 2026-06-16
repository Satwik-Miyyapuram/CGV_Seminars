import "./comparison.css";
import { ComparisonManager } from "./ComparisonManager";

/**
 * ComparisonTab builds the redesigned "Artifact Comparison" tab (Tab B) and wires up the
 * existing diagram/viewer logic.
 *
 * Design goals (see the card sequence below):
 *   - A guided, numbered narrative instead of disconnected panels.
 *   - A consistent visual system + fixed colour legend (3DGS=red, GES=green, surfel=cyan, gaussian=magenta).
 *   - Every visual is labelled with what's the INPUT, the INTERMEDIATE, and the OUTPUT.
 *   - Plain-English takeaways up front; the maths tucked into a collapsible "show the maths".
 *   - 2-column layout kept: concept cards on the left, the live 3D demo as a sticky hero on the right.
 *
 * The markup re-creates the element ids the diagram classes expect (ray-canvas, orderSlider,
 * tile-canvas, tileSplatSlider, tileStatus, leaking-canvas, compDeltaSlider, reflection-container,
 * viewport-3dgs/ges, debug-3dgs/ges), then ComparisonManager constructs the diagrams against them.
 */
export class ComparisonTab {
    private manager: ComparisonManager | null = null;

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

    /** Wire the (unchanged) diagram + 3D viewer logic against the new markup. */
    private init() {
        this.manager = new ComparisonManager();
    }

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

    private markup(): string {
        return `
        <div class="cmp-root">
            ${this.header()}
            ${this.intro()}
            <div class="cmp-grid">
                <div class="cmp-col cmp-col-left">
                    ${this.cardPopping()}
                    ${this.cardTiles()}
                    ${this.cardLeaking()}
                    ${this.cardSpecular()}
                </div>
                <div class="cmp-col cmp-col-right">
                    ${this.cardLive()}
                </div>
            </div>
        </div>`;
    }

    private header(): string {
        return `
        <div class="cmp-header">
            <div>
                <h1 class="cmp-title">GES vs 3DGS — Artifact Explorer</h1>
                <div class="cmp-sub">Why Gaussian-enhanced Surfels render fast, view-consistent images — and where 3DGS breaks.</div>
            </div>
            <div class="cmp-legend">
                <span class="legend-chip"><span class="legend-dot" style="background:var(--cmp-3dgs)"></span>3DGS</span>
                <span class="legend-chip"><span class="legend-dot" style="background:var(--cmp-ges)"></span>GES</span>
                <span class="legend-chip"><span class="legend-dot" style="background:var(--cmp-surfel)"></span>Surfel (opaque)</span>
                <span class="legend-chip"><span class="legend-dot" style="background:var(--cmp-gauss)"></span>Gaussian (detail)</span>
            </div>
        </div>`;
    }

    /** Two-pass overview schematic with INPUT / INTERMEDIATE / OUTPUT labels. */
    private intro(): string {
        return `
        <div class="ccard intro">
            <div class="ccard-head">
                <div class="ccard-num">0</div>
                <h3 class="ccard-title">How GES renders — two passes, no sorting</h3>
            </div>
            <p class="ccard-take">GES draws an <b style="color:var(--cmp-surfel)">opaque surfel</b> surface first (giving a colour map and a depth map), then <b style="color:var(--cmp-gauss)">adds Gaussian</b> detail on top with a depth test — and <b>never sorts</b>. That's what makes it fast and pop-free.</p>
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
            </div>
            <details class="math">
                <summary>show the maths</summary>
                <div class="math-body">
                    Final image combines the surfel colour map and the normalized Gaussian sum:
                    <div class="row">$I = \\dfrac{w_S\\,C_S + \\sum_i \\mathbb{1}(d_i < D_S + \\delta)\\,c_i\\alpha_i}{w_S + \\sum_i \\mathbb{1}(d_i < D_S + \\delta)\\,\\alpha_i}$</div>
                    with fixed surfel weight $w_S = 1$. The Gaussian sum is order-independent, so no sorting is needed.
                </div>
            </details>
        </div>`;
    }

    /** ① Popping under view change (RayCompositingDiagram). */
    private cardPopping(): string {
        return `
        <div class="ccard">
            <div class="ccard-head">
                <div class="ccard-num">1</div>
                <h3 class="ccard-title">Orbit the camera → 3DGS "pops"</h3>
            </div>
            <p class="ccard-take">3DGS sorts splats by depth, then alpha-blends. As you orbit, that global order <b>flips all at once</b> and the colour jumps — the <span class="red">popping</span> artifact. GES just <b>adds</b> the splats up, so the result never jumps.</p>
            <div class="viz-flow">
                <span class="viz-tag t-in"><span class="k">input</span> camera angle</span>
                <span class="viz-arrow">→</span>
                <span class="viz-tag t-out"><span class="k">output</span> 3DGS strip</span>
                <span class="viz-tag t-out"><span class="k">output</span> GES strip</span>
            </div>
            <div class="control-group">
                <label for="orderSlider">Drag to orbit the camera <span class="slider-label-val" id="orderValue">0°</span></label>
                <input type="range" id="orderSlider" min="-45" max="45" step="1" value="0">
            </div>
            <canvas class="diagram-canvas" id="ray-canvas"></canvas>
            <div class="viz-caption">Top = a top-down view of the camera orbiting the splats. Bottom strips = the rendered pixels: watch the <span class="red">3DGS</span> strip's colours snap as the sort order flips, while the <span class="green">GES</span> strip stays smooth.</div>
            <details class="math">
                <summary>show the maths</summary>
                <div class="math-body">
                    <div class="row"><span class="lbl red">3DGS</span> $C = \\sum_i c_i\\alpha_i \\prod_{j<i}(1-\\alpha_j)$ &nbsp;— order-dependent (front-to-back)</div>
                    <div class="row"><span class="lbl green">GES</span> $C = \\dfrac{\\sum_i c_i\\alpha_i}{\\sum_i \\alpha_i}$ &nbsp;— a sum, so order doesn't matter</div>
                </div>
            </details>
        </div>`;
    }

    /** ② Why it pops — tile membership (TileBoundaryDiagram). */
    private cardTiles(): string {
        return `
        <div class="ccard">
            <div class="ccard-head">
                <div class="ccard-num">2</div>
                <h3 class="ccard-title">Why it happens: tiles drop splats</h3>
            </div>
            <p class="ccard-take">3DGS chops the screen into <b>tiles</b>; each tile only draws the blobs assigned to it. Slide blob <b style="color:var(--cmp-surfel)">B</b> across a tile border: when a tile drops it, the two tiles disagree and a hard <span class="red">seam</span> appears. GES has no tiles → no seam.</p>
            <div class="viz-flow">
                <span class="viz-tag t-in"><span class="k">input</span> blob B position</span>
                <span class="viz-arrow">→</span>
                <span class="viz-tag t-out"><span class="k">output ①</span> 3DGS tiled</span>
                <span class="viz-tag t-out"><span class="k">output ②</span> GES no-tiles</span>
            </div>
            <div class="control-group">
                <label for="tileSplatSlider">Slide blob B across the tile border <span class="slider-label-val" id="tileSplatValue">in Tile 1</span></label>
                <input type="range" id="tileSplatSlider" min="0" max="100" step="1" value="80">
            </div>
            <div id="tileStatus" class="status-box"></div>
            <canvas class="diagram-canvas" id="tile-canvas"></canvas>
            <div class="viz-caption">Each row shows the same two blobs. Top = 3DGS (two tiles); the result bar gets a <span class="red">hard seam</span> where a tile skips B. Bottom = GES (one pass); the same blobs stay <span class="green">smooth</span>.</div>
        </div>`;
    }

    /** ③ Depth test → no colour leaking (ColorLeakingDiagram). */
    private cardLeaking(): string {
        return `
        <div class="ccard">
            <div class="ccard-head">
                <div class="ccard-num">3</div>
                <h3 class="ccard-title">Depth test → no colour leaking</h3>
            </div>
            <p class="ccard-take">A bright Gaussian hidden <b>behind</b> the surface shouldn't be visible. GES culls it with a depth test against the surfel depth (with a small offset <b>δ</b>); sort-free methods blend it anyway and <span class="red">leak</span> its colour through.</p>
            <div class="viz-flow">
                <span class="viz-tag t-in"><span class="k">input</span> depth offset δ</span>
                <span class="viz-arrow">→</span>
                <span class="viz-tag t-out"><span class="k">output</span> SortFreeGS (leaks)</span>
                <span class="viz-tag t-out"><span class="k">output</span> GES (culled)</span>
            </div>
            <div class="control-group">
                <label for="compDeltaSlider">Depth-test offset δ <span class="slider-label-val" id="compDeltaValue">0.5</span></label>
                <input type="range" id="compDeltaSlider" min="0.0" max="1.5" step="0.05" value="0.5">
            </div>
            <canvas class="diagram-canvas" id="leaking-canvas"></canvas>
            <div class="viz-caption">A cyan <b style="color:var(--cmp-surfel)">surfel</b> sits in front of a red Gaussian. Left = SortFreeGS always shows the red <span class="red">leak</span>. Right = GES culls the red when it's deeper than the surfel + δ (<span class="green">clean</span>). Push δ too high and even GES starts to leak.</div>
            <details class="math">
                <summary>show the maths</summary>
                <div class="math-body">
                    The indicator $\\mathbb{1}(d_i < D_S + \\delta)$ keeps a Gaussian only if its depth $d_i$ is in front of the surfel depth $D_S$ (plus offset):
                    <div class="row">$C = \\dfrac{w_S C_S + \\sum_i \\mathbb{1}(d_i < D_S + \\delta)\\,c_i\\alpha_i}{w_S + \\sum_i \\mathbb{1}(d_i < D_S + \\delta)\\,\\alpha_i + \\epsilon}$</div>
                </div>
            </details>
        </div>`;
    }

    /** ④ Limitation — specular reflections (Reflection3DViewer). */
    private cardSpecular(): string {
        return `
        <div class="ccard">
            <div class="ccard-head">
                <div class="ccard-num">4</div>
                <h3 class="ccard-title">Honest limitation: specular reflections</h3>
            </div>
            <p class="ccard-take">3DGS can fake a mirror reflection by hiding transparent "floaters" <b>under</b> the surface — you see through to them. GES's <b style="color:var(--cmp-surfel)">opaque surfel</b> blocks anything underground, so reflections must be faked differently and are harder to capture.</p>
            <div class="viz-flow">
                <span class="viz-tag t-in"><span class="k">input</span> drag to orbit</span>
                <span class="viz-arrow">→</span>
                <span class="viz-tag t-out"><span class="k">output</span> 3DGS (sees floaters)</span>
                <span class="viz-tag t-out"><span class="k">output</span> GES (blocked)</span>
            </div>
            <div id="reflection-container" style="position:relative; border-radius:8px; overflow:hidden; border:1px solid rgba(255,255,255,0.05); background:#0d0c14;">
                <div style="position:absolute; left:10px; top:10px; color:var(--cmp-3dgs); font-size:11px; font-weight:bold; background:rgba(0,0,0,0.5); padding:4px 6px; border-radius:4px; z-index:10;">3DGS — alpha blend</div>
                <div style="position:absolute; right:10px; top:10px; color:var(--cmp-ges); font-size:11px; font-weight:bold; background:rgba(0,0,0,0.5); padding:4px 6px; border-radius:4px; z-index:10;">GES — opaque surfel</div>
            </div>
            <div class="viz-caption">Orbit below the floor: 3DGS (left) lets the camera see the underground orange floater through the semi-transparent floor; GES (right) blocks it, so the same trick can't be used.</div>
        </div>`;
    }

    /** Live 3D side-by-side (SideBySide3DViewer) — the sticky hero on the right. */
    private cardLive(): string {
        return `
        <div class="ccard">
            <div class="ccard-head">
                <div class="ccard-num live">▶</div>
                <h3 class="ccard-title">Live 3D — the two-pass renderer</h3>
            </div>
            <p class="ccard-take">Orbit the splats. The cyan <b style="color:var(--cmp-surfel)">surfel</b> writes depth and <b>blocks</b> Gaussians behind it (GES, right). 3DGS (left) re-sorts every frame, so its colours <span class="red">pop</span> as you move.</p>
            <div class="viz-flow">
                <span class="viz-tag t-in"><span class="k">input</span> orbit camera</span>
                <span class="viz-arrow">→</span>
                <span class="viz-tag t-mid"><span class="k">intermediate</span> surfel depth</span>
                <span class="viz-arrow">→</span>
                <span class="viz-tag t-out"><span class="k">output</span> composited frame</span>
            </div>
            <div class="sbs-viewport-container">
                <div class="sbs-viewport" id="viewport-3dgs">
                    <div class="viewport-label" style="color:var(--cmp-3dgs); border-color:var(--cmp-3dgs);">3DGS · alpha blend (sorted)</div>
                    <div id="debug-3dgs" style="position:absolute; bottom:10px; left:10px; font-family:monospace; font-size:10px; color:#fff; background:rgba(0,0,0,0.8); padding:4px 6px; border-radius:4px; z-index:2; pointer-events:none;"></div>
                </div>
                <div class="sbs-viewport" id="viewport-ges">
                    <div class="viewport-label" style="color:var(--cmp-ges); border-color:var(--cmp-ges);">GES · two-pass depth test</div>
                    <div id="debug-ges" style="position:absolute; bottom:10px; left:10px; font-family:monospace; font-size:10px; color:#fff; background:rgba(0,0,0,0.8); padding:4px 6px; border-radius:4px; z-index:2; pointer-events:none;"></div>
                </div>
            </div>
            <div class="viz-caption">The debug readout lists each splat's view-space depth and the current draw order. When 3DGS's order flips, its blend jumps (popping). On the GES side a ✗ marks Gaussians the surfel's depth test culls.</div>
        </div>`;
    }
}
