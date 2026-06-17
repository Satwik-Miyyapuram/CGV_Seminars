import { gaussWeight, orthoProject, SPLATS, CENTER_X, CENTER_Z, SCALE_Z } from "../shared";

/**
 * RayCompositingDiagram visualizes camera orbiting and ray projection.
 * Shows the difference between tile-based sorted alpha blending in 3DGS
 * (which causes popping due to sorting key changes) and additive commutative blending in GES.
 */
export class RayCompositingDiagram {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private orderSlider: HTMLInputElement;
    private orderValue: HTMLElement;

    constructor() {
        this.canvas = document.getElementById("ray-canvas") as HTMLCanvasElement;
        this.ctx = this.canvas.getContext("2d")!;
        this.orderSlider = document.getElementById("orderSlider") as HTMLInputElement;
        this.orderValue = document.getElementById("orderValue")!;
        
        // Listen to slider movements to redraw the diagram
        this.orderSlider.addEventListener("input", () => {
            this.orderValue.textContent = `${this.orderSlider.value}°`;
            this.draw();
        });
        
        this.resize();
        this.draw();
    }

    /**
     * Resize canvas with respect to Device Pixel Ratio (DPR) to avoid blurriness.
     */
    public resize() {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.ctx.scale(dpr, dpr);
    }

    /**
     * Main drawing function.
     */
    public draw() {
        const W = this.canvas.width / (window.devicePixelRatio || 1);
        const H = this.canvas.height / (window.devicePixelRatio || 1);
        const ctx = this.ctx;
        ctx.clearRect(0, 0, W, H);
        
        const halfW = W / 2;
        const angleRad = (parseFloat(this.orderSlider.value) * Math.PI) / 180;

        // Project all splats through camera rotation
        const projected = SPLATS.map(s => {
            const wx = s.x - CENTER_X;
            const wz = (s.depth - CENTER_Z) * SCALE_Z;
            const { depth: effDepth, projX: pxRaw } = orthoProject(wx, wz, angleRad);
            return {
                s,
                effDepth,
                projX: CENTER_X + pxRaw,
                topDownX: wx * 0.8,
                topDownZ: wz * 0.8,
            };
        });

        // 1. TOP VIEW: Centered Top-Down Camera Orbit
        const topY = 55;
        ctx.save();
        ctx.translate(halfW, topY);
        
        // Orbit boundary circle
        ctx.strokeStyle = "rgba(255,255,255,0.05)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(0, 0, 45, 0, Math.PI * 2);
        ctx.stroke();

        // Draw splat representations in 2D top-down space
        projected.forEach(p => {
            ctx.fillStyle = p.s.color;
            ctx.beginPath();
            ctx.arc(p.topDownX, p.topDownZ, 5, 0, Math.PI * 2);
            ctx.fill();
        });

        // Camera (Rotated representation)
        ctx.rotate(angleRad);
        ctx.fillStyle = "#fff";
        ctx.font = "11px monospace";
        ctx.fillText("👁", -8, -50);
        ctx.strokeStyle = "rgba(255,255,255,0.4)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(-25, -42);
        ctx.lineTo(25, -42);
        ctx.stroke();

        // Camera Ray line
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = "rgba(255,255,255,0.25)";
        ctx.beginPath();
        ctx.moveTo(0, -42);
        ctx.lineTo(0, 55);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();

        ctx.fillStyle = "#888";
        ctx.font = "9px monospace";
        ctx.textAlign = "center";
        ctx.fillText("INTERMEDIATE: Top-Down Orbit View", halfW, 12);
        ctx.textAlign = "left";

        // 2. MIDDLE VIEW: Divider + Labels
        const midY = 115;
        
        // Horizontal separation line
        ctx.strokeStyle = "rgba(255,255,255,0.08)";
        ctx.beginPath();
        ctx.moveTo(15, midY - 5);
        ctx.lineTo(W - 15, midY - 5);
        ctx.stroke();

        // Vertical division line
        ctx.strokeStyle = "rgba(255,255,255,0.1)";
        ctx.beginPath();
        ctx.moveTo(halfW, midY - 5);
        ctx.lineTo(halfW, H - 5);
        ctx.stroke();

        // 3DGS label and math subtitle
        ctx.fillStyle = "#fff";
        ctx.font = "bold 11px 'Space Grotesk'";
        ctx.fillText("3DGS — Tile-Based α-Blending", 15, midY + 10);
        ctx.fillStyle = "#888";
        ctx.font = "9px monospace";
        ctx.fillText("Sort order computed per-tile", 15, midY + 22);

        // GES label and math subtitle
        ctx.fillStyle = "#fff";
        ctx.font = "bold 11px 'Space Grotesk'";
        ctx.fillText("GES — Additive + Normalize", halfW + 10, midY + 10);
        ctx.fillStyle = "#00ff87";
        ctx.font = "9px monospace";
        ctx.fillText("All splats: additive (order-free)", halfW + 10, midY + 22);

        // 3. BOTTOM VIEW: Gaussian Curves & Rendered Color Strips
        // (curveBaseY pushed down so the "INTERMEDIATE" label gets its own line below
        //  the subtitle instead of colliding with it.)
        const curveBaseY = midY + 68;
        const curveH = 28;
        const stripY = curveBaseY + 5;
        const stripH = 25;

        // Render Left: 3DGS tile-based alpha-blended strip
        this.drawStrip3DGS(ctx, 15, halfW - 10, curveBaseY, curveH, stripY, stripH, projected);

        // Render Right: GES additive normalized strip
        this.drawStripGES(ctx, halfW + 10, W - 15, curveBaseY, curveH, stripY, stripH, projected);
    }

    /**
     * Draw 3DGS style tile-based sorted alpha-blending representation.
     */
    private drawStrip3DGS(
        ctx: CanvasRenderingContext2D,
        x0: number, x1: number,
        baseY: number, curveH: number,
        stripY: number, stripH: number,
        all: any[]
    ) {
        const w = x1 - x0;

        // Intermediate stage: the per-blob blending curves (own line below the subtitle).
        ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
        ctx.font = "8px monospace";
        ctx.fillText("INTERMEDIATE: Blending Curves", x0, baseY - curveH - 6);

        ctx.strokeStyle = "rgba(255,255,255,0.12)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x0, baseY);
        ctx.lineTo(x1, baseY);
        ctx.stroke();

        // Draw standard Gaussian curves globally for reference
        const globalSorted = [...all].sort((a, b) => a.effDepth - b.effDepth);
        const backToFront = [...globalSorted].reverse();
        backToFront.forEach(p => {
            ctx.fillStyle = p.s.color;
            ctx.beginPath();
            ctx.moveTo(x0, baseY);
            for (let px = x0; px <= x1; px++) {
                const screenX = ((px - x0) / w) * 240 + 70;
                const wt = gaussWeight(screenX, p.projX, p.s.sigma, p.s.opacity);
                ctx.lineTo(px, baseY - wt * curveH);
            }
            ctx.lineTo(x1, baseY);
            ctx.fill();
        });

        // Strip background
        ctx.fillStyle = "#09090e";
        ctx.fillRect(x0, stripY, w, stripH);
        ctx.strokeStyle = "rgba(255,255,255,0.08)";
        ctx.strokeRect(x0, stripY, w, stripH);

        // Simulate 4 tiles across the strip width
        const numTiles = 4;
        const tileW = w / numTiles;

        for (let t = 0; t < numTiles; t++) {
            const tx0 = x0 + t * tileW;
            const tx1 = tx0 + tileW;
            const screenTx0 = ((tx0 - x0) / w) * 240 + 70;
            const screenTx1 = ((tx1 - x0) / w) * 240 + 70;

            // Step 1: Tile Culling (splats within 3*sigma bounds of tile)
            const tileSplats = all.filter(p => {
                const minX = p.projX - 3 * p.s.sigma;
                const maxX = p.projX + 3 * p.s.sigma;
                return maxX >= screenTx0 && minX <= screenTx1;
            });

            // Step 2: Sort relative to depths
            const tileSorted = tileSplats.sort((a, b) => a.effDepth - b.effDepth);

            // Step 3: Accumulate Alpha Blending (T = T * (1 - alpha))
            for (let px = tx0; px < tx1; px += 2) {
                const screenX = ((px - x0) / w) * 240 + 70;
                let T = 1.0, cR = 0, cG = 0, cB = 0;
                tileSorted.forEach(p => {
                    const a = gaussWeight(screenX, p.projX, p.s.sigma, p.s.opacity);
                    cR += p.s.rgb[0] * a * T;
                    cG += p.s.rgb[1] * a * T;
                    cB += p.s.rgb[2] * a * T;
                    T *= (1 - a);
                });
                cR = Math.min(1, cR * 1.3);
                cG = Math.min(1, cG * 1.3);
                cB = Math.min(1, cB * 1.3);
                ctx.fillStyle = `rgb(${cR * 255 | 0},${cG * 255 | 0},${cB * 255 | 0})`;
                ctx.fillRect(px, stripY, 2, stripH);
            }

            // Draw tile boundary markers
            if (t > 0) {
                ctx.strokeStyle = "rgba(255,255,255,0.2)";
                ctx.setLineDash([3, 3]);
                ctx.beginPath();
                ctx.moveTo(tx0, baseY - curveH - 5);
                ctx.lineTo(tx0, stripY + stripH);
                ctx.stroke();
                ctx.setLineDash([]);
            }

            // Print render order text above the tile
            const orderStr = tileSorted.map(p => p.s.name.substring(0, 1)).join("→");
            ctx.fillStyle = "rgba(255,255,255,0.6)";
            ctx.font = "8px monospace";
            ctx.textAlign = "center";
            ctx.fillText(orderStr || "None", tx0 + tileW / 2, stripY - 4);
            ctx.textAlign = "left";
        }

        // Output stage: the rendered pixel strip.
        ctx.fillStyle = "rgba(255, 255, 255, 0.35)";
        ctx.font = "8px monospace";
        ctx.fillText("OUTPUT: Rendered Pixels", x0, stripY + stripH + 12);
    }

    /**
     * Draw GES style additive blending and normalization representation.
     */
    private drawStripGES(
        ctx: CanvasRenderingContext2D,
        x0: number, x1: number,
        baseY: number, curveH: number,
        stripY: number, stripH: number,
        items: any[]
    ) {
        const w = x1 - x0;

        // Intermediate stage: the per-blob blending curves (own line below the subtitle).
        ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
        ctx.font = "8px monospace";
        ctx.fillText("INTERMEDIATE: Blending Curves", x0, baseY - curveH - 6);

        ctx.strokeStyle = "rgba(255,255,255,0.12)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x0, baseY);
        ctx.lineTo(x1, baseY);
        ctx.stroke();

        // Render reference Gaussians (from back to front)
        const backToFront = [...items].sort((a, b) => b.effDepth - a.effDepth);
        backToFront.forEach(p => {
            ctx.fillStyle = p.s.color;
            ctx.beginPath();
            ctx.moveTo(x0, baseY);
            for (let px = x0; px <= x1; px++) {
                const screenX = ((px - x0) / w) * 240 + 70;
                const wt = gaussWeight(screenX, p.projX, p.s.sigma, p.s.opacity);
                ctx.lineTo(px, baseY - wt * curveH);
            }
            ctx.lineTo(x1, baseY);
            ctx.fill();
        });

        // Strip background
        ctx.fillStyle = "#09090e";
        ctx.fillRect(x0, stripY, w, stripH);
        ctx.strokeStyle = "rgba(255,255,255,0.08)";
        ctx.strokeRect(x0, stripY, w, stripH);

        // Compute normalized average blending (Order-independent)
        // C = (Σ c_i * alpha_i) / (Σ alpha_i) * (1 - exp(-Σ alpha_i))
        for (let px = x0; px < x1; px += 2) {
            const screenX = ((px - x0) / w) * 240 + 70;
            let sumCR = 0, sumCG = 0, sumCB = 0, sumA = 0;
            
            items.forEach(p => {
                const a = gaussWeight(screenX, p.projX, p.s.sigma, p.s.opacity);
                sumCR += p.s.rgb[0] * a;
                sumCG += p.s.rgb[1] * a;
                sumCB += p.s.rgb[2] * a;
                sumA += a;
            });

            let fR = 0, fG = 0, fB = 0;
            if (sumA > 0.001) {
                fR = sumCR / sumA;
                fG = sumCG / sumA;
                fB = sumCB / sumA;
                
                // Dim color slightly based on overall alpha accumulation
                const aggAlpha = 1.0 - Math.exp(-sumA);
                fR *= aggAlpha;
                fG *= aggAlpha;
                fB *= aggAlpha;
            }

            fR = Math.min(1, fR * 1.3);
            fG = Math.min(1, fG * 1.3);
            fB = Math.min(1, fB * 1.3);
            
            ctx.fillStyle = `rgb(${fR * 255 | 0},${fG * 255 | 0},${fB * 255 | 0})`;
            ctx.fillRect(px, stripY, 2, stripH);
        }

        // Output stage: the rendered pixel strip.
        ctx.fillStyle = "rgba(255, 255, 255, 0.35)";
        ctx.font = "8px monospace";
        ctx.fillText("OUTPUT: Rendered Pixels", x0, stripY + stripH + 12);
    }
}
