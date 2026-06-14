import { gaussWeight, orthoProject, SPLATS, CENTER_X, CENTER_Z, SCALE_Z } from "./shared";
// ============================================================
// Panel A: Popping Demo — 3DGS (sorted α-blend) vs GES (additive)
// ============================================================
export class RayCompositingDiagram {
    canvas: HTMLCanvasElement;
    ctx: CanvasRenderingContext2D;
    orderSlider: HTMLInputElement;
    orderValue: HTMLElement;
    prevSortKey: string = "";
    constructor() {
        this.canvas = document.getElementById("ray-canvas") as HTMLCanvasElement;
        this.ctx = this.canvas.getContext("2d")!;
        this.orderSlider = document.getElementById("orderSlider") as HTMLInputElement;
        this.orderValue = document.getElementById("orderValue")!;
        this.orderSlider.addEventListener("input", () => {
            this.orderValue.textContent = `${this.orderSlider.value}°`;
            this.draw();
        });
        this.resize();
        this.draw();
    }
    resize() {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.ctx.scale(dpr, dpr);
    }
    draw() {
        const W = this.canvas.width / (window.devicePixelRatio || 1);
        const H = this.canvas.height / (window.devicePixelRatio || 1);
        const ctx = this.ctx;
        ctx.clearRect(0, 0, W, H);
        const halfW = W / 2;
        const angleRad = (parseFloat(this.orderSlider.value) * Math.PI) / 180;
        // --- Project all splats ---
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
        // (Variables removed: Tile-based sorting is now handled inside drawStrip3DGS)
        // ========== TOP: Centered Top-Down Camera Orbit ==========
        const topY = 55;
        ctx.save();
        ctx.translate(halfW, topY);
        // Orbit boundary
        ctx.strokeStyle = "rgba(255,255,255,0.05)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(0, 0, 45, 0, Math.PI * 2);
        ctx.stroke();
        // Splat dots
        projected.forEach(p => {
            ctx.fillStyle = p.s.color;
            ctx.beginPath();
            ctx.arc(p.topDownX, p.topDownZ, 5, 0, Math.PI * 2);
            ctx.fill();
        });
        // Camera (rotated)
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
        // Ray
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = "rgba(255,255,255,0.25)";
        ctx.beginPath();
        ctx.moveTo(0, -42);
        ctx.lineTo(0, 55);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();
        ctx.fillStyle = "#666";
        ctx.font = "9px monospace";
        ctx.fillText("TOP-DOWN ORBIT VIEW", halfW - 55, 12);
        // ========== MIDDLE: Divider + Titles ==========
        const midY = 115;
        // Horizontal separator
        ctx.strokeStyle = "rgba(255,255,255,0.08)";
        ctx.beginPath();
        ctx.moveTo(15, midY - 5);
        ctx.lineTo(W - 15, midY - 5);
        ctx.stroke();
        // Vertical divider
        ctx.strokeStyle = "rgba(255,255,255,0.1)";
        ctx.beginPath();
        ctx.moveTo(halfW, midY - 5);
        ctx.lineTo(halfW, H - 5);
        ctx.stroke();
        // 3DGS label
        ctx.fillStyle = "#fff";
        ctx.font = "bold 11px 'Space Grotesk'";
        ctx.fillText("3DGS — Tile-Based α-Blending", 15, midY + 10);
        ctx.fillStyle = "#888";
        ctx.font = "9px monospace";
        ctx.fillText("Sort order computed per-tile", 15, midY + 22);
        // GES label
        ctx.fillStyle = "#fff";
        ctx.font = "bold 11px 'Space Grotesk'";
        ctx.fillText("GES — Additive + Normalize", halfW + 10, midY + 10);
        ctx.fillStyle = "#00ff87";
        ctx.font = "9px monospace";
        ctx.fillText("All splats: additive (order-free)", halfW + 10, midY + 22);
        // ========== BOTTOM: Gaussian Curves + Color Strips ==========
        const curveBaseY = midY + 55;
        const curveH = 28;
        const stripY = curveBaseY + 5;
        const stripH = 25;
        // --- Left: 3DGS (tile-based alpha blending) ---
        this.drawStrip3DGS(ctx, 15, halfW - 10, curveBaseY, curveH, stripY, stripH, projected);
        // --- Right: GES (additive + normalize, NO culling) ---
        this.drawStripGES(ctx, halfW + 10, W - 15, curveBaseY, curveH, stripY, stripH, projected);
    }
    /** 3DGS: Tile-based sorted alpha blending */
    drawStrip3DGS(
        ctx: CanvasRenderingContext2D,
        x0: number, x1: number,
        baseY: number, curveH: number,
        stripY: number, stripH: number,
        all: any[]
    ) {
        const w = x1 - x0;
        // Draw baseline
        ctx.strokeStyle = "rgba(255,255,255,0.12)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x0, baseY);
        ctx.lineTo(x1, baseY);
        ctx.stroke();

        // Draw Gaussian curves globally (for visual reference)
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

        // Color strip container
        ctx.fillStyle = "#09090e";
        ctx.fillRect(x0, stripY, w, stripH);
        ctx.strokeStyle = "rgba(255,255,255,0.08)";
        ctx.strokeRect(x0, stripY, w, stripH);

        // Tile-based rendering (4 tiles)
        const numTiles = 4;
        const tileW = w / numTiles;

        for (let t = 0; t < numTiles; t++) {
            const tx0 = x0 + t * tileW;
            const tx1 = tx0 + tileW;
            const screenTx0 = ((tx0 - x0) / w) * 240 + 70;
            const screenTx1 = ((tx1 - x0) / w) * 240 + 70;

            // 1. Tile Culling: Only include splats whose 3*sigma radius intersects this tile
            const tileSplats = all.filter(p => {
                const minX = p.projX - 3 * p.s.sigma;
                const maxX = p.projX + 3 * p.s.sigma;
                return maxX >= screenTx0 && minX <= screenTx1;
            });

            // 2. Tile Sorting: Sort the included splats by center depth
            const tileSorted = tileSplats.sort((a, b) => a.effDepth - b.effDepth);

            // 3. Tile Rasterization
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

            // Draw tile divider
            if (t > 0) {
                ctx.strokeStyle = "rgba(255,255,255,0.2)";
                ctx.setLineDash([3, 3]);
                ctx.beginPath();
                ctx.moveTo(tx0, baseY - curveH - 5);
                ctx.lineTo(tx0, stripY + stripH);
                ctx.stroke();
                ctx.setLineDash([]);
            }

            // Write tile order
            const orderStr = tileSorted.map(p => p.s.name.substring(0, 1)).join("→");
            ctx.fillStyle = "rgba(255,255,255,0.6)";
            ctx.font = "8px monospace";
            ctx.textAlign = "center";
            ctx.fillText(orderStr || "None", tx0 + tileW / 2, stripY - 4);
            ctx.textAlign = "left";
        }
    }
    /** GES: additive blending + normalization — ALL splats, NO culling.
     *  C = (Σ c_i · α_i) / (Σ α_i)  — commutative sum, order doesn't matter! */
    drawStripGES(
        ctx: CanvasRenderingContext2D,
        x0: number, x1: number,
        baseY: number, curveH: number,
        stripY: number, stripH: number,
        items: any[]
    ) {
        const w = x1 - x0;
        // Draw baseline
        ctx.strokeStyle = "rgba(255,255,255,0.12)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x0, baseY);
        ctx.lineTo(x1, baseY);
        ctx.stroke();
        // Draw ALL curves (no culling distinction here)
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
        // Color strip: ADDITIVE + NORMALIZE (order-independent!)
        // C = (Σ c_i · α_i) / (Σ α_i) · (1 - exp(-Σα_i))
        ctx.fillStyle = "#09090e";
        ctx.fillRect(x0, stripY, w, stripH);
        ctx.strokeStyle = "rgba(255,255,255,0.08)";
        ctx.strokeRect(x0, stripY, w, stripH);
        for (let px = x0; px < x1; px += 2) {
            const screenX = ((px - x0) / w) * 240 + 70;
            // Additive accumulation — ORDER DOESN'T MATTER
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
                // Normalized weighted average
                fR = sumCR / sumA;
                fG = sumCG / sumA;
                fB = sumCB / sumA;
                // Scale by aggregate opacity
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
    }
}