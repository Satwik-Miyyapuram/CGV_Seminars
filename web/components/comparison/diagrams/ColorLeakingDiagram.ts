/**
 * ColorLeakingDiagram visualizes color leaking behavior.
 * Compares SortFreeGS (which always leaks background elements) with GES,
 * demonstrating how GES uses the delta (δ) threshold to cull occluded Gaussians.
 */
export class ColorLeakingDiagram {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private deltaSlider: HTMLInputElement;
    private deltaValue: HTMLElement;

    constructor() {
        this.canvas = document.getElementById("leaking-canvas") as HTMLCanvasElement;
        this.ctx = this.canvas.getContext("2d")!;
        this.deltaSlider = document.getElementById("compDeltaSlider") as HTMLInputElement;
        this.deltaValue = document.getElementById("compDeltaValue")!;
        
        // Redraw on threshold delta slider updates
        this.deltaSlider.addEventListener("input", () => {
            this.deltaValue.textContent = parseFloat(this.deltaSlider.value).toFixed(1);
            this.draw();
        });
        
        this.resize();
        this.draw();
    }

    /**
     * Resize canvas keeping device pixel ratio.
     */
    public resize() {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.ctx.scale(dpr, dpr);
    }

    /**
     * Draw side-by-side leak representation.
     */
    public draw() {
        const W = this.canvas.width / (window.devicePixelRatio || 1);
        const H = this.canvas.height / (window.devicePixelRatio || 1);
        const ctx = this.ctx;
        ctx.clearRect(0, 0, W, H);
        
        const halfW = W / 2;
        const delta = parseFloat(this.deltaSlider.value);

        // Divider
        ctx.strokeStyle = "rgba(255,255,255,0.1)";
        ctx.beginPath();
        ctx.moveTo(halfW, 10);
        ctx.lineTo(halfW, H - 10);
        ctx.stroke();

        ctx.fillStyle = "#fff";
        ctx.font = "bold 11px 'Space Grotesk'";
        ctx.fillText("SortFreeGS — No Depth Culling", 15, 22);
        ctx.fillText("GES — Depth Culling with δ", halfW + 15, 22);

        // Draw Left (SortFreeGS - no depth test culling)
        this.drawSide(ctx, 15, halfW - 15, H, false, delta);

        // Draw Right (GES - with depth test culling threshold delta)
        this.drawSide(ctx, halfW + 15, W - 15, H, true, delta);
    }

    /**
     * Draw individual side panel.
     */
    private drawSide(ctx: CanvasRenderingContext2D, x0: number, x1: number, H: number, enableCull: boolean, delta: number) {
        const w = x1 - x0;
        const midY = H / 2 + 10;

        // Input/setup stage label
        ctx.fillStyle = "rgba(255, 255, 255, 0.35)";
        ctx.font = "8px monospace";
        ctx.fillText("INTERMEDIATE: Setup", x0, 34);

        // Visual representations: Cyan Surfel (in front) and Red Gaussian (behind)
        const surfel = { cx: x0 + w / 2 - 8, cy: midY, r: 38, rgb: [0.0, 0.82, 1.0] as [number, number, number], depth: 2.0, opacity: 0.95 };
        const gauss  = { cx: x0 + w / 2 + 8, cy: midY, rgb: [1.0, 0.29, 0.35] as [number, number, number], depth: 2.5, opacity: 0.90, sigma: 28 };

        const imgH = 65;
        const imgData = ctx.createImageData(Math.ceil(w), imgH);
        const data = imgData.data;

        for (let y = 0; y < imgH; y++) {
            const wy = midY - 32 + y;
            for (let lx = 0; lx < Math.ceil(w); lx++) {
                const wx = x0 + lx;
                
                // Check if coordinate falls within the surfel radius
                const dsSq = (wx - surfel.cx) ** 2 + (wy - surfel.cy) ** 2;
                const wS = dsSq < (surfel.r * surfel.r) ? surfel.opacity : 0.0;
                
                // Evaluate Gaussian falloff
                const dgSq = (wx - gauss.cx) ** 2 + (wy - gauss.cy) ** 2;
                const wG = gauss.opacity * Math.exp(-0.5 * dgSq / (gauss.sigma ** 2));
                
                // Depth culling: cull if depth is further than surfel depth + delta offset
                const D_S = wS > 0 ? surfel.depth : Infinity;
                const culled = enableCull && (gauss.depth > D_S + delta);
                const wG_final = culled ? 0.0 : wG;

                // Additive blend + normalization (premultiplied representation)
                const sumA = wS + wG_final;
                let r = 0, g = 0, b = 0;
                if (sumA > 0.001) {
                    r = (surfel.rgb[0] * wS + gauss.rgb[0] * wG_final) / sumA;
                    g = (surfel.rgb[1] * wS + gauss.rgb[1] * wG_final) / sumA;
                    b = (surfel.rgb[2] * wS + gauss.rgb[2] * wG_final) / sumA;
                    
                    const agg = 1.0 - Math.exp(-sumA);
                    r *= agg;
                    g *= agg;
                    b *= agg;
                }

                const off = (y * imgData.width + lx) * 4;
                data[off]     = Math.min(255, r * 255 | 0);
                data[off + 1] = Math.min(255, g * 255 | 0);
                data[off + 2] = Math.min(255, b * 255 | 0);
                data[off + 3] = 255;
            }
        }

        const tmp = document.createElement("canvas");
        tmp.width = Math.ceil(w);
        tmp.height = imgH;
        tmp.getContext("2d")!.putImageData(imgData, 0, 0);
        ctx.drawImage(tmp, x0, midY - 32);

        // Output stage label, overlaid on the rendered-pixel band
        ctx.fillStyle = "rgba(255, 255, 255, 0.5)";
        ctx.font = "8px monospace";
        ctx.fillText("OUTPUT: Rendered Pixels", x0 + 5, midY - 20);

        // Render annotations
        ctx.fillStyle = "#888";
        ctx.font = "9px monospace";
        ctx.fillText("Surfel (z=2.0)", x0 + 5, midY - 38);
        ctx.fillText("Gaussian (z=2.5)", x0 + 5, midY + 42);

        const isLeaking = (gauss.depth <= surfel.depth + delta);
        if (enableCull) {
            ctx.fillStyle = isLeaking ? "#ff4a5a" : "#00ff87";
            ctx.font = "10px 'Space Grotesk'";
            ctx.fillText(
                isLeaking ? "⚠️ Leaking (δ too high)" : "✓ Culled Correctly",
                x0 + 5, midY + 52
            );
        } else {
            ctx.fillStyle = "#ff4a5a";
            ctx.font = "10px 'Space Grotesk'";
            ctx.fillText("⚠️ Red Leaks Through", x0 + 5, midY + 52);
        }
    }
}
