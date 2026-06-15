/**
 * TileBoundaryDiagram visualizes tile-based sorting.
 * Shows how dividing the screen into tiles and sorting Gaussians independently 
 * in 3DGS creates seams at tile boundaries, whereas GES's additive blending is seamless.
 */
export class TileBoundaryDiagram {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;

    constructor() {
        this.canvas = document.getElementById("tile-canvas") as HTMLCanvasElement;
        this.ctx = this.canvas.getContext("2d")!;
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
     * Draw the side-by-side comparison diagram.
     */
    public draw() {
        const W = this.canvas.width / (window.devicePixelRatio || 1);
        const H = this.canvas.height / (window.devicePixelRatio || 1);
        const ctx = this.ctx;
        ctx.clearRect(0, 0, W, H);
        
        const halfW = W / 2;

        // Vertical divider separating 3DGS and GES comparison columns
        ctx.strokeStyle = "rgba(255,255,255,0.1)";
        ctx.beginPath();
        ctx.moveTo(halfW, 10);
        ctx.lineTo(halfW, H - 10);
        ctx.stroke();

        // Section Titles
        ctx.fillStyle = "#fff";
        ctx.font = "bold 11px 'Space Grotesk'";
        ctx.fillText("3DGS — Per-Tile Sorting", 15, 20);
        ctx.fillText("GES — Additive (No Sorting)", halfW + 15, 20);

        // Draw individual sides
        this.draw3DGSSide(ctx, 10, halfW - 10, H);
        this.drawGESSide(ctx, halfW + 10, W - 10, H);
    }

    /**
     * Draw 3DGS side showcasing the visual seam between TILE 0 and TILE 1.
     */
    private draw3DGSSide(ctx: CanvasRenderingContext2D, x0: number, x1: number, H: number) {
        const w = x1 - x0;
        const cx = x0 + w / 2;
        const tileW = 56;
        const tileH = 56;

        // Two Gaussians: A (Magenta, depth 1.0) and B (Cyan, depth 1.5)
        const gA = { cx: cx - 14, cy: 82, rgb: [1.0, 0.1, 0.6] as [number,number,number], sigma: 30, opacity: 0.95, depth: 1.0 };
        const gB = { cx: cx + 14, cy: 82, rgb: [0.0, 0.8, 1.0] as [number,number,number], sigma: 30, opacity: 0.95, depth: 1.5 };

        const gridY = 30;
        const leftTileX = cx - tileW - 1;
        const rightTileX = cx + 1;

        // Tile background fills
        ctx.fillStyle = "rgba(0,210,255,0.03)";
        ctx.fillRect(leftTileX, gridY, tileW, tileH);
        ctx.fillStyle = "rgba(255,159,67,0.03)";
        ctx.fillRect(rightTileX, gridY, tileW, tileH);

        // Tile outlines
        ctx.strokeStyle = "rgba(0,210,255,0.3)";
        ctx.lineWidth = 1;
        ctx.strokeRect(leftTileX, gridY, tileW, tileH);
        ctx.strokeStyle = "rgba(255,159,67,0.3)";
        ctx.strokeRect(rightTileX, gridY, tileW, tileH);

        // Tile boundary line (Red, dotted)
        ctx.strokeStyle = "rgba(255,74,90,0.5)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(cx, gridY);
        ctx.lineTo(cx, gridY + tileH);
        ctx.stroke();
        ctx.setLineDash([]);

        // Tile Labels
        ctx.fillStyle = "rgba(0,210,255,0.6)";
        ctx.font = "bold 9px monospace";
        ctx.fillText("TILE 0", leftTileX + 8, gridY + 12);
        ctx.fillStyle = "rgba(255,159,67,0.6)";
        ctx.fillText("TILE 1", rightTileX + 8, gridY + 12);

        // Gaussian A Footprint
        ctx.strokeStyle = "rgba(255,26,153,0.5)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(gA.cx, gA.cy, gA.sigma * 0.7, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = `rgba(${gA.rgb[0]*255|0},${gA.rgb[1]*255|0},${gA.rgb[2]*255|0},0.3)`;
        ctx.fill();

        // Gaussian B Footprint
        ctx.strokeStyle = `rgba(${gB.rgb[0]*255|0},${gB.rgb[1]*255|0},${gB.rgb[2]*255|0},0.5)`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(gB.cx, gB.cy, gB.sigma * 0.7, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = `rgba(${gB.rgb[0]*255|0},${gB.rgb[1]*255|0},${gB.rgb[2]*255|0},0.3)`;
        ctx.fill();

        // Footprint Identifier labels
        ctx.fillStyle = "#fff";
        ctx.font = "bold 10px 'Space Grotesk'";
        ctx.fillText("A", gA.cx - 4, gA.cy + 4);
        ctx.fillText("B", gB.cx - 4, gB.cy + 4);

        ctx.fillStyle = "#888";
        ctx.font = "8px monospace";
        ctx.fillText("z=1.0", gA.cx - 12, gA.cy + 15);
        ctx.fillText("z=1.5", gB.cx - 12, gB.cy + 15);

        const sortY = gridY + tileH + 14;
        ctx.fillStyle = "#aaa";
        ctx.font = "9px monospace";
        ctx.fillText("Tile 0 sort: A→B", leftTileX, sortY);
        ctx.fillText("Tile 1 sort: B→A", rightTileX, sortY);

        ctx.fillStyle = "#666";
        ctx.font = "8px 'Space Grotesk'";
        ctx.fillText("Each tile sorts by center depth.", x0 + 5, sortY + 13);
        ctx.fillText("Different tiles → different order → seam!", x0 + 5, sortY + 24);

        // Draw visual seam on the pixel strip
        const stripY = sortY + 32;
        const stripH = 28;
        const imgW = Math.ceil(w - 10);
        const imgData = ctx.createImageData(imgW, stripH);
        const data = imgData.data;

        for (let y = 0; y < stripH; y++) {
            for (let lx = 0; lx < imgW; lx++) {
                const wx = x0 + 5 + lx;
                const wy = stripY + y;
                const dAsq = (wx - gA.cx) ** 2 + (wy - (stripY + stripH/2)) ** 2;
                const wA = gA.opacity * Math.exp(-0.5 * dAsq / (gA.sigma ** 2));
                const dBsq = (wx - gB.cx) ** 2 + (wy - (stripY + stripH/2)) ** 2;
                const wB = gB.opacity * Math.exp(-0.5 * dBsq / (gB.sigma ** 2));
                
                let r = 0, g = 0, b = 0;
                const isLeftTile = wx < cx;
                if (isLeftTile) {
                    r = gA.rgb[0] * wA + gB.rgb[0] * wB * (1 - wA);
                    g = gA.rgb[1] * wA + gB.rgb[1] * wB * (1 - wA);
                    b = gA.rgb[2] * wA + gB.rgb[2] * wB * (1 - wA);
                } else {
                    r = gB.rgb[0] * wB + gA.rgb[0] * wA * (1 - wB);
                    g = gB.rgb[1] * wB + gA.rgb[1] * wA * (1 - wB);
                    b = gB.rgb[2] * wB + gA.rgb[2] * wA * (1 - wB);
                }
                const off = (y * imgData.width + lx) * 4;
                data[off]     = Math.min(255, r * 255 | 0);
                data[off + 1] = Math.min(255, g * 255 | 0);
                data[off + 2] = Math.min(255, b * 255 | 0);
                data[off + 3] = 255;
            }
        }

        const tmp = document.createElement("canvas");
        tmp.width = imgW;
        tmp.height = stripH;
        tmp.getContext("2d")!.putImageData(imgData, 0, 0);
        ctx.drawImage(tmp, x0 + 5, stripY);

        ctx.strokeStyle = "rgba(255,74,90,0.9)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(cx, stripY - 2);
        ctx.lineTo(cx, stripY + stripH + 2);
        ctx.stroke();

        ctx.fillStyle = "#ff4a5a";
        ctx.font = "bold 9px 'Space Grotesk'";
        ctx.fillText("▲ SEAM", cx - 18, stripY + stripH + 14);
    }

    /**
     * Draw GES side showing a seamless composition across the middle.
     */
    private drawGESSide(ctx: CanvasRenderingContext2D, x0: number, x1: number, H: number) {
        const w = x1 - x0;
        const cx = x0 + w / 2;

        const gA = { cx: cx - 14, cy: 82, rgb: [1.0, 0.1, 0.6] as [number,number,number], sigma: 30, opacity: 0.95 };
        const gB = { cx: cx + 14, cy: 82, rgb: [0.0, 0.8, 1.0] as [number,number,number], sigma: 30, opacity: 0.95 };

        ctx.fillStyle = "rgba(0,255,135,0.03)";
        ctx.fillRect(x0 + 5, 30, w - 10, 56);
        ctx.strokeStyle = "rgba(0,255,135,0.2)";
        ctx.lineWidth = 1;
        ctx.strokeRect(x0 + 5, 30, w - 10, 56);
        
        ctx.fillStyle = "rgba(0,255,135,0.5)";
        ctx.font = "bold 9px monospace";
        ctx.fillText("NO TILES — SINGLE PASS", x0 + 15, 42);

        // Gaussian A
        ctx.strokeStyle = `rgba(${gA.rgb[0]*255|0},${gA.rgb[1]*255|0},${gA.rgb[2]*255|0},0.5)`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(gA.cx, gA.cy, gA.sigma * 0.7, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = `rgba(${gA.rgb[0]*255|0},${gA.rgb[1]*255|0},${gA.rgb[2]*255|0},0.3)`;
        ctx.fill();

        // Gaussian B
        ctx.strokeStyle = `rgba(${gB.rgb[0]*255|0},${gB.rgb[1]*255|0},${gB.rgb[2]*255|0},0.5)`;
        ctx.beginPath();
        ctx.arc(gB.cx, gB.cy, gB.sigma * 0.7, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = `rgba(${gB.rgb[0]*255|0},${gB.rgb[1]*255|0},${gB.rgb[2]*255|0},0.3)`;
        ctx.fill();

        ctx.fillStyle = "#fff";
        ctx.font = "bold 10px 'Space Grotesk'";
        ctx.fillText("A", gA.cx - 4, gA.cy + 4);
        ctx.fillText("B", gB.cx - 4, gB.cy + 4);

        const sortY = 30 + 56 + 14;
        ctx.fillStyle = "#aaa";
        ctx.font = "9px monospace";
        ctx.fillText("Additive: C_G = Σ c_i·α_i", x0 + 8, sortY);
        ctx.fillText("Normalize: C = C_G / Σα_i", x0 + 8, sortY + 13);
        
        ctx.fillStyle = "#666";
        ctx.font = "8px 'Space Grotesk'";
        ctx.fillText("Sum is commutative → order doesn't matter!", x0 + 5, sortY + 26);

        const stripY = sortY + 34;
        const stripH = 28;
        const imgW = Math.ceil(w - 10);
        const imgData = ctx.createImageData(imgW, stripH);
        const data = imgData.data;

        for (let y = 0; y < stripH; y++) {
            for (let lx = 0; lx < imgW; lx++) {
                const wx = x0 + 5 + lx;
                const wy = stripY + y;
                const dAsq = (wx - gA.cx) ** 2 + (wy - (stripY + stripH/2)) ** 2;
                const wA = gA.opacity * Math.exp(-0.5 * dAsq / (gA.sigma ** 2));
                const dBsq = (wx - gB.cx) ** 2 + (wy - (stripY + stripH/2)) ** 2;
                const wB = gB.opacity * Math.exp(-0.5 * dBsq / (gB.sigma ** 2));
                
                const sumA = wA + wB;
                let r = 0, g = 0, b = 0;
                if (sumA > 0.005) {
                    r = (gA.rgb[0] * wA + gB.rgb[0] * wB) / sumA;
                    g = (gA.rgb[1] * wA + gB.rgb[1] * wB) / sumA;
                    b = (gA.rgb[2] * wA + gB.rgb[2] * wB) / sumA;
                    const agg = 1.0 - Math.exp(-sumA);
                    r *= agg; g *= agg; b *= agg;
                }
                const off = (y * imgData.width + lx) * 4;
                data[off]     = Math.min(255, r * 255 | 0);
                data[off + 1] = Math.min(255, g * 255 | 0);
                data[off + 2] = Math.min(255, b * 255 | 0);
                data[off + 3] = 255;
            }
        }

        const tmp = document.createElement("canvas");
        tmp.width = imgW;
        tmp.height = stripH;
        tmp.getContext("2d")!.putImageData(imgData, 0, 0);
        ctx.drawImage(tmp, x0 + 5, stripY);

        ctx.fillStyle = "#00ff87";
        ctx.font = "bold 9px 'Space Grotesk'";
        ctx.fillText("✓ NO SEAM — Smooth everywhere", x0 + 15, stripY + stripH + 14);
    }
}
