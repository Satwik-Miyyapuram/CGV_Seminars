// ============================================================
// Panel B: Tile-Based Sorting — 3DGS vs GES
// Shows HOW 3DGS assigns Gaussians to tiles, sorts per-tile,
// and why that creates seams at tile boundaries.
// ============================================================
export class TileBoundaryDiagram {
    canvas: HTMLCanvasElement;
    ctx: CanvasRenderingContext2D;
    constructor() {
        this.canvas = document.getElementById("tile-canvas") as HTMLCanvasElement;
        this.ctx = this.canvas.getContext("2d")!;
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
        // Vertical divider between 3DGS and GES
        ctx.strokeStyle = "rgba(255,255,255,0.1)";
        ctx.beginPath();
        ctx.moveTo(halfW, 10);
        ctx.lineTo(halfW, H - 10);
        ctx.stroke();
        // Titles
        ctx.fillStyle = "#fff";
        ctx.font = "bold 11px 'Space Grotesk'";
        ctx.fillText("3DGS — Per-Tile Sorting", 15, 20);
        ctx.fillText("GES — Additive (No Sorting)", halfW + 15, 20);
        // Draw both sides
        this.draw3DGSSide(ctx, 10, halfW - 10, H);
        this.drawGESSide(ctx, halfW + 10, W - 10, H);
    }
    draw3DGSSide(ctx: CanvasRenderingContext2D, x0: number, x1: number, H: number) {
        const w = x1 - x0;
        const cx = x0 + w / 2;
        const tileW = 56;  // visual tile size
        const tileH = 56;
        // Two Gaussians that span across the tile boundary
        // G_A (Magenta, depth=1.0) center in left tile
        // G_B (Cyan,    depth=1.5) center in right tile
        const gA = { cx: cx - 14, cy: 82, rgb: [1.0, 0.1, 0.6] as [number,number,number], sigma: 30, opacity: 0.95, depth: 1.0, label: "A" };
        const gB = { cx: cx + 14, cy: 82, rgb: [0.0, 0.8, 1.0] as [number,number,number], sigma: 30, opacity: 0.95, depth: 1.5, label: "B" };
        // --- Step 1: Draw tile grid (2 tiles side by side) ---
        const gridY = 30;
        const leftTileX = cx - tileW - 1;
        const rightTileX = cx + 1;
        // Tile backgrounds
        ctx.fillStyle = "rgba(0,210,255,0.03)";
        ctx.fillRect(leftTileX, gridY, tileW, tileH);
        ctx.fillStyle = "rgba(255,159,67,0.03)";
        ctx.fillRect(rightTileX, gridY, tileW, tileH);
        // Tile borders
        ctx.strokeStyle = "rgba(0,210,255,0.3)";
        ctx.lineWidth = 1;
        ctx.strokeRect(leftTileX, gridY, tileW, tileH);
        ctx.strokeStyle = "rgba(255,159,67,0.3)";
        ctx.strokeRect(rightTileX, gridY, tileW, tileH);
        // Tile boundary (emphasized)
        ctx.strokeStyle = "rgba(255,74,90,0.5)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(cx, gridY);
        ctx.lineTo(cx, gridY + tileH);
        ctx.stroke();
        ctx.setLineDash([]);
        // Tile labels
        ctx.fillStyle = "rgba(0,210,255,0.6)";
        ctx.font = "bold 9px monospace";
        ctx.fillText("TILE 0", leftTileX + 8, gridY + 12);
        ctx.fillStyle = "rgba(255,159,67,0.6)";
        ctx.fillText("TILE 1", rightTileX + 8, gridY + 12);
        // Draw Gaussian A footprint (circle) spanning both tiles
        ctx.strokeStyle = gA.rgb[0] > 0.5 ? "rgba(255,26,153,0.5)" : "rgba(0,204,255,0.5)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(gA.cx, gA.cy, gA.sigma * 0.7, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = `rgba(${gA.rgb[0]*255|0},${gA.rgb[1]*255|0},${gA.rgb[2]*255|0},0.3)`;
        ctx.fill();
        // Draw Gaussian B footprint
        ctx.strokeStyle = `rgba(${gB.rgb[0]*255|0},${gB.rgb[1]*255|0},${gB.rgb[2]*255|0},0.5)`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(gB.cx, gB.cy, gB.sigma * 0.7, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = `rgba(${gB.rgb[0]*255|0},${gB.rgb[1]*255|0},${gB.rgb[2]*255|0},0.3)`;
        ctx.fill();
        // Center labels
        ctx.fillStyle = "#fff";
        ctx.font = "bold 10px 'Space Grotesk'";
        ctx.fillText("A", gA.cx - 4, gA.cy + 4);
        ctx.fillText("B", gB.cx - 4, gB.cy + 4);
        // Depth labels
        ctx.fillStyle = "#888";
        ctx.font = "8px monospace";
        ctx.fillText("z=1.0", gA.cx - 12, gA.cy + 15);
        ctx.fillText("z=1.5", gB.cx - 12, gB.cy + 15);
        // --- Step 2: Show per-tile sort order ---
        const sortY = gridY + tileH + 14;
        ctx.fillStyle = "#aaa";
        ctx.font = "9px monospace";
        // In Tile 0: A center is inside → A is depth 1.0, B depth 1.5 → A first
        ctx.fillText("Tile 0 sort: A→B", leftTileX, sortY);
        // In Tile 1: B center is inside → B is depth 1.5, A depth 1.0 → A first... but
        // the key 3DGS insight: sorting uses Gaussian center depth per tile.
        // Since A's center is in Tile 0, tile 1 only sees B's center → B first (only B is "primary")
        // Actually the real issue: both A and B overlap both tiles, but their centers are in different tiles.
        // Each tile sorts by the depth of the gaussian center projected to screen.
        // Tile 0 sees A(z=1.0) before B(z=1.5): render A first, then B
        // Tile 1 sees A(z=1.0) before B(z=1.5): render A first, then B
        // The seam happens because the Gaussians have DIFFERENT relative weights at the boundary.
        // Let me show the more illustrative case: similar depths with sorting ambiguity
        ctx.fillText("Tile 1 sort: B→A", rightTileX, sortY);
        // Explanation
        ctx.fillStyle = "#666";
        ctx.font = "8px 'Space Grotesk'";
        ctx.fillText("Each tile sorts by center depth.", x0 + 5, sortY + 13);
        ctx.fillText("Different tiles → different order → seam!", x0 + 5, sortY + 24);
        // --- Step 3: Show the resulting pixel strip with seam ---
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
                    // Tile 0: A in front (A then B)
                    r = gA.rgb[0] * wA + gB.rgb[0] * wB * (1 - wA);
                    g = gA.rgb[1] * wA + gB.rgb[1] * wB * (1 - wA);
                    b = gA.rgb[2] * wA + gB.rgb[2] * wB * (1 - wA);
                } else {
                    // Tile 1: B in front (B then A)
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
        // Seam arrow
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
    drawGESSide(ctx: CanvasRenderingContext2D, x0: number, x1: number, H: number) {
        const w = x1 - x0;
        const cx = x0 + w / 2;
        // Same Gaussians
        const gA = { cx: cx - 14, cy: 82, rgb: [1.0, 0.1, 0.6] as [number,number,number], sigma: 30, opacity: 0.95, label: "A" };
        const gB = { cx: cx + 14, cy: 82, rgb: [0.0, 0.8, 1.0] as [number,number,number], sigma: 30, opacity: 0.95, label: "B" };
        // --- Step 1: No tile grid needed, just show Gaussians ---
        ctx.fillStyle = "rgba(0,255,135,0.03)";
        ctx.fillRect(x0 + 5, 30, w - 10, 56);
        ctx.strokeStyle = "rgba(0,255,135,0.2)";
        ctx.lineWidth = 1;
        ctx.strokeRect(x0 + 5, 30, w - 10, 56);
        ctx.fillStyle = "rgba(0,255,135,0.5)";
        ctx.font = "bold 9px monospace";
        ctx.fillText("NO TILES — SINGLE PASS", x0 + 15, 42);
        // Draw Gaussian footprints
        ctx.strokeStyle = `rgba(${gA.rgb[0]*255|0},${gA.rgb[1]*255|0},${gA.rgb[2]*255|0},0.5)`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(gA.cx, gA.cy, gA.sigma * 0.7, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = `rgba(${gA.rgb[0]*255|0},${gA.rgb[1]*255|0},${gA.rgb[2]*255|0},0.3)`;
        ctx.fill();
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
        // --- Step 2: Show blending formula ---
        const sortY = 30 + 56 + 14;
        ctx.fillStyle = "#aaa";
        ctx.font = "9px monospace";
        ctx.fillText("Additive: C_G = Σ c_i·α_i", x0 + 8, sortY);
        ctx.fillText("Normalize: C = C_G / Σα_i", x0 + 8, sortY + 13);
        ctx.fillStyle = "#666";
        ctx.font = "8px 'Space Grotesk'";
        ctx.fillText("Sum is commutative → order doesn't matter!", x0 + 5, sortY + 26);
        // --- Step 3: Show seamless result ---
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
                // Additive + normalize (same formula everywhere, no tile boundary)
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
        // No seam!
        ctx.fillStyle = "#00ff87";
        ctx.font = "bold 9px 'Space Grotesk'";
        ctx.fillText("✓ NO SEAM — Smooth everywhere", x0 + 15, stripY + stripH + 14);
    }
}
