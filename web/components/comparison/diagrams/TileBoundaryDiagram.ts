/**
 * TileBoundaryDiagram — an interactive, beginner-friendly explanation of why 3DGS's
 * tile-based rasterization can produce a visible "seam" while GES (no tiles) does not.
 *
 * The SAME two blobs (A and B) are shown in both rows; only the algorithm differs:
 *   - 3DGS (top): the screen is split into tiles. Each tile only draws the blobs assigned
 *     to it. The user drags B across the tile border; when B sits inside one tile, the
 *     neighbouring tile stops drawing it, so the two tiles disagree → a hard SEAM at the
 *     border. (The depth sort order is the same global order in every tile — the seam is
 *     about WHICH blobs a tile draws, not their order.)
 *   - GES (bottom): no tiles. The same A and B are blended everywhere → smooth, no seam.
 *
 * Teaching note: a single pair of blobs is used so the mechanism is legible. The tile
 * "assignment" here is based on each blob's bright core, so B's soft glow can visibly spill
 * across the border while the neighbour tile still ignores it — making the seam easy to see.
 * Real 3DGS uses a 3σ bounding box and many overlapping blobs, where these per-tile steps
 * accumulate (and flicker as the camera moves) into the popping artifacts described in the paper.
 */

interface Blob {
    x: number;
    y: number;
    rgb: [number, number, number];
    sigma: number;
    opacity: number;
    depth: number;   // global center depth = sort key
    name: string;
}

interface Scene {
    xL: number; xR: number; boundary: number; tileW: number;
    A: Blob; B: Blob;
    memb0: boolean; memb1: boolean;  // is B assigned to Tile 0 / Tile 1
    seam: boolean;
}

export class TileBoundaryDiagram {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private slider: HTMLInputElement | null;
    private sliderLabel: HTMLElement | null;
    private status: HTMLElement | null;

    // Vertical layout (CSS px). Two stacked rows: 3DGS on top, GES below.
    private readonly A_titleY = 18;
    private readonly A_tileY = 28;  private readonly A_tileH = 80;   // 28..108
    private readonly A_badgeY = 122;
    private readonly A_barY = 130;  private readonly barH = 36;       // 130..166
    private readonly A_capY = 178;
    private readonly dividerY = 188;
    private readonly B_titleY = 200;
    private readonly B_tileY = 210; private readonly B_tileH = 80;    // 210..290
    private readonly B_badgeY = 304;
    private readonly B_barY = 312;                                    // 312..348
    private readonly B_capY = 360;

    private readonly marginX = 24;
    // Tile "assignment" uses the blob's bright core (fraction of sigma), so a blob's wide
    // glow can spill past the border while the neighbour tile still skips it.
    private readonly CORE = 0.45;

    constructor() {
        this.canvas = document.getElementById("tile-canvas") as HTMLCanvasElement;
        this.ctx = this.canvas.getContext("2d")!;
        this.slider = document.getElementById("tileSplatSlider") as HTMLInputElement | null;
        this.sliderLabel = document.getElementById("tileSplatValue");
        this.status = document.getElementById("tileStatus");

        this.slider?.addEventListener("input", () => this.draw());

        this.resize();
        this.draw();
    }

    public resize() {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.ctx.scale(dpr, dpr);
    }

    /** Soft Gaussian weight of a blob at a screen pixel (used to render the result bars). */
    private weight(b: Blob, wx: number, wy: number): number {
        const d2 = (wx - b.x) ** 2 + (wy - b.y) ** 2;
        return b.opacity * Math.exp(-0.5 * d2 / (b.sigma ** 2));
    }

    /** Build the shared scene (same A,B for both rows) from the current slider value. */
    private buildScene(W: number, rowY: number): Scene {
        const xL = this.marginX;
        const xR = W - this.marginX;
        const boundary = (xL + xR) / 2;
        const tileW = Math.max(20, (xR - xL) / 2);

        // A: magenta, NEAR, straddles the border — always drawn by both tiles.
        const A: Blob = {
            x: boundary, y: rowY, rgb: [1.0, 0.12, 0.62],
            sigma: Math.max(6, Math.min(46, tileW * 0.5)), opacity: 0.72, depth: 1.0, name: "A",
        };

        // B: cyan, FAR, moves with the slider across the border.
        const t = this.slider ? parseFloat(this.slider.value) / 100 : 0.8;
        const bx = xL + (0.14 + 0.72 * t) * (xR - xL);
        const B: Blob = {
            x: bx, y: rowY, rgb: [0.0, 0.82, 1.0],
            sigma: Math.max(5, Math.min(34, tileW * 0.42)), opacity: 0.95, depth: 1.5, name: "B",
        };

        // Tile assignment by bright core.
        const core = this.CORE * B.sigma;
        const memb0 = (B.x - core) <= boundary;      // core reaches into Tile 0
        const memb1 = (B.x + core) >= boundary;       // core reaches into Tile 1
        const seam = memb0 !== memb1;                 // B drawn by exactly one tile

        return { xL, xR, boundary, tileW, A, B, memb0, memb1, seam };
    }

    public draw() {
        const W = this.canvas.width / (window.devicePixelRatio || 1);
        const ctx = this.ctx;
        ctx.clearRect(0, 0, W, this.canvas.height);

        // Canvas not laid out yet (e.g. the comparison tab is still hidden → zero width).
        // Bail out; the ResizeObserver/window-resize will call draw() again once it has a real
        // size. (Drawing here would otherwise compute degenerate negative sizes.)
        if (W < 80) return;

        const sceneA = this.buildScene(W, this.A_tileY + this.A_tileH / 2);
        const sceneB = this.buildScene(W, this.B_tileY + this.B_tileH / 2);

        this.drawTiledRow(ctx, sceneA);
        this.drawSmoothRow(ctx, sceneB);

        // Divider between the two rows
        ctx.strokeStyle = "rgba(255,255,255,0.08)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(this.marginX, this.dividerY);
        ctx.lineTo(W - this.marginX, this.dividerY);
        ctx.stroke();

        this.updateText(sceneA);
    }

    /** Draw a soft round blob with a radial-gradient glow, clipped to the given band. */
    private drawBlob(ctx: CanvasRenderingContext2D, b: Blob, bandY: number, bandH: number, dim = false) {
        const R = b.sigma * 2.0;
        const col = `${b.rgb[0] * 255 | 0},${b.rgb[1] * 255 | 0},${b.rgb[2] * 255 | 0}`;
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, bandY, this.canvas.width, bandH);
        ctx.clip();
        const g = ctx.createRadialGradient(b.x, b.y, 0, b.x, b.y, R);
        const a = dim ? 0.18 : 0.95;
        g.addColorStop(0, `rgba(${col},${a})`);
        g.addColorStop(0.5, `rgba(${col},${a * 0.5})`);
        g.addColorStop(1, `rgba(${col},0)`);
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(b.x, b.y, R, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

        // Label chip
        ctx.fillStyle = dim ? "rgba(255,255,255,0.4)" : "#fff";
        ctx.font = "bold 13px 'Space Grotesk'";
        ctx.textAlign = "center";
        ctx.fillText(b.name, b.x, b.y + 5);
        ctx.textAlign = "left";
    }

    /** Top row: 3DGS with two tiles. */
    private drawTiledRow(ctx: CanvasRenderingContext2D, s: Scene) {
        const { xL, xR, boundary, A, B, memb0, memb1 } = s;
        const ty = this.A_tileY, th = this.A_tileH;

        // Row title
        ctx.fillStyle = "#ff7a85";
        ctx.font = "bold 12px 'Space Grotesk'";
        ctx.textAlign = "left";
        ctx.fillText("① 3DGS — screen split into TILES", xL, this.A_titleY);

        // Stage label (right-aligned so it clears the row title)
        ctx.textAlign = "right";
        ctx.fillStyle = "rgba(255, 255, 255, 0.35)";
        ctx.font = "8px monospace";
        ctx.fillText("INPUT: Tile Setup", xR, this.A_titleY);
        ctx.textAlign = "left";

        // Tile fills + outlines
        ctx.fillStyle = "rgba(0,210,255,0.04)";
        ctx.fillRect(xL, ty, boundary - xL, th);
        ctx.fillStyle = "rgba(255,159,67,0.04)";
        ctx.fillRect(boundary, ty, xR - boundary, th);
        ctx.lineWidth = 1;
        ctx.strokeStyle = "rgba(0,210,255,0.35)";
        ctx.strokeRect(xL, ty, boundary - xL, th);
        ctx.strokeStyle = "rgba(255,159,67,0.35)";
        ctx.strokeRect(boundary, ty, xR - boundary, th);

        // Tile labels
        ctx.font = "bold 10px monospace";
        ctx.fillStyle = "rgba(0,210,255,0.7)";
        ctx.fillText("TILE 0", xL + 8, ty + 14);
        ctx.fillStyle = "rgba(255,159,67,0.7)";
        ctx.fillText("TILE 1", boundary + 8, ty + 14);

        // Blob A (drawn by both tiles), then B — dimmed inside any tile that skips it.
        this.drawBlob(ctx, A, ty, th, false);
        // B drawn solid in the tile(s) that keep it; a dim "ghost" in the tile that skips it.
        if (memb0 && memb1) {
            this.drawBlob(ctx, B, ty, th, false);
        } else {
            // Solid half (kept tile) + dim half (skipped tile), split at the border.
            const keptLeft = memb0;
            // dim ghost everywhere first, then redraw solid clipped to the kept tile
            this.drawBlob(ctx, B, ty, th, true);
            ctx.save();
            ctx.beginPath();
            if (keptLeft) ctx.rect(xL, ty, boundary - xL, th);
            else ctx.rect(boundary, ty, xR - boundary, th);
            ctx.clip();
            this.drawBlob(ctx, B, ty, th, false);
            ctx.restore();

            // "skipped here" tag in the tile that drops B
            const skipX = keptLeft ? (boundary + (xR - boundary) / 2) : (xL + (boundary - xL) / 2);
            ctx.fillStyle = "#ff4a5a";
            ctx.font = "bold 9px 'Space Grotesk'";
            ctx.textAlign = "center";
            ctx.fillText("✗ B skipped here", skipX, ty + th - 8);
            ctx.textAlign = "left";
        }

        // Tile boundary line (dotted red)
        ctx.strokeStyle = "rgba(255,74,90,0.7)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(boundary, ty);
        ctx.lineTo(boundary, this.A_barY + this.barH);
        ctx.stroke();
        ctx.setLineDash([]);

        // Badges: what each tile draws
        ctx.font = "10px monospace";
        ctx.textAlign = "left";
        ctx.fillStyle = "#cfd2dc";
        ctx.fillText(`Tile 0 draws: A${memb0 ? " + B" : ""}`, xL, this.A_badgeY);
        ctx.fillText(`Tile 1 draws: A${memb1 ? " + B" : ""}`, boundary + 8, this.A_badgeY);

        // Result bar: per pixel, blend the blobs that pixel's tile draws (front-to-back).
        this.renderBar(ctx, xL, xR, this.A_barY, (wx) => {
            const inLeft = wx < boundary;
            const drawsB = inLeft ? memb0 : memb1;
            const members = drawsB ? [A, B] : [A];
            members.sort((p, q) => p.depth - q.depth);
            let T = 1, r = 0, g = 0, b = 0;
            for (const m of members) {
                const a = this.weight(m, wx, A.y);
                r += m.rgb[0] * a * T; g += m.rgb[1] * a * T; b += m.rgb[2] * a * T;
                T *= (1 - a);
            }
            return [r, g, b];
        });

        // Output stage label, overlaid on the result bar (own baseline → no clash with badges)
        ctx.textAlign = "right";
        ctx.fillStyle = "rgba(255, 255, 255, 0.6)";
        ctx.font = "8px monospace";
        ctx.fillText("OUTPUT: Rendered Pixels", xR - 4, this.A_barY + 13);
        ctx.textAlign = "left";

        // Seam marker
        if (s.seam) {
            ctx.strokeStyle = "#ff4a5a";
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            ctx.moveTo(boundary, this.A_barY - 2);
            ctx.lineTo(boundary, this.A_barY + this.barH + 2);
            ctx.stroke();
            ctx.fillStyle = "#ff4a5a";
            ctx.font = "bold 11px 'Space Grotesk'";
            ctx.textAlign = "center";
            ctx.fillText("◀ SEAM (hard edge!) ▶", boundary, this.A_capY);
            ctx.textAlign = "left";
        } else {
            ctx.fillStyle = "#00ff87";
            ctx.font = "bold 11px 'Space Grotesk'";
            ctx.textAlign = "center";
            ctx.fillText("B touches both tiles → no seam (keep dragging…)", (xL + xR) / 2, this.A_capY);
            ctx.textAlign = "left";
        }
    }

    /** Bottom row: GES with no tiles. */
    private drawSmoothRow(ctx: CanvasRenderingContext2D, s: Scene) {
        const { xL, xR, A, B } = s;
        const ty = this.B_tileY, th = this.B_tileH;

        ctx.fillStyle = "#46e0a0";
        ctx.font = "bold 12px 'Space Grotesk'";
        ctx.textAlign = "left";
        ctx.fillText("② GES — NO tiles, one smooth pass", xL, this.B_titleY);

        // Stage label (right-aligned so it clears the row title)
        ctx.textAlign = "right";
        ctx.fillStyle = "rgba(255, 255, 255, 0.35)";
        ctx.font = "8px monospace";
        ctx.fillText("INPUT: Full Region Setup", xR, this.B_titleY);
        ctx.textAlign = "left";

        // One continuous region (no divider)
        ctx.fillStyle = "rgba(0,255,135,0.04)";
        ctx.fillRect(xL, ty, xR - xL, th);
        ctx.strokeStyle = "rgba(0,255,135,0.25)";
        ctx.lineWidth = 1;
        ctx.strokeRect(xL, ty, xR - xL, th);
        ctx.fillStyle = "rgba(0,255,135,0.7)";
        ctx.font = "bold 10px monospace";
        ctx.fillText("ONE REGION — every blob drawn everywhere", xL + 8, ty + 14);

        // Same blobs, both fully drawn
        this.drawBlob(ctx, A, ty, th, false);
        this.drawBlob(ctx, B, ty, th, false);

        ctx.font = "10px monospace";
        ctx.fillStyle = "#cfd2dc";
        ctx.fillText("Everywhere draws: A + B   (additive: C = Σ cᵢ·αᵢ / Σ αᵢ)", xL, this.B_badgeY);

        // Result bar: additive normalized blend of BOTH blobs at every pixel.
        this.renderBar(ctx, xL, xR, this.B_barY, (wx) => {
            let sumA = 0, cr = 0, cg = 0, cb = 0;
            for (const m of [A, B]) {
                const a = this.weight(m, wx, A.y);
                cr += m.rgb[0] * a; cg += m.rgb[1] * a; cb += m.rgb[2] * a; sumA += a;
            }
            if (sumA <= 0.004) return [0, 0, 0];
            const agg = 1 - Math.exp(-sumA);
            return [cr / sumA * agg, cg / sumA * agg, cb / sumA * agg];
        });

        // Output stage label, overlaid on the result bar
        ctx.textAlign = "right";
        ctx.fillStyle = "rgba(255, 255, 255, 0.6)";
        ctx.font = "8px monospace";
        ctx.fillText("OUTPUT: Rendered Pixels", xR - 4, this.B_barY + 13);
        ctx.textAlign = "left";

        ctx.fillStyle = "#00ff87";
        ctx.font = "bold 11px 'Space Grotesk'";
        ctx.textAlign = "center";
        ctx.fillText("✓ Always smooth — no seam, ever", (xL + xR) / 2, this.B_capY);
        ctx.textAlign = "left";
    }

    /** Render a full-width result bar by sampling colorAt(wx) and filling the bar height. */
    private renderBar(
        ctx: CanvasRenderingContext2D,
        x0: number, x1: number, barY: number,
        colorAt: (wx: number) => [number, number, number]
    ) {
        const imgW = Math.max(1, Math.ceil(x1 - x0));
        const img = ctx.createImageData(imgW, this.barH);
        const data = img.data;
        for (let lx = 0; lx < imgW; lx++) {
            const [r, g, b] = colorAt(x0 + lx);
            const R = Math.min(255, r * 255 | 0), G = Math.min(255, g * 255 | 0), B = Math.min(255, b * 255 | 0);
            for (let y = 0; y < this.barH; y++) {
                const off = (y * imgW + lx) * 4;
                data[off] = R; data[off + 1] = G; data[off + 2] = B; data[off + 3] = 255;
            }
        }
        const tmp = document.createElement("canvas");
        tmp.width = imgW; tmp.height = this.barH;
        tmp.getContext("2d")!.putImageData(img, 0, 0);
        ctx.drawImage(tmp, x0, barY);
        ctx.strokeStyle = "rgba(255,255,255,0.12)";
        ctx.lineWidth = 1;
        ctx.strokeRect(x0, barY, imgW, this.barH);
    }

    /** Update the slider chip + the plain-language status box below the controls. */
    private updateText(s: Scene) {
        if (this.sliderLabel) {
            this.sliderLabel.textContent = !s.memb0 ? "in Tile 1" : !s.memb1 ? "in Tile 0" : "on the border";
        }
        if (this.status) {
            if (s.seam) {
                const skipped = s.memb1 ? "Tile 0" : "Tile 1";
                const kept = s.memb1 ? "Tile 1" : "Tile 0";
                this.status.style.borderLeftColor = "#ff4a5a";
                this.status.style.background = "rgba(255,74,90,0.08)";
                this.status.innerHTML =
                    `<strong style="color:#ff4a5a">Seam!</strong> B's core sits in <b>${kept}</b>, so <b>${skipped}</b> doesn't draw it — ` +
                    `even though B's glow clearly spills across the border. The two tiles paint different colours, ` +
                    `meeting in a hard line. The GES row below draws B everywhere, so it stays smooth.`;
            } else {
                this.status.style.borderLeftColor = "#00ff87";
                this.status.style.background = "rgba(0,255,135,0.07)";
                this.status.innerHTML =
                    `<strong style="color:#00ff87">No seam right now.</strong> B straddles the border, so <b>both</b> tiles draw it. ` +
                    `Keep dragging: the instant B's core leaves a tile, that tile drops B and the seam snaps back — ` +
                    `that sudden change is exactly the "popping" 3DGS suffers as the camera moves.`;
            }
        }
    }
}
