/**
 * Base for the canvas-2D comparison diagrams. Owns the `<canvas>` element and its 2D context,
 * and provides the device-pixel-ratio-aware `resize()` they all share. Subclasses wire up their
 * own controls in the constructor and implement `draw()`.
 */
export abstract class Canvas2DDiagram {
    protected canvas: HTMLCanvasElement;
    protected ctx: CanvasRenderingContext2D;

    /** Look up the canvas by id and grab its 2D context. */
    constructor(canvasId: string) {
        this.canvas = document.getElementById(canvasId) as HTMLCanvasElement;
        this.ctx = this.canvas.getContext("2d")!;
    }

    /** Resize the backing store to the element's CSS size × devicePixelRatio (crisp on HiDPI). */
    public resize() {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.ctx.scale(dpr, dpr);
    }

    /** Render the diagram at the current size and control state. */
    public abstract draw(): void;
}
