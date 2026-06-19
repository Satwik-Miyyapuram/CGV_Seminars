/**
 * VideoWipeViewer renders a refnerf-style before/after wipe between two synced
 * videos rendered along the same interpolated camera path:
 *
 *     | 3DGS | GES |
 *
 * split by a single draggable vertical divider. Both videos share the same
 * camera trajectory, frame count and fps (produced by `ns-render interpolate`,
 * then ffmpeg). 3DGS is the clock; GES is re-synced to its currentTime when it
 * drifts, so the seam compares the *same pose* across the two methods.
 *
 * 3DGS is the base layer (fills the whole frame); GES is stacked on top and
 * revealed to the right of the divider via clip-path.
 *
 * Markup it attaches to (see ComparisonTab.reflectionWipe):
 *   #wipe-container
 *     video#wipe-3dgs
 *     .wipe-clip > video#wipe-ges
 *     .wipe-handle
 */
export class VideoWipeViewer {
    private container: HTMLElement | null;
    private v3dgs!: HTMLVideoElement;
    private vges!: HTMLVideoElement;
    private clip!: HTMLElement;     // wrapper around GES, masked by clip-path
    private handle!: HTMLElement;
    private dragging = false;

    constructor() {
        this.container = document.getElementById("wipe-container");
        if (!this.container) return;

        this.v3dgs = this.container.querySelector("#wipe-3dgs") as HTMLVideoElement;
        this.vges = this.container.querySelector("#wipe-ges") as HTMLVideoElement;
        this.clip = this.container.querySelector(".wipe-clip") as HTMLElement;
        this.handle = this.container.querySelector(".wipe-handle") as HTMLElement;
        if (!this.v3dgs || !this.vges || !this.clip || !this.handle) return;

        // 3DGS is the clock; keep GES re-synced if it drifts more than ~1 frame.
        this.v3dgs.addEventListener("timeupdate", () => {
            if (Math.abs(this.vges.currentTime - this.v3dgs.currentTime) > 0.04) {
                this.vges.currentTime = this.v3dgs.currentTime;
            }
        });

        this.setWipe(0.5);
        this.bindDrag();

        [this.v3dgs, this.vges].forEach((v) => {
            v.muted = true;
            v.loop = true;
            v.play().catch(() => {/* user gesture may be required */});
        });
    }

    /** Position the divider at fraction f∈[0,1]; GES shows to the right of it. */
    private setWipe(f: number) {
        const pct = Math.max(0, Math.min(1, f)) * 100;
        this.clip.style.clipPath = `inset(0 0 0 ${pct}%)`;
        this.handle.style.left = `${pct}%`;
    }

    private wipeFromClientX(clientX: number) {
        const rect = this.container!.getBoundingClientRect();
        this.setWipe((clientX - rect.left) / rect.width);
    }

    private bindDrag() {
        const start = (e: PointerEvent) => { this.dragging = true; this.wipeFromClientX(e.clientX); };
        const move = (e: PointerEvent) => { if (this.dragging) this.wipeFromClientX(e.clientX); };
        const end = () => { this.dragging = false; };

        this.container!.addEventListener("pointerdown", start);
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", end);
    }

    /** Videos are CSS-sized; nothing to recompute. Kept for the manager's uniform API. */
    public handleResize() {}
}
