import { RayCompositingDiagram } from "./diagrams/RayCompositingDiagram";
import { TileBoundaryDiagram } from "./diagrams/TileBoundaryDiagram";
import { ColorLeakingDiagram } from "./diagrams/ColorLeakingDiagram";
import { Reflection3DViewer } from "./viewers/Reflection3DViewer";
import { SideBySide3DViewer } from "./viewers/SideBySide3DViewer";

/**
 * ComparisonManager coordinates the initialization and resize handling of all
 * comparison diagrams and 3D viewports under the "Artifact Comparison" tab.
 */
export class ComparisonManager {
    private rayDiagram!: RayCompositingDiagram;
    private tileDiagram!: TileBoundaryDiagram;
    private leakDiagram!: ColorLeakingDiagram;
    private reflectionViewer!: Reflection3DViewer;
    private sideBySideViewer!: SideBySide3DViewer;

    constructor() {
        this.init();
    }

    /**
     * Instantiate all diagrams and 3D viewers.
     */
    private init() {
        console.log("[ComparisonManager] Initializing comparison tab diagrams and viewers...");
        this.rayDiagram = new RayCompositingDiagram();
        this.tileDiagram = new TileBoundaryDiagram();
        this.leakDiagram = new ColorLeakingDiagram();
        this.reflectionViewer = new Reflection3DViewer();
        this.sideBySideViewer = new SideBySide3DViewer();

        // Listen for standard resize event to trigger redraws
        window.addEventListener("resize", this.handleResize);

        // The comparison tab starts hidden (display:none), so the canvases are first sized
        // while their container has zero size. Watch the container so that when the tab is
        // shown (or otherwise resized) every diagram re-measures and redraws at the correct
        // size — otherwise the 2D diagrams would stay blank until a window resize.
        const container = document.getElementById("comparison-container");
        if (container && "ResizeObserver" in window) {
            const ro = new ResizeObserver(() => {
                if (container.clientWidth > 0) this.handleResize();
            });
            ro.observe(container);
        }
    }

    /**
     * Propagate resize events to each sub-component so they redraw correctly.
     */
    public handleResize = () => {
        // Skip while the comparison tab is hidden (zero size): some diagrams compute negative
        // canvas dimensions at zero width and would throw.
        const container = document.getElementById("comparison-container");
        if (container && container.clientWidth === 0) return;

        this.rayDiagram.resize();
        this.rayDiagram.draw();

        this.tileDiagram.resize();
        this.tileDiagram.draw();

        this.leakDiagram.resize();
        this.leakDiagram.draw();

        this.reflectionViewer.handleResize();
        this.sideBySideViewer.handleResize();
    };
}
