import { RayCompositingDiagram } from "./diagrams/RayCompositingDiagram";
import { TileBoundaryDiagram } from "./diagrams/TileBoundaryDiagram";
import { ColorLeakingDiagram } from "./diagrams/ColorLeakingDiagram";
import { Reflection3DViewer } from "./viewers/Reflection3DViewer";
import { SideBySide3DViewer } from "./viewers/SideBySide3DViewer";
import { VideoWipeViewer } from "./viewers/VideoWipeViewer";

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
    private wipeViewer!: VideoWipeViewer;

    constructor() {
        this.init();
    }

    private init() {
        this.rayDiagram = new RayCompositingDiagram();
        this.tileDiagram = new TileBoundaryDiagram();
        this.leakDiagram = new ColorLeakingDiagram();
        this.reflectionViewer = new Reflection3DViewer();
        this.sideBySideViewer = new SideBySide3DViewer();
        this.wipeViewer = new VideoWipeViewer();

        window.addEventListener("resize", this.handleResize);

        // The tab starts hidden (zero size), so canvases are first sized at zero width. Redraw
        // every diagram once the container gains a real size, else the 2D ones stay blank.
        const container = document.getElementById("comparison-container");
        if (container && "ResizeObserver" in window) {
            const ro = new ResizeObserver(() => {
                if (container.clientWidth > 0) this.handleResize();
            });
            ro.observe(container);
        }
    }

    public handleResize = () => {
        // Skip while hidden: some diagrams compute negative dimensions at zero width and throw.
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
        this.wipeViewer.handleResize();
    };
}
