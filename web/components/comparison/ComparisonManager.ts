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
    }

    /**
     * Propagate resize events to each sub-component so they redraw correctly.
     */
    public handleResize = () => {
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
