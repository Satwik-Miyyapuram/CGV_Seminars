import { RayCompositingDiagram } from "./diagrams/RayCompositingDiagram";
import { TileBoundaryDiagram } from "./diagrams/TileBoundaryDiagram";
import { ColorLeakingDiagram } from "./diagrams/ColorLeakingDiagram";
import { Reflection3DViewer } from "./diagrams/Reflection3DViewer";
import { SideBySide3DViewer } from "./diagrams/SideBySide3DViewer";

// Wait for DOM
document.addEventListener("DOMContentLoaded", () => {
    // 1. Initialize diagrams
    const rayDiagram = new RayCompositingDiagram();
    const tileDiagram = new TileBoundaryDiagram();
    const leakDiagram = new ColorLeakingDiagram();
    const reflection3D = new Reflection3DViewer();
    const sbs3d = new SideBySide3DViewer();

    window.addEventListener("resize", () => {
        rayDiagram.resize(); rayDiagram.draw();
        tileDiagram.resize(); tileDiagram.draw();
        leakDiagram.resize(); leakDiagram.draw();
        reflection3D.handleResize();
        sbs3d.handleResize();
    });
});
setTimeout(() => window.dispatchEvent(new Event("resize")), 150);