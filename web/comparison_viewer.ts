import { RayCompositingDiagram } from "./diagrams/RayCompositingDiagram";
import { TileBoundaryDiagram } from "./diagrams/TileBoundaryDiagram";
import { ColorLeakingDiagram } from "./diagrams/ColorLeakingDiagram";
import { SideBySide3DViewer } from "./diagrams/SideBySide3DViewer";
// ============================================================
// Initialize everything
// ============================================================
const rayDiagram = new RayCompositingDiagram();
const tileDiagram = new TileBoundaryDiagram();
const leakDiagram = new ColorLeakingDiagram();
const sbs3d = new SideBySide3DViewer();
window.addEventListener("resize", () => {
    rayDiagram.resize(); rayDiagram.draw();
    tileDiagram.resize(); tileDiagram.draw();
    leakDiagram.resize(); leakDiagram.draw();
    sbs3d.handleResize();
});
setTimeout(() => window.dispatchEvent(new Event("resize")), 150);