export interface Splat {
    x: number;          // world-space horizontal position
    depth: number;      // world-space depth (camera z)
    color: string;      // CSS rgba string for canvas drawing
    rgb: [number, number, number]; // normalized [0..1]
    opacity: number;
    sigma: number;      // Gaussian std-dev in screen px
    isSurfel?: boolean; // true = surface element (GES first pass)
    name: string;
}
// ============================================================
// Helper: evaluate 1D Gaussian weight at screen coordinate
// ============================================================
export function gaussWeight(screenX: number, splatProjX: number, sigma: number, opacity: number): number {
    const d = screenX - splatProjX;
    return opacity * Math.exp(-0.5 * (d * d) / (sigma * sigma));
}
// ============================================================
// Helper: project world (x, z) through a 2D orthographic camera
// at angle θ. Returns { depth, projX }.
// View direction = (-sinθ, cosθ),  Right vector = (cosθ, sinθ)
// ============================================================
export function orthoProject(worldX: number, worldZ: number, angleRad: number) {
    const depth = -worldX * Math.sin(angleRad) + worldZ * Math.cos(angleRad);
    const projX  =  worldX * Math.cos(angleRad) + worldZ * Math.sin(angleRad);
    return { depth, projX };
}
// ============================================================
// Shared splat definitions — CLUSTERED for maximum overlap
// All at similar x positions so they overlap heavily on screen.
// Different depths so sort order flips with small camera rotation.
// ============================================================
export const SPLATS: Splat[] = [
    { x: 175, depth: 1.8, color: "rgba(0,210,255,0.9)",  rgb: [0.0,0.82,1.0],  opacity: 0.90, sigma: 40, isSurfel: true,  name: "Cyan"   },
    { x: 195, depth: 2.1, color: "rgba(255,74,90,0.9)",   rgb: [1.0,0.29,0.35], opacity: 0.85, sigma: 35, isSurfel: false, name: "Red"    },
    { x: 185, depth: 1.5, color: "rgba(255,159,67,0.9)",  rgb: [1.0,0.62,0.26], opacity: 0.88, sigma: 38, isSurfel: false, name: "Orange" },
    { x: 205, depth: 2.5, color: "rgba(0,255,135,0.9)",   rgb: [0.0,1.0,0.53],  opacity: 0.85, sigma: 32, isSurfel: false, name: "Green"  },
];
// World-space mapping constants
export const CENTER_X = 190;
export const CENTER_Z = 2.0;
export const SCALE_Z  = 80;  // tighter scale so depth differences are more dramatic
