import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

/**
 * The GES Web Viewer. Surfels and gaussians are loaded as one shared scene and drawn together
 * with a real z-buffer: the opaque surfels write depth (render first) so the hardware occludes
 * gaussians behind them — no custom two-pass. Driven entirely by the glassmorphic #ui panel:
 *   - `sceneSplit`        → (re)load a scene (dispatched by the file input + autoloader)
 *   - `viewerMode`        → show surfels / gaussians (the Show Surfels/Gaussians checkboxes)
 *   - `viewerBackground`  → background colour (from config.json)
 */
class CombinedViewerApp {
    private container: HTMLElement;

    private scene!: THREE.Scene;
    private camera!: THREE.PerspectiveCamera;
    private renderer!: THREE.WebGLRenderer;
    private controls!: OrbitControls;
    private initialized = false;
    private isAnimating = false;

    private surfelDropIn: any | null = null;
    private gaussianDropIn: any | null = null;

    // Scene bounds (from the sceneSplit event) for framing + zoom scale.
    private sceneCenter = new THREE.Vector3(0, 0, 0);
    private sceneRadius = 4;
    private zoomMinDist = 0.1; // closest approach to the pivot + zoom speed floor

    constructor() {
        this.container = document.getElementById('viewer') as HTMLElement;
        if (!this.container) return;

        window.addEventListener('sceneSplit', (e: any) => {
            const d = e.detail || {};
            if (Array.isArray(d.center)) this.sceneCenter.set(d.center[0], d.center[1], d.center[2]);
            if (typeof d.radius === 'number' && d.radius > 0) this.sceneRadius = d.radius;
            this.loadScenes(d.surfelUrl ?? null, d.gaussianUrl ?? null);
        });

        window.addEventListener('viewerMode', (e: any) => {
            const d = e.detail || {};
            this.setMode(d.surfels !== false, d.gaussians !== false);
        });

        window.addEventListener('viewerBackground', (e: any) => {
            const c = e.detail;
            if (Array.isArray(c) && this.renderer) {
                this.renderer.setClearColor(new THREE.Color(c[0], c[1], c[2]), 1);
            }
        });
    }

    /** Set up the shared Three.js scene, camera, renderer, and Blender-style controls (once). */
    private initThreeJS() {
        this.scene = new THREE.Scene();

        const w = this.container.clientWidth || window.innerWidth;
        const h = this.container.clientHeight || window.innerHeight;

        this.camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 1000);
        // 3DGS/COLMAP scenes are Y-down; a clean -Y up keeps orbit/pan level.
        this.camera.up.set(0, -1, 0);

        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        this.renderer.setSize(w, h);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setClearColor(0x000000, 1);
        this.container.appendChild(this.renderer.domElement);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        // No damping → movement tracks the mouse directly at a constant rate (no inertia).
        this.controls.enableDamping = false;
        this.controls.enablePan = true;
        this.controls.panSpeed = 1.0;            // pan 1:1 with the mouse
        this.controls.screenSpacePanning = true;
        this.controls.rotateSpeed = 0.6;
        // Blender-style: left = orbit, right = pan. Zoom is our own wheel handler.
        this.controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, RIGHT: THREE.MOUSE.PAN };
        this.controls.enableZoom = false;

        // Custom dolly zoom toward the (fixed) pivot: speed is on a curve (proportional to
        // distance, eases as you approach) but floored so it never crawls. Zoom-in stops just
        // short of the pivot so it never crosses it (crossing makes OrbitControls' lookAt snap
        // the camera 180° — the "flip"). You still fly through the surfaces around the pivot.
        this.renderer.domElement.addEventListener('wheel', (e: WheelEvent) => {
            e.preventDefault();
            const outward = new THREE.Vector3().subVectors(this.camera.position, this.controls.target);
            const dist = outward.length();
            if (dist < 1e-6) return;
            outward.divideScalar(dist);
            const step = Math.max(dist * 0.15, this.zoomMinDist);
            const newDist = e.deltaY < 0
                ? Math.max(dist - step, this.zoomMinDist)
                : dist + step;
            this.camera.position.copy(this.controls.target).addScaledVector(outward, newDist);
            this.controls.update();
        }, { passive: false });

        window.addEventListener('resize', () => {
            if (!this.renderer) return;
            const cw = this.container.clientWidth || window.innerWidth;
            const ch = this.container.clientHeight || window.innerHeight;
            this.camera.aspect = cw / ch;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(cw, ch);
        });

        this.initialized = true;
    }

    /** Frame the camera to the current scene bounds (called on every (re)load). */
    private frameCamera() {
        const r = this.sceneRadius > 0 ? this.sceneRadius : 4;
        const dist = (r / Math.sin((this.camera.fov * Math.PI) / 180 / 2)) * 1.1;

        this.camera.near = Math.max(r * 0.0002, 0.0005);
        this.camera.far = (dist + r) * 5000;
        this.camera.updateProjectionMatrix();
        this.camera.position.set(this.sceneCenter.x, this.sceneCenter.y, this.sceneCenter.z + dist);

        this.controls.target.copy(this.sceneCenter);
        this.zoomMinDist = r * 0.02;
        this.controls.update();
    }

    /**
     * Configure a DropInViewer's splat material. Opaque surfels write depth and render first so
     * they occlude gaussians behind them; gaussians depth-test but don't write (no popping).
     */
    private configureMaterial(dropInViewer: any, isSurfel: boolean, retries = 30) {
        const material = dropInViewer.viewer?.splatMesh?.material;
        const mesh = dropInViewer.viewer?.splatMesh;
        if (!material) {
            if (retries > 0) requestAnimationFrame(() => this.configureMaterial(dropInViewer, isSurfel, retries - 1));
            else console.warn('[Viewer] Could not access splatMesh.material after retries');
            return;
        }

        material.depthTest = true;
        material.depthWrite = isSurfel;            // only the opaque surfels write depth
        if (mesh) mesh.renderOrder = isSurfel ? 0 : 1; // surfels first → their depth is ready

        if (isSurfel) {
            material.onBeforeCompile = (shader: any) => {
                // GES opaque flat-disc surfel: constant opacity inside the 3.33σ boundary.
                shader.fragmentShader = shader.fragmentShader.replace(
                    /float\s+opacity\s*=\s*exp\(\s*-0\.5\s*\*\s*A\s*\)\s*\*\s*vColor\.a\s*;/g,
                    `float g = exp(-0.5 * A);
                    if (g < 1.0 / 255.0) discard;
                    float val = 255.0 * vColor.a * g;
                    float opacity = min(0.99, val);
                    if (opacity < 0.002) discard;`
                );
                shader.fragmentShader = shader.fragmentShader.replace(
                    /gl_FragColor\s*=\s*vec4\s*\(\s*color\.rgb\s*,\s*opacity\s*\)\s*;/g,
                    `gl_FragColor = vec4(color.rgb * opacity, opacity);`
                );
            };
        }
        material.needsUpdate = true;
    }

    private startAnimation() {
        if (this.isAnimating) return;
        this.isAnimating = true;
        const animate = () => {
            if (!this.isAnimating) return;
            requestAnimationFrame(animate);
            this.controls.update();
            this.renderer.render(this.scene, this.camera);
        };
        animate();
    }

    private unloadScenes() {
        if (this.surfelDropIn) { this.scene.remove(this.surfelDropIn); this.surfelDropIn = null; }
        if (this.gaussianDropIn) { this.scene.remove(this.gaussianDropIn); this.gaussianDropIn = null; }
    }

    public async loadScenes(surfelUrl: string | null, gaussianUrl: string | null) {
        if (!surfelUrl && !gaussianUrl) return;
        if (!this.initialized) {
            this.initThreeJS();
            this.startAnimation();
        }
        this.unloadScenes();

        if (surfelUrl) {
            this.surfelDropIn = new GaussianSplats3D.DropInViewer({ gpuAcceleratedSort: false });
            await this.surfelDropIn.addSplatScenes([{ path: surfelUrl, format: GaussianSplats3D.SceneFormat.Ply }]);
            this.scene.add(this.surfelDropIn);
            this.configureMaterial(this.surfelDropIn, true);
        }
        if (gaussianUrl) {
            this.gaussianDropIn = new GaussianSplats3D.DropInViewer({ gpuAcceleratedSort: false });
            await this.gaussianDropIn.addSplatScenes([{ path: gaussianUrl, format: GaussianSplats3D.SceneFormat.Ply }]);
            this.scene.add(this.gaussianDropIn);
            this.configureMaterial(this.gaussianDropIn, false);
        }

        this.frameCamera();
        this.applyModeFromDOM();
        console.log('[Viewer] Scene loaded.');
    }

    private setMode(showSurfels: boolean, showGaussians: boolean) {
        if (this.surfelDropIn) this.surfelDropIn.visible = showSurfels;
        if (this.gaussianDropIn) this.gaussianDropIn.visible = showGaussians;
    }

    /** Sync visibility to the current #ui checkbox state. */
    private applyModeFromDOM() {
        const s = document.getElementById('toggleSurfels') as HTMLInputElement | null;
        const g = document.getElementById('toggleGaussians') as HTMLInputElement | null;
        this.setMode(s ? s.checked : true, g ? g.checked : true);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new CombinedViewerApp();
});
