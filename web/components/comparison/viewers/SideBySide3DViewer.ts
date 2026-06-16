import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { Splat, SPLATS } from "../shared";

/**
 * SideBySide3DViewer creates two interactive, synchronized 3D viewports.
 * The left viewport shows the 3DGS rendering pipeline (alpha blending, order-dependent sorting).
 * The right viewport shows the GES rendering pipeline (opaque surfel depth culling + additive blending).
 * Camera position and orientation are synchronized interactively.
 */
export class SideBySide3DViewer {
    private scene3dgs!: THREE.Scene;
    private camera3dgs!: THREE.PerspectiveCamera;
    private renderer3dgs!: THREE.WebGLRenderer;
    private controls3dgs!: OrbitControls;

    private sceneGes!: THREE.Scene;
    private cameraGes!: THREE.PerspectiveCamera;
    private rendererGes!: THREE.WebGLRenderer;
    private controlsGes!: OrbitControls;

    private meshes3dgs: { splat: Splat, mesh: THREE.Mesh }[] = [];
    private meshesGes: { splat: Splat, mesh: THREE.Mesh }[] = [];

    // Track active viewport interaction to sync cameras from active to inactive
    private activeViewport: "3dgs" | "ges" | null = null;

    constructor() {
        this.init3DGS();
        this.initGES();
        this.bindControls();
        this.animate();
        window.addEventListener("resize", () => this.handleResize());
    }

    /**
     * Set up the 3DGS side.
     */
    private init3DGS() {
        const container = document.getElementById("viewport-3dgs")!;
        this.scene3dgs = new THREE.Scene();
        this.scene3dgs.background = new THREE.Color(0x0a0a10);
        
        this.camera3dgs = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 100);
        this.camera3dgs.position.set(0, 0, 3);
        
        this.renderer3dgs = new THREE.WebGLRenderer({ antialias: true });
        this.renderer3dgs.setSize(container.clientWidth, container.clientHeight);
        this.renderer3dgs.setPixelRatio(window.devicePixelRatio);
        container.appendChild(this.renderer3dgs.domElement);
        
        this.controls3dgs = new OrbitControls(this.camera3dgs, this.renderer3dgs.domElement);
        this.controls3dgs.enableDamping = true;
        this.controls3dgs.dampingFactor = 0.05;
        
        this.buildScene(this.scene3dgs, false);
    }

    /**
     * Set up the GES side.
     */
    private initGES() {
        const container = document.getElementById("viewport-ges")!;
        this.sceneGes = new THREE.Scene();
        this.sceneGes.background = new THREE.Color(0x0a0a10);
        
        this.cameraGes = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 100);
        this.cameraGes.position.set(0, 0, 3);
        
        this.rendererGes = new THREE.WebGLRenderer({ antialias: true });
        this.rendererGes.setSize(container.clientWidth, container.clientHeight);
        this.rendererGes.setPixelRatio(window.devicePixelRatio);
        container.appendChild(this.rendererGes.domElement);
        
        this.controlsGes = new OrbitControls(this.cameraGes, this.rendererGes.domElement);
        this.controlsGes.enableDamping = true;
        this.controlsGes.dampingFactor = 0.05;
        
        this.buildScene(this.sceneGes, true);
    }

    /**
     * Build the 3D representations of Gaussians and Surfels.
     */
    private buildScene(scene: THREE.Scene, isGES: boolean) {
        // Billboard vertex shader (keeps Gaussians facing the camera)
        const gaussVS = `
            varying vec2 vUv;
            void main() {
                vUv = uv - 0.5;
                vec4 mvPos = viewMatrix * modelMatrix * vec4(0.0, 0.0, 0.0, 1.0);
                mvPos.xy += position.xy;
                gl_Position = projectionMatrix * mvPos;
            }
        `;

        // Fixed plane vertex shader (represents physical surfels, which are oriented)
        const surfelVS = `
            varying vec2 vUv;
            void main() {
                vUv = uv - 0.5;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `;

        // Standard Gaussian fragment shader (soft alpha disk representation)
        const gaussFS = `
            varying vec2 vUv;
            uniform vec3 uColor;
            uniform float uOpacity;
            void main() {
                float d2 = dot(vUv, vUv);
                float g = exp(-18.0 * d2);
                if (g < 0.005) discard;
                gl_FragColor = vec4(uColor, uOpacity * g);
            }
        `;

        // Surfel fragment shader (solid disc representation)
        const surfelFS = `
            varying vec2 vUv;
            uniform vec3 uColor;
            uniform float uOpacity;
            void main() {
                float d2 = dot(vUv, vUv);
                if (d2 > 0.25) discard;
                float edge = smoothstep(0.25, 0.22, d2);
                gl_FragColor = vec4(uColor * (0.85 + 0.15 * edge), uOpacity * edge);
            }
        `;

        // GES Gaussian fragment shader: outputs PREMULTIPLIED color for additive blending
        const gesGaussFS = `
            varying vec2 vUv;
            uniform vec3 uColor;
            uniform float uOpacity;
            void main() {
                float d2 = dot(vUv, vUv);
                float g = exp(-18.0 * d2);
                if (g < 0.005) discard;
                float alpha = uOpacity * g;
                gl_FragColor = vec4(uColor * alpha, alpha);
            }
        `;

        SPLATS.forEach(s => {
            const isSurfel = isGES && s.isSurfel;
            const mat = new THREE.ShaderMaterial({
                vertexShader: isSurfel ? surfelVS : gaussVS,
                fragmentShader: isSurfel ? surfelFS : (isGES ? gesGaussFS : gaussFS),
                uniforms: { 
                    uColor: { value: new THREE.Color(s.rgb[0], s.rgb[1], s.rgb[2]) }, 
                    uOpacity: { value: s.opacity } 
                },
                transparent: true,
                depthWrite: !!isSurfel,
                depthTest: true,
                side: THREE.DoubleSide,
                ...(!isSurfel && isGES ? {
                    blending: THREE.CustomBlending,
                    blendSrc: THREE.OneFactor,
                    blendDst: THREE.OneFactor,
                    blendSrcAlpha: THREE.OneFactor,
                    blendDstAlpha: THREE.OneFactor,
                } : {})
            });

            const size = isSurfel ? 1.5 : 1.3;
            const mesh = new THREE.Mesh(new THREE.PlaneGeometry(size, size), mat);
            mesh.position.set(s.meshPos[0], s.meshPos[1], s.meshPos[2]);

            if (isSurfel) {
                // Rotate the surfel 90° about Y so its plane normal points along +X.
                // The Gaussians are spread along the X axis, so the surfel now stands as a
                // YZ-plane divider with the other splats on either side of it — clarifying the
                // two-pass depth test (Gaussians behind the opaque surfel are culled).
                mesh.rotation.y = Math.PI / 2;
            }

            scene.add(mesh);

            if (isGES) {
                mesh.renderOrder = isSurfel ? 1 : 2;
                this.meshesGes.push({ splat: s, mesh });
            } else {
                this.meshes3dgs.push({ splat: s, mesh });
            }
        });
    }

    /**
     * Synchronize controls by tracking user clicks/drags.
     */
    private bindControls() {
        const el3dgs = this.renderer3dgs.domElement;
        const elGes = this.rendererGes.domElement;
        el3dgs.addEventListener("pointerdown", () => { this.activeViewport = "3dgs"; });
        elGes.addEventListener("pointerdown", () => { this.activeViewport = "ges"; });
        window.addEventListener("pointerup", () => { this.activeViewport = null; });
    }

    /**
     * Resizing handler.
     */
    public handleResize() {
        const c3d = document.getElementById("viewport-3dgs")!;
        const cGes = document.getElementById("viewport-ges")!;
        if (c3d && cGes && c3d.clientWidth > 0) {
            this.camera3dgs.aspect = c3d.clientWidth / c3d.clientHeight;
            this.camera3dgs.updateProjectionMatrix();
            this.renderer3dgs.setSize(c3d.clientWidth, c3d.clientHeight);
            this.cameraGes.aspect = cGes.clientWidth / cGes.clientHeight;
            this.cameraGes.updateProjectionMatrix();
            this.rendererGes.setSize(cGes.clientWidth, cGes.clientHeight);
        }
    }

    // Reusable temporary for view-space depth computation (avoids per-frame allocation).
    private _tmpVec = new THREE.Vector3();

    /**
     * View-space depth of a mesh's center: z in camera space, returned as a positive
     * distance in front of the camera (larger = further). This is the key three.js uses
     * to order transparent objects, and matches the paper's depth (view z) for depth tests.
     */
    private viewDepth(camera: THREE.PerspectiveCamera, mesh: THREE.Mesh): number {
        camera.updateMatrixWorld();
        mesh.getWorldPosition(this._tmpVec);
        this._tmpVec.applyMatrix4(camera.matrixWorldInverse);
        return -this._tmpVec.z;
    }

    /**
     * Rendering loop.
     */
    private animate() {
        requestAnimationFrame(() => this.animate());
        this.controls3dgs.update();
        this.controlsGes.update();

        // Sync cameras: copy state from active to inactive viewport
        if (this.activeViewport === "3dgs") {
            this.cameraGes.position.copy(this.camera3dgs.position);
            this.cameraGes.quaternion.copy(this.camera3dgs.quaternion);
            this.controlsGes.target.copy(this.controls3dgs.target);
        } else if (this.activeViewport === "ges") {
            this.camera3dgs.position.copy(this.cameraGes.position);
            this.camera3dgs.quaternion.copy(this.cameraGes.quaternion);
            this.controls3dgs.target.copy(this.controlsGes.target);
        }

        // 3DGS side: Render sorting order debug output.
        // Use VIEW-SPACE depth (z in camera space), which is exactly what three.js uses to
        // sort transparent meshes — Euclidean distance can disagree at oblique angles.
        if (this.meshes3dgs.length > 0) {
            const dists = this.meshes3dgs.map(m => ({
                name: m.splat.name,
                d: this.viewDepth(this.camera3dgs, m.mesh)
            }));
            dists.sort((a, b) => a.d - b.d);
            const el = document.getElementById("debug-3dgs");
            if (el) {
                el.innerHTML = dists.map(x => `${x.name}: ${x.d.toFixed(2)}`).join("<br>") +
                               `<br>Order: ${dists.map(x => x.name).join(" → ")}`;
            }
        }

        // GES side: Render depth testing culling check (cull if d_i > d_s + delta).
        // View-space depth matches the paper's use of depth (view z) for the depth test.
        if (this.meshesGes.length > 0) {
            const surfelMesh = this.meshesGes.find(m => m.splat.isSurfel)?.mesh;
            if (surfelMesh) {
                const dS = this.viewDepth(this.cameraGes, surfelMesh);
                const deltaEl = document.getElementById("compDeltaSlider") as HTMLInputElement;
                const delta = deltaEl ? parseFloat(deltaEl.value) : 0.5;

                const debugLines: string[] = [`Surfel: ${dS.toFixed(2)}`];

                this.meshesGes.forEach(m => {
                    if (!m.splat.isSurfel) {
                        const dG = this.viewDepth(this.cameraGes, m.mesh);
                        const cull = dG > dS + delta;
                        m.mesh.visible = !cull;
                        debugLines.push(`${m.splat.name}: ${dG.toFixed(2)} ${cull ? "❌" : "✅"}`);
                    }
                });

                const el = document.getElementById("debug-ges");
                if (el) {
                    el.innerHTML = debugLines.join("<br>");
                }
            }
        }

        this.renderer3dgs.render(this.scene3dgs, this.camera3dgs);
        this.rendererGes.render(this.sceneGes, this.cameraGes);
    }
}
