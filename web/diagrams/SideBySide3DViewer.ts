import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
// ============================================================
// 3D Interactive Side-by-Side Viewer
// ============================================================
export class SideBySide3DViewer {
    scene3dgs!: THREE.Scene;
    camera3dgs!: THREE.PerspectiveCamera;
    renderer3dgs!: THREE.WebGLRenderer;
    controls3dgs!: OrbitControls;
    sceneGes!: THREE.Scene;
    cameraGes!: THREE.PerspectiveCamera;
    rendererGes!: THREE.WebGLRenderer;
    controlsGes!: OrbitControls;
    blueMesh3dgs!: THREE.Mesh;
    redMesh3dgs!: THREE.Mesh;
    blueMeshGes!: THREE.Mesh;
    redMeshGes!: THREE.Mesh;
    greenMeshGes!: THREE.Mesh;
    // Track which viewport the user is actively interacting with
    private activeViewport: "3dgs" | "ges" | null = null;
    constructor() {
        this.init3DGS();
        this.initGES();
        this.bindControls();
        this.animate();
        window.addEventListener("resize", () => this.handleResize());
    }
    init3DGS() {
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
    initGES() {
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
    buildScene(scene: THREE.Scene, isGES: boolean) {
        // Billboard vertex shader (always face camera)
        const gaussVS = `
            varying vec2 vUv;
            void main() {
                vUv = uv - 0.5;
                vec4 mvPos = viewMatrix * modelMatrix * vec4(0.0, 0.0, 0.0, 1.0);
                mvPos.xy += position.xy;
                gl_Position = projectionMatrix * mvPos;
            }
        `;
        // Fixed plane vertex shader (surfel)
        const surfelVS = `
            varying vec2 vUv;
            void main() {
                vUv = uv - 0.5;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `;
        // Gaussian fragment (fuzzy blob)
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
        // Surfel fragment (solid disk)
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
        const zSpace = 0.05;
        // --- GES Gaussian fragment shader: outputs PREMULTIPLIED color for additive blending ---
        // gl_FragColor = vec4(color * opacity * gaussian, opacity * gaussian)
        // This allows additive hardware blending: src=ONE, dst=ONE
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
        // Blue (surfel in GES, gaussian in 3DGS)
        const blueMat = new THREE.ShaderMaterial({
            vertexShader: isGES ? surfelVS : gaussVS,
            fragmentShader: isGES ? surfelFS : gaussFS,
            uniforms: { uColor: { value: new THREE.Color(0x00d2ff) }, uOpacity: { value: 0.95 } },
            transparent: true,
            depthWrite: isGES,
            depthTest: true,
            side: THREE.DoubleSide,
        });
        const blueMesh = new THREE.Mesh(new THREE.PlaneGeometry(1.5, 1.5), blueMat);
        blueMesh.position.set(-0.2, 0, zSpace);
        scene.add(blueMesh);
        // Red (always gaussian billboard)
        const redMat = new THREE.ShaderMaterial({
            vertexShader: gaussVS,
            fragmentShader: isGES ? gesGaussFS : gaussFS,  // GES: premultiplied for additive
            uniforms: { uColor: { value: new THREE.Color(0xff4a5a) }, uOpacity: { value: 0.9 } },
            transparent: true,
            depthWrite: false,
            depthTest: isGES,  // GES: depth test against surfel depth buffer (but no depth write)
            side: THREE.DoubleSide,
            // GES floaters use ADDITIVE blending: src=ONE, dst=ONE
            // This makes the result ORDER-INDEPENDENT (no popping!)
            ...(isGES ? {
                blending: THREE.CustomBlending,
                blendSrc: THREE.OneFactor,
                blendDst: THREE.OneFactor,
                blendSrcAlpha: THREE.OneFactor,
                blendDstAlpha: THREE.OneFactor,
            } : {}),
        });
        const redMesh = new THREE.Mesh(new THREE.PlaneGeometry(1.3, 1.3), redMat);
        redMesh.position.set(0.2, 0, -zSpace);
        scene.add(redMesh);
        // Green (always gaussian billboard)
        const greenMat = new THREE.ShaderMaterial({
            vertexShader: gaussVS,
            fragmentShader: isGES ? gesGaussFS : gaussFS,  // GES: premultiplied for additive
            uniforms: { uColor: { value: new THREE.Color(0x00ff87) }, uOpacity: { value: 0.85 } },
            transparent: true,
            depthWrite: false,
            depthTest: isGES,  // GES: depth test against surfel depth buffer (but no depth write)
            side: THREE.DoubleSide,
            ...(isGES ? {
                blending: THREE.CustomBlending,
                blendSrc: THREE.OneFactor,
                blendDst: THREE.OneFactor,
                blendSrcAlpha: THREE.OneFactor,
                blendDstAlpha: THREE.OneFactor,
            } : {}),
        });
        const greenMesh = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), greenMat);
        greenMesh.position.set(0.0, 0.2, 0.0);
        scene.add(greenMesh);
        if (isGES) {
            this.blueMeshGes = blueMesh;
            this.redMeshGes = redMesh;
            this.greenMeshGes = greenMesh;
            // Surfel renders first (writes depth), then floaters render additively
            blueMesh.renderOrder = 1;
            redMesh.renderOrder = 2;
            greenMesh.renderOrder = 2;
        } else {
            this.blueMesh3dgs = blueMesh;
            this.redMesh3dgs = redMesh;
        }
    }
    bindControls() {
        // Track which viewport the user is interacting with via pointer events
        const el3dgs = this.renderer3dgs.domElement;
        const elGes = this.rendererGes.domElement;
        el3dgs.addEventListener("pointerdown", () => { this.activeViewport = "3dgs"; });
        elGes.addEventListener("pointerdown", () => { this.activeViewport = "ges"; });
        window.addEventListener("pointerup", () => { this.activeViewport = null; });
    }
    handleResize() {
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
    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls3dgs.update();
        this.controlsGes.update();
        // Sync cameras: copy from active viewport to the other
        if (this.activeViewport === "3dgs") {
            this.cameraGes.position.copy(this.camera3dgs.position);
            this.cameraGes.quaternion.copy(this.camera3dgs.quaternion);
            this.controlsGes.target.copy(this.controls3dgs.target);
        } else if (this.activeViewport === "ges") {
            this.camera3dgs.position.copy(this.cameraGes.position);
            this.camera3dgs.quaternion.copy(this.cameraGes.quaternion);
            this.controls3dgs.target.copy(this.controlsGes.target);
        }
        // 3DGS debug overlay
        if (this.blueMesh3dgs && this.redMesh3dgs) {
            const dB = this.camera3dgs.position.distanceTo(this.blueMesh3dgs.position);
            const dR = this.camera3dgs.position.distanceTo(this.redMesh3dgs.position);
            const el = document.getElementById("debug-3dgs");
            if (el) {
                const order = dB > dR ? "Blue → Red" : "Red → Blue";
                el.innerHTML = `Blue: ${dB.toFixed(2)}<br>Red: ${dR.toFixed(2)}<br>Order: ${order}`;
            }
        }
        // GES: floater culling (d_i < D_S + δ)
        if (this.blueMeshGes && this.redMeshGes && this.greenMeshGes) {
            const dS = this.cameraGes.position.distanceTo(this.blueMeshGes.position);
            const dR = this.cameraGes.position.distanceTo(this.redMeshGes.position);
            const dG = this.cameraGes.position.distanceTo(this.greenMeshGes.position);
            const deltaEl = document.getElementById("compDeltaSlider") as HTMLInputElement;
            const delta = deltaEl ? parseFloat(deltaEl.value) : 0.5;
            const cullR = dR > dS + delta;
            const cullG = dG > dS + delta;
            this.redMeshGes.visible = !cullR;
            this.greenMeshGes.visible = !cullG;
            const el = document.getElementById("debug-ges");
            if (el) {
                el.innerHTML = `Surfel: ${dS.toFixed(2)}<br>Red: ${dR.toFixed(2)} ${cullR ? "❌" : "✅"}<br>Green: ${dG.toFixed(2)} ${cullG ? "❌" : "✅"}`;
            }
        }
        this.renderer3dgs.render(this.scene3dgs, this.camera3dgs);
        this.rendererGes.render(this.sceneGes, this.cameraGes);
    }
}
