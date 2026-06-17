import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

/**
 * SceneManager is responsible for setting up and managing the Three.js rendering pipeline,
 * including the perspective camera, OrbitControls, WebGLRenderer, offscreen half-float 
 * render targets, and the final orthographic compositing pass.
 */
export class SceneManager {
    private container: HTMLElement;
    public scene: THREE.Scene;
    public camera: THREE.PerspectiveCamera;
    public renderer: THREE.WebGLRenderer;
    public controls: OrbitControls;
    public renderTarget: THREE.WebGLRenderTarget;
    private clock: THREE.Clock;
    
    // References to viewers for multi-pass rendering
    public surfelViewer: any | null = null;
    public gaussianViewer: any | null = null;

    // Compositing objects (Two-pass composite)
    private compositeScene: THREE.Scene;
    private compositeCamera: THREE.OrthographicCamera;
    public compositeMaterial: THREE.ShaderMaterial;
    private fullScreenQuad: THREE.Mesh;

    private isAnimating: boolean = false;
    public surfelColorTexture: THREE.FramebufferTexture;

    constructor(containerId: string) {
        const el = document.getElementById(containerId);
        if (!el) {
            throw new Error(`Container with id "${containerId}" not found.`);
        }
        this.container = el;
        this.clock = new THREE.Clock();

        this.scene = new THREE.Scene();

        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 100);
        this.camera.position.set(0, 1.5, 3);

        // Setup Renderer with transparent clear color (alpha = 0)
        // This is crucial for alpha accumulation to work correctly.
        this.renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setClearColor(0x000000, 0); 
        this.renderer.autoClear = false;
        this.container.appendChild(this.renderer.domElement);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true; // smooth panning
        this.controls.dampingFactor = 0.05;
        this.controls.screenSpacePanning = false;

        // Render target for two-pass rendering
        this.renderTarget = new THREE.WebGLRenderTarget(window.innerWidth, window.innerHeight, {
            type: THREE.HalfFloatType,
            format: THREE.RGBAFormat,
            depthBuffer: true,
        });

        // Texture to store the Surfel pass color and alpha
        this.surfelColorTexture = new THREE.FramebufferTexture(window.innerWidth, window.innerHeight, THREE.RGBAFormat);
        this.surfelColorTexture.type = THREE.HalfFloatType;

        this.setupCompositing();
        this.setupResizeListener();
    }

    /**
     * Set up the orthographic compositing pass.
     * Takes the offscreen render target texture and resolves the normalized
     * accumulated splat colors over the background.
     */
    private setupCompositing() {
        this.compositeMaterial = new THREE.ShaderMaterial({
            uniforms: { 
                tSurfel: { value: this.surfelColorTexture },
                tGaussian: { value: this.renderTarget.texture },
                uBackgroundColor: { value: new THREE.Color(1.0, 1.0, 1.0) } // Default to white background
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform sampler2D tSurfel;
                uniform sampler2D tGaussian;
                uniform vec3 uBackgroundColor;
                varying vec2 vUv;
                void main() {
                    vec4 surfel = texture2D(tSurfel, vUv);
                    vec4 gaussian = texture2D(tGaussian, vUv);
                    
                    // Paper Eq 5: C = (C_S + C_G) / (W_S + W_G)
                    // We sum the premultiplied colors and the weights independently
                    vec3 combinedColor = (surfel.rgb + gaussian.rgb) / max(surfel.a + gaussian.a, 0.0001);
                    float totalAlpha = clamp(surfel.a + gaussian.a, 0.0, 1.0);
                    
                    // Blend foreground with background color based on total accumulated opacity
                    vec3 finalColor = mix(uBackgroundColor, combinedColor, totalAlpha);
                    
                    gl_FragColor = vec4(finalColor, 1.0);
                }
            `,
            depthWrite: false,
            depthTest: false
        });

        this.compositeScene = new THREE.Scene();
        this.compositeCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
        const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.compositeMaterial);
        this.compositeScene.add(quad);
    }

    /**
     * Update the canvas size when the window is resized.
     */
    private setupResizeListener() {
        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
            this.renderTarget.setSize(window.innerWidth, window.innerHeight);
            // Recreate FramebufferTexture to match new size
            this.surfelColorTexture.dispose();
            this.surfelColorTexture = new THREE.FramebufferTexture(window.innerWidth, window.innerHeight, THREE.RGBAFormat);
            this.surfelColorTexture.type = THREE.HalfFloatType;
            if (this.compositeMaterial) {
                this.compositeMaterial.uniforms.tSurfel.value = this.surfelColorTexture;
            }
        });
    }

    /**
     * Set the background color for the scene.
     */
    public setBackgroundColor(color: string) {
        if (this.compositeMaterial) {
            this.compositeMaterial.uniforms.uBackgroundColor.value.set(color);
        }
    }

    /**
     * The custom multi-pass render loop.
     */
    public render() {
        if (this.surfelViewer) this.surfelViewer.update(this.camera, this.renderer);
        if (this.gaussianViewer) this.gaussianViewer.update(this.camera, this.renderer);

        // --------------------------------------------------
        // PASS 1: Render Surfels to the offscreen target
        // --------------------------------------------------
        this.renderer.setRenderTarget(this.renderTarget);
        // Clear BOTH Color and Depth for the first pass
        this.renderer.clear(true, true, false);

        if (this.surfelViewer) {
            // Render surfels with standard alpha compositing and write depth to Z-buffer
            this.renderer.render(this.surfelViewer.scene, this.camera);
        }

        // Copy the C_S and W_S from the render target color buffer to our texture
        this.renderer.copyFramebufferToTexture(new THREE.Vector2(0, 0), this.surfelColorTexture);

        // --------------------------------------------------
        // PASS 2: Render Gaussians to the same target
        // --------------------------------------------------
        // Clear ONLY the color buffer, keeping the Surfel Depth buffer intact!
        this.renderer.clearColor();

        if (this.gaussianViewer) {
            // Render Gaussians with standard alpha compositing.
            // Because the Z-buffer contains the Surfel depths, the hardware depth test
            // will correctly cull the Gaussians.
            this.renderer.render(this.gaussianViewer.scene, this.camera);
        }

        // --------------------------------------------------
        // PASS 3: Final Composition
        // --------------------------------------------------
        // Resolve the stored C_S, W_S and current C_G, W_G to the screen
        this.renderer.setRenderTarget(null);
        this.renderer.render(this.compositeScene, this.compositeCamera);
    }

    /**
     * Frame the camera so a scene of the given center/radius fits the view, and set
     * near/far planes appropriately. Called after a scene loads so the user starts
     * outside the point cloud instead of buried inside it.
     */
    public frameScene(center: [number, number, number], radius: number) {
        const c = new THREE.Vector3(center[0], center[1], center[2]);
        const r = Math.max(radius, 0.01);
        const fov = (this.camera.fov * Math.PI) / 180;
        const dist = (r / Math.sin(fov / 2)) * 1.1;

        // Offset along a pleasant 3/4 viewing direction.
        const dir = new THREE.Vector3(0.4, 0.45, 1).normalize();
        this.camera.position.copy(c).add(dir.multiplyScalar(dist));
        this.camera.near = Math.max(dist * 0.01, 0.01);
        this.camera.far = dist * 4 + r * 6;
        this.camera.updateProjectionMatrix();

        this.controls.target.copy(c);
        this.controls.update();
    }

    /**
     * Start the rendering animation loop.
     */
    public start() {
        if (this.isAnimating) return;
        this.isAnimating = true;
        this.animate();
    }

    private animate = () => {
        if (!this.isAnimating) return;
        requestAnimationFrame(this.animate);
        
        this.controls.update();

        // Render scene into the Half Float render target
        this.renderer.setRenderTarget(this.renderTarget);
        this.renderer.clear();

        const surfel = this.surfelViewer;
        const gaussian = this.gaussianViewer;

        const surfelVisible = surfel ? surfel.visible : false;
        const gaussianVisible = gaussian ? gaussian.visible : false;

        // ==========================================
        // PASS 1: Surfel Culling (Depth Pass)
        // ==========================================
        // The paper uses the surfels as a solid surface. 
        // We write their depth to the Z-buffer first to cull occluded Gaussians/Surfels.
        if (surfel && surfelVisible) {
            const material = surfel.viewer?.splatMesh?.material;
            if (material) {
                if (gaussian) gaussian.visible = false;
                surfel.visible = true;

                material.depthWrite = true;
                material.colorWrite = false;
                this.renderer.render(this.scene, this.camera);
            }
        }

        // ==========================================
        // PASS 2: Additive Color Accumulation
        // ==========================================
        // We accumulate colors independently at the visible surface without sorting.
        // Both Surfels and Gaussians are rendered additively (depthTest = true, depthWrite = false).
        
        // Render Surfels Color
        if (surfel && surfelVisible) {
            const material = surfel.viewer?.splatMesh?.material;
            if (material) {
                if (gaussian) gaussian.visible = false;
                surfel.visible = true;

                material.depthWrite = false;
                material.colorWrite = true;
                // Important: ensure depth testing is LEQUAL so it passes against its own depth buffer!
                material.depthFunc = THREE.LessEqualDepth;
                this.renderer.render(this.scene, this.camera);
            }
        }

        // Render Gaussians Color
        if (gaussian && gaussianVisible) {
            const material = gaussian.viewer?.splatMesh?.material;
            if (material) {
                if (surfel) surfel.visible = false;
                gaussian.visible = true;
                
                material.depthWrite = false;
                material.colorWrite = true;
                material.depthFunc = THREE.LessEqualDepth;
                this.renderer.render(this.scene, this.camera);
            }
        }

        // Restore original visibilities
        if (surfel) surfel.visible = surfelVisible;
        if (gaussian) gaussian.visible = gaussianVisible;

        // Composite final screen image (normalize and blend with background).
        this.renderer.setRenderTarget(null);
        this.renderer.clear();
        this.renderer.render(this.compositeScene, this.compositeCamera);
    };

    /**
     * Stop the rendering animation loop.
     */
    public stop() {
        this.isAnimating = false;
    }
}
