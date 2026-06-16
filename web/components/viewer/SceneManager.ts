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
    
    // References to viewers for multi-pass rendering
    public surfelViewer: any | null = null;
    public gaussianViewer: any | null = null;

    // Compositing objects (Two-pass composite)
    private compositeScene: THREE.Scene;
    private compositeCamera: THREE.OrthographicCamera;
    public compositeMaterial: THREE.ShaderMaterial;
    private fullScreenQuad: THREE.Mesh;

    private isAnimating: boolean = false;

    constructor(containerId: string) {
        const el = document.getElementById(containerId);
        if (!el) {
            throw new Error(`Container with id "${containerId}" not found.`);
        }
        this.container = el;

        this.scene = new THREE.Scene();
        
        // Setup Perspective Camera for 3D navigation
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.camera.position.set(0, 0, 5);

        // Setup Renderer with transparent clear color (alpha = 0)
        // This is crucial for additive alpha accumulation across surfels/gaussians.
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setClearColor(0x000000, 0.0);
        this.container.appendChild(this.renderer.domElement);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);

        // Render target for two-pass rendering:
        // Pass 1 & 2: Surfels and Gaussians are rendered here.
        // HalfFloatType matches high precision and color depth.
        this.renderTarget = new THREE.WebGLRenderTarget(window.innerWidth, window.innerHeight, {
            type: THREE.HalfFloatType,
            format: THREE.RGBAFormat,
            depthBuffer: true,
        });

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
                tDiffuse: { value: this.renderTarget.texture },
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
                uniform sampler2D tDiffuse;
                uniform vec3 uBackgroundColor;
                varying vec2 vUv;
                void main() {
                    vec4 texel = texture2D(tDiffuse, vUv);
                    
                    // Normalize accumulated premultiplied colors by accumulated weight (alpha)
                    // C = (C_S + C_G) / (W_S + W_G + epsilon)
                    vec3 foregroundColor = texel.rgb / max(texel.a, 0.0001);
                    
                    // Clamp to prevent glowing spikes (due to raw SH coefficients or additive accumulation)
                    foregroundColor = clamp(foregroundColor, 0.0, 1.0);
                    
                    float finalAlpha = clamp(texel.a, 0.0, 1.0);
                    
                    // Blend foreground with background color based on accumulated opacity
                    vec3 finalColor = mix(uBackgroundColor, foregroundColor, finalAlpha);
                    
                    gl_FragColor = vec4(finalColor, 1.0);
                }
            `,
            depthWrite: false,
            depthTest: false,
        });

        this.compositeScene = new THREE.Scene();
        this.compositeCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
        this.fullScreenQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.compositeMaterial);
        this.compositeScene.add(this.fullScreenQuad);
    }

    /**
     * Handle resizing the renderer and camera aspects.
     */
    public handleResize = () => {
        const width = window.innerWidth;
        const height = window.innerHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();

        this.renderer.setSize(width, height);
        this.renderTarget.setSize(width, height);
    };

    private setupResizeListener() {
        window.addEventListener("resize", this.handleResize);
    }

    /**
     * Set the background color used in the compositing pass.
     */
    public setBackgroundColor(color: THREE.Color) {
        this.compositeMaterial.uniforms.uBackgroundColor.value.copy(color);
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

        // Pass 1: Render Surfels (Color only)
        if (surfel && surfelVisible) {
            const material = surfel.viewer?.splatMesh?.material;
            if (material) {
                // Ensure Gaussians are temporarily hidden during surfel color pass
                if (gaussian) gaussian.visible = false;
                surfel.visible = true;

                // Disable depth writing during color pass so edges blend correctly
                material.depthWrite = false;
                material.colorWrite = true;
                this.renderer.render(this.scene, this.camera);

                // Pass 2: Render Surfels (Depth only)
                // Write depth values of the sorted surfels to the depth buffer for Gaussian culling
                if (gaussian && gaussianVisible) {
                    material.depthWrite = true;
                    material.colorWrite = false;
                    this.renderer.render(this.scene, this.camera);
                }

                // Reset material properties back to defaults
                material.depthWrite = false;
                material.colorWrite = true;
            } else {
                if (gaussian) gaussian.visible = false;
                this.renderer.render(this.scene, this.camera);
            }
        }

        // Pass 3: Render Gaussians (tested against Pass 2 depth buffer)
        if (gaussian && gaussianVisible) {
            if (surfel) surfel.visible = false;
            gaussian.visible = true;
            this.renderer.render(this.scene, this.camera);
        }

        // Restore original visibilities for loop/UI consistency
        if (surfel) surfel.visible = surfelVisible;
        if (gaussian) gaussian.visible = gaussianVisible;

        // Composite final screen image (normalize and blend with background)
        this.renderer.setRenderTarget(null);
        this.renderer.render(this.compositeScene, this.compositeCamera);
    };

    /**
     * Stop the rendering animation loop.
     */
    public stop() {
        this.isAnimating = false;
    }
}
