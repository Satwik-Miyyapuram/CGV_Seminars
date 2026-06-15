import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

/**
 * Reflection3DViewer illustrates a limitation of the GES representation.
 * 3DGS simulates specular reflections by placing transparent floaters below a floor.
 * In contrast, GES's opaque surfel blocks underground floaters, meaning specular
 * reflections must be faked by placing highlights just above the surfel.
 */
export class Reflection3DViewer {
    private container: HTMLElement;
    private renderer: THREE.WebGLRenderer;
    private scene3DGS: THREE.Scene;
    private sceneGES: THREE.Scene;
    private camera: THREE.PerspectiveCamera;
    private controls: OrbitControls;

    constructor() {
        this.container = document.getElementById("reflection-container") as HTMLElement;
        if (!this.container) return;

        // Renderer configuration with scissor testing enabled for side-by-side splits
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setScissorTest(true);
        this.container.appendChild(this.renderer.domElement);

        // Split camera setup (Shared for synchronized movements)
        this.camera = new THREE.PerspectiveCamera(45, (this.container.clientWidth / 2) / this.container.clientHeight, 0.1, 100);
        this.camera.position.set(0, 1.5, 3.5);
        
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.target.set(0, 0, 0);

        this.scene3DGS = new THREE.Scene();
        this.sceneGES = new THREE.Scene();

        // Populate side-by-side scenes
        this.buildScene(this.scene3DGS, false);
        this.buildScene(this.sceneGES, true);

        // Bind and launch loop
        this.animate = this.animate.bind(this);
        this.animate();

        window.addEventListener('resize', this.handleResize.bind(this));
    }

    /**
     * Build the floor and splats for both 3DGS and GES views.
     */
    private buildScene(scene: THREE.Scene, isGES: boolean) {
        const splatGeo = new THREE.PlaneGeometry(0.3, 0.3);
        const floorGeo = new THREE.PlaneGeometry(2.5, 2.5);

        // Generate Gaussian Alpha Texture dynamically
        const canvas = document.createElement('canvas');
        canvas.width = 128;
        canvas.height = 128;
        const context = canvas.getContext('2d')!;
        const grad = context.createRadialGradient(64, 64, 0, 64, 64, 64);
        grad.addColorStop(0, 'rgba(255,255,255,1)');
        grad.addColorStop(1, 'rgba(255,255,255,0)');
        context.fillStyle = grad;
        context.fillRect(0, 0, 128, 128);
        const gaussianTex = new THREE.CanvasTexture(canvas);

        const createMat = (colorHex: number, opacity: number, depthWrite: boolean, blending: THREE.Blending, isFloor: boolean = false) => {
            return new THREE.MeshBasicMaterial({
                color: colorHex,
                side: THREE.DoubleSide,
                transparent: true,
                opacity: opacity,
                depthWrite: depthWrite,
                blending: blending,
                alphaMap: isFloor && isGES ? null : gaussianTex, // GES floor is a solid surfel, others are Gaussians
                alphaTest: 0.01
            });
        };

        // 1. Floor configuration
        // 3DGS floor: Alpha-blended semi-transparent Gaussian plane
        // GES floor: Solid opaque Surfel plane writing to depth buffer
        const floorMat = createMat(
            0x4a4a5a, 
            isGES ? 1.0 : 0.6,
            isGES,
            isGES ? THREE.AdditiveBlending : THREE.NormalBlending,
            true
        );
        const floor = new THREE.Mesh(floorGeo, floorMat);
        floor.rotation.x = -Math.PI / 2;
        floor.position.y = 0;
        scene.add(floor);

        // 2. Float Splat Object (floating above the floor)
        const mainMat = createMat(0xff9f43, 0.9, false, isGES ? THREE.AdditiveBlending : THREE.NormalBlending);
        const mainObject = new THREE.Mesh(splatGeo, mainMat);
        mainObject.position.set(0, 0.8, 0);
        mainObject.userData.isBillboard = true;
        scene.add(mainObject);

        // 3. Specular Reflection Representation
        // 3DGS reflection: Placed underground (y = -0.8), visible through transparent floor
        // GES reflection: Placed slightly above floor (y = 0.05) and scaled to represent specular reflection
        const reflectMat = createMat(0xff9f43, 0.6, false, isGES ? THREE.AdditiveBlending : THREE.NormalBlending);
        const mirroredObject = new THREE.Mesh(splatGeo, reflectMat);
        
        if (isGES) {
            mirroredObject.position.set(0, 0.05, 0);
            mirroredObject.rotation.x = -Math.PI / 2;
            mirroredObject.scale.set(3, 3, 1);
            mirroredObject.userData.isBillboard = false;
        } else {
            mirroredObject.position.set(0, -0.8, 0);
            mirroredObject.userData.isBillboard = true;
        }
        
        scene.add(mirroredObject);
    }

    /**
     * Resizing handler.
     */
    public handleResize() {
        if (!this.container) return;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        this.renderer.setSize(width, height);
        this.camera.aspect = (width / 2) / height;
        this.camera.updateProjectionMatrix();
    }

    /**
     * Billboard handler making splats face the camera.
     */
    private updateBillboards(scene: THREE.Scene) {
        scene.traverse((child) => {
            if (child instanceof THREE.Mesh && child.userData.isBillboard) {
                child.quaternion.copy(this.camera.quaternion);
            }
        });
    }

    private animate() {
        requestAnimationFrame(this.animate);
        this.controls.update();

        this.updateBillboards(this.scene3DGS);
        this.updateBillboards(this.sceneGES);

        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        const hw = width / 2;

        // Render Left Viewport: 3DGS (Alpha Blending)
        this.renderer.setScissor(0, 0, hw, height);
        this.renderer.setViewport(0, 0, hw, height);
        this.renderer.render(this.scene3DGS, this.camera);

        // Render Right Viewport: GES (Surfel depth testing)
        this.renderer.setScissor(hw, 0, hw, height);
        this.renderer.setViewport(hw, 0, hw, height);
        this.renderer.render(this.sceneGES, this.camera);
    }
}
