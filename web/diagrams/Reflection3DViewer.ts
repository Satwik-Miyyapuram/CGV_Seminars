import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

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

        // Renderer Setup
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setScissorTest(true);
        this.container.appendChild(this.renderer.domElement);

        // Camera & Controls Setup
        this.camera = new THREE.PerspectiveCamera(45, (this.container.clientWidth / 2) / this.container.clientHeight, 0.1, 100);
        this.camera.position.set(0, 1.5, 3.5);
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.target.set(0, 0, 0);

        // Scenes Setup
        this.scene3DGS = new THREE.Scene();
        this.sceneGES = new THREE.Scene();

        this.buildScene(this.scene3DGS, false);
        this.buildScene(this.sceneGES, true);

        // Animation Loop
        this.animate = this.animate.bind(this);
        this.animate();

        window.addEventListener('resize', this.handleResize.bind(this));
    }

    private buildScene(scene: THREE.Scene, isGES: boolean) {
        const splatGeo = new THREE.PlaneGeometry(0.3, 0.3);
        const floorGeo = new THREE.PlaneGeometry(2.5, 2.5);

        // Generate Gaussian Texture
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

        // Helper to create splat material
        const createMat = (colorHex: number, opacity: number, depthWrite: boolean, blending: THREE.Blending, isFloor: boolean = false) => {
            return new THREE.MeshBasicMaterial({
                color: colorHex,
                side: THREE.DoubleSide,
                transparent: true,
                opacity: opacity,
                depthWrite: depthWrite,
                blending: blending,
                alphaMap: isFloor && isGES ? null : gaussianTex, // GES floor is a solid surfel, everything else is a Gaussian
                alphaTest: 0.01 // Help WebGL sorting
            });
        };

        // 1. "Shiny Floor"
        // 3DGS: The floor is made of Gaussians (alpha blended, no depth write). We simulate this with a giant Gaussian.
        // GES: The floor is an opaque Surfel (depthWrite: true, culling objects behind it). Solid square.
        const floorMat = createMat(
            0x4a4a5a, 
            isGES ? 1.0 : 0.6, // GES surfel is opaque
            isGES,             // GES writes depth for the surfel
            isGES ? THREE.AdditiveBlending : THREE.NormalBlending,
            true               // Is Floor
        );
        const floor = new THREE.Mesh(floorGeo, floorMat);
        floor.rotation.x = -Math.PI / 2;
        floor.position.y = 0;
        scene.add(floor);

        // 2. Main Object (Above Floor)
        const mainMat = createMat(0xff9f43, 0.9, false, isGES ? THREE.AdditiveBlending : THREE.NormalBlending);
        const mainObject = new THREE.Mesh(splatGeo, mainMat);
        mainObject.position.set(0, 0.8, 0);
        mainObject.userData.isBillboard = true;
        scene.add(mainObject);

        // 3. Fake Reflection Object
        // 3DGS: Places the reflected Gaussian physically UNDERGROUND (e.g. y = -0.8). It shows through the semi-transparent floor.
        // GES: Since the surfel culls anything underground, GES must place Gaussians JUST ABOVE the surfel (e.g. y = 0.05) to model specular reflections.
        const reflectMat = createMat(0xff9f43, 0.6, false, isGES ? THREE.AdditiveBlending : THREE.NormalBlending);
        const mirroredObject = new THREE.Mesh(splatGeo, reflectMat);
        
        if (isGES) {
            // GES: Floater placed just above the surfel to fake reflection
            mirroredObject.position.set(0, 0.05, 0);
            
            // To make it look like a reflection on the floor, we'll keep it flat against the surfel
            // instead of billboarding it like a floating ball
            mirroredObject.rotation.x = -Math.PI / 2;
            mirroredObject.scale.set(3, 3, 1); // Widen it to look like a glossy highlight
            mirroredObject.userData.isBillboard = false;
        } else {
            // 3DGS: Reflected object placed underground
            mirroredObject.position.set(0, -0.8, 0);
            mirroredObject.userData.isBillboard = true;
        }
        
        scene.add(mirroredObject);
    }

    public handleResize() {
        if (!this.container) return;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        this.renderer.setSize(width, height);
        this.camera.aspect = (width / 2) / height;
        this.camera.updateProjectionMatrix();
    }

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

        // Render Left (3DGS)
        this.renderer.setScissor(0, 0, hw, height);
        this.renderer.setViewport(0, 0, hw, height);
        this.renderer.render(this.scene3DGS, this.camera);

        // Render Right (GES)
        this.renderer.setScissor(hw, 0, hw, height);
        this.renderer.setViewport(hw, 0, hw, height);
        this.renderer.render(this.sceneGES, this.camera);
    }
}
