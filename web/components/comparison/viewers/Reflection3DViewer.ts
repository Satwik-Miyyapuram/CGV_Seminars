import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

/**
 * Side-by-side 3D scene showing how each method fakes a specular reflection of a
 * fixed 3D-Gaussian light in a flat mirror. Only the camera moves.
 *
 *   3DGS (left):  translucent Gaussian mirror; reflection = the light's true mirror
 *                 image placed underground, seen through the mirror (view-stable).
 *   GES  (right): opaque surfel mirror blocks any underground floater, so the
 *                 reflection is a Gaussian fixed on top whose colour/opacity are
 *                 view-dependent (Fresnel warm→cool) — it can recolour but not move.
 */
export class Reflection3DViewer {
    private container: HTMLElement;
    private renderer: THREE.WebGLRenderer;
    private scene3DGS: THREE.Scene;
    private sceneGES: THREE.Scene;
    private camera: THREE.PerspectiveCamera;
    private controls: OrbitControls;
    private gesReflection?: THREE.Group;   // the faked GES highlight (view-dependent)

    private readonly lightPos = new THREE.Vector3(0.35, 0.7, 0.35);
    private readonly virtualImage = new THREE.Vector3(0.35, -0.7, 0.35); // light mirrored across y=0 (fixed)
    private readonly reflectSpot = new THREE.Vector3(0.35, 0.08, 0.35);  // GES highlight — fixed on the surfel
    private readonly warm = new THREE.Color(0xffcf73);  // grazing-angle colour
    private readonly cool = new THREE.Color(0xaadcff);  // head-on colour

    constructor() {
        this.container = document.getElementById("reflection-container") as HTMLElement;
        if (!this.container) return;

        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setScissorTest(true);
        this.container.appendChild(this.renderer.domElement);

        // Shared camera → both halves orbit together.
        this.camera = new THREE.PerspectiveCamera(45, (this.container.clientWidth / 2) / this.container.clientHeight, 0.1, 100);
        this.camera.position.set(0.1, 1.25, 2.7);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.target.set(0, 0.15, 0);
        this.controls.minPolarAngle = 0.2;
        this.controls.maxPolarAngle = Math.PI / 2 - 0.06;   // stay above the mirror

        this.scene3DGS = new THREE.Scene();
        this.sceneGES = new THREE.Scene();
        this.buildScene(this.scene3DGS, false);
        this.buildScene(this.sceneGES, true);

        this.animate = this.animate.bind(this);
        this.animate();
        window.addEventListener('resize', this.handleResize.bind(this));
    }

    /**
     * A 3D Gaussian: a bright opaque-ish core sphere wrapped in two additive
     * translucent shells, giving a soft volumetric falloff that reads as a blob.
     */
    private gaussianBlob(color: number, coreOpacity: number, radius: number): THREE.Group {
        const g = new THREE.Group();
        const core = new THREE.Mesh(
            new THREE.SphereGeometry(radius * 0.45, 20, 20),
            new THREE.MeshBasicMaterial({ color, transparent: true, opacity: coreOpacity, depthWrite: false })
        );
        const halo1 = new THREE.Mesh(
            new THREE.SphereGeometry(radius * 0.8, 20, 20),
            new THREE.MeshBasicMaterial({ color, transparent: true, opacity: coreOpacity * 0.35, depthWrite: false, blending: THREE.AdditiveBlending })
        );
        const halo2 = new THREE.Mesh(
            new THREE.SphereGeometry(radius, 20, 20),
            new THREE.MeshBasicMaterial({ color, transparent: true, opacity: coreOpacity * 0.15, depthWrite: false, blending: THREE.AdditiveBlending })
        );
        g.add(core, halo1, halo2);
        return g;
    }

    /** A flat disc lying in the XZ plane: the mirror surface (opaque surfel, or translucent Gaussian). */
    private flatDisc(color: number, opacity: number, radius: number, opaque: boolean): THREE.Mesh {
        const mat = new THREE.MeshBasicMaterial({
            color, side: THREE.DoubleSide,
            transparent: !opaque, opacity, depthWrite: opaque,
        });
        const m = new THREE.Mesh(new THREE.CircleGeometry(radius, 48), mat);
        m.rotation.x = -Math.PI / 2;
        return m;
    }

    private buildScene(scene: THREE.Scene, isGES: boolean) {
        const mirror = isGES
            ? this.flatDisc(0x223040, 1.0, 0.9, true)     // opaque surfel
            : this.flatDisc(0x9fbada, 0.22, 0.95, false); // translucent Gaussian
        scene.add(mirror);

        const light = this.gaussianBlob(0xfff0c0, 0.95, 0.22);
        light.position.copy(this.lightPos);
        scene.add(light);

        if (isGES) {
            // Flatten onto the surface and lift clear of the opaque surfel (additive shells) so it
            // reads as a glow sitting on top rather than a sphere sinking through.
            const refl = this.gaussianBlob(0xffcf73, 0.8, 0.34);
            refl.scale.set(1, 0.3, 1);
            refl.position.copy(this.reflectSpot);
            refl.children.forEach((c) => {
                ((c as THREE.Mesh).material as THREE.MeshBasicMaterial).blending = THREE.AdditiveBlending;
            });
            this.gesReflection = refl;
            scene.add(refl);
        } else {
            const refl = this.gaussianBlob(0xfff0c0, 0.6, 0.2);
            refl.position.copy(this.virtualImage);
            scene.add(refl);
        }
    }

    public handleResize() {
        if (!this.container) return;
        const w = this.container.clientWidth, h = this.container.clientHeight;
        this.renderer.setSize(w, h);
        this.camera.aspect = (w / 2) / h;
        this.camera.updateProjectionMatrix();
    }

    private animate() {
        requestAnimationFrame(this.animate);
        this.controls.update();

        // GES fake highlight. The highlight Gaussian is FIXED on the surfel (a 3D Gaussian can't
        // move its specular spot — only recolour it via view-dependent SH). So as the camera
        // orbits we change only its colour and opacity (Fresnel), never its position.
        if (this.gesReflection) {
            const viewDir = new THREE.Vector3().subVectors(this.camera.position, this.reflectSpot).normalize();
            const dot = Math.max(0, viewDir.dot(new THREE.Vector3(0, 1, 0))); // 1 = straight down
            const fresnel = 0.15 + 0.85 * Math.pow(1 - dot, 3);               // stronger at grazing angles
            const col = this.cool.clone().lerp(this.warm, fresnel);
            this.gesReflection.children.forEach((c, i) => {
                const mat = (c as THREE.Mesh).material as THREE.MeshBasicMaterial;
                mat.color.copy(col);
                mat.opacity = [0.8, 0.28, 0.12][i] * (0.3 + 0.7 * fresnel);
            });
        }

        const w = this.container.clientWidth, h = this.container.clientHeight, hw = w / 2;
        this.renderer.setScissor(0, 0, hw, h);
        this.renderer.setViewport(0, 0, hw, h);
        this.renderer.render(this.scene3DGS, this.camera);

        this.renderer.setScissor(hw, 0, hw, h);
        this.renderer.setViewport(hw, 0, hw, h);
        this.renderer.render(this.sceneGES, this.camera);
    }
}
