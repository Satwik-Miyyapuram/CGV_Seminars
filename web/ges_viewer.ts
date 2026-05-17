import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
// import { SplatMesh } from "../external_code/spark/src/SplatMesh";
import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";

async function init() {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 5);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);

    const renderTarget = new THREE.WebGLRenderTarget(window.innerWidth, window.innerHeight, {
        type: THREE.HalfFloatType,
        format: THREE.RGBAFormat,
        depthBuffer: true,
    });

    let surfelViewer = null;
    let gaussianViewer = null;

    // Load Surfels
    document.getElementById('surfelInput').addEventListener('change', async (e) => {   
        const file = (e.target as HTMLInputElement).files[0];
        if (file) {
            const url = URL.createObjectURL(file);

            // Create the DropInViewer
            surfelViewer = new GaussianSplats3D.DropInViewer({
                'gpuAcceleratedSort': false
            });

            // Load the PLY
            await surfelViewer.addSplatScenes([{
                'path': url,
                'format': GaussianSplats3D.SceneFormat.Ply
            }]);

            // Extract the hidden material and apply the GES Z-Buffer math
            const material = surfelViewer.viewer.splatMesh.material;
            material.depthWrite = true;
            material.depthTest = true;
            // material.alphaTest = 0.5;
            material.polygonOffset = true;
            material.polygonOffsetUnits = parseFloat((document.getElementById('deltaSlider') as HTMLInputElement).value);        

            surfelViewer.renderOrder = 0;
            scene.add(surfelViewer);
            console.log("Surfels loaded!");
        }
    });

    // Load Gaussians
    document.getElementById('gaussianInput').addEventListener('change', async (e) => { 
        const file = (e.target as HTMLInputElement).files[0];
        if (file) {
            const url = URL.createObjectURL(file);

            gaussianViewer = new GaussianSplats3D.DropInViewer({
                // We turn sorting OFF because Additive Blending handles it automatically!
                'gpuAcceleratedSort': false
            });

            await gaussianViewer.addSplatScenes([{
                'path': url,
                'format': GaussianSplats3D.SceneFormat.Ply
            }]);

            // Extract the hidden material and apply the Additive Blending
            const material = gaussianViewer.viewer.splatMesh.material;
            material.depthTest = true;
            material.depthWrite = false;
            material.transparent = true;
            material.blending = THREE.CustomBlending;
            material.blendEquation = THREE.AddEquation;
            material.blendSrc = THREE.SrcAlphaFactor;
            material.blendDst = THREE.OneFactor;
            material.blendSrcAlpha = THREE.OneFactor;
            material.blendDstAlpha = THREE.OneFactor;

            gaussianViewer.renderOrder = 1;
            scene.add(gaussianViewer);
            console.log("Gaussians loaded!");
        }
    });

    const compositeMaterial = new THREE.ShaderMaterial({
        uniforms: { tDiffuse: {value: renderTarget.texture} },
        vertexShader: `
            varying vec2 vUv;
            void main() {
                vUv = uv;
                gl_Position = vec4(position, 1.0);
            }
        `,
        fragmentShader: `
            uniform sampler2D tDiffuse;
            varying vec2 vUv;
            void main() {
                vec4 texel = texture2D(tDiffuse, vUv);
                // (C_S+C_G)/(W_S+W_G+epsilon)
                vec3 foregroundColor = texel.rgb /max(texel.a, 0.0001);
                float finalAlpha = clamp(texel.a, 0.0, 1.0);
                vec3 backgroundColor = vec3(0.0); // Assuming black background
                vec3 finalColor = mix(backgroundColor, foregroundColor, finalAlpha);
                finalColor = pow(foregroundColor, vec3(1.0/2.2)); // Gamma correction
                gl_FragColor = vec4(finalColor, 1.0);
            }
        `,
        depthWrite: false,
        depthTest: false,
    })

    const compositeScene = new THREE.Scene();
    const compositeCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const fullScreenQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), compositeMaterial);
    compositeScene.add(fullScreenQuad);

    document.getElementById('deltaSlider').addEventListener('input', (event) => {
        const value = parseFloat((event.target as HTMLInputElement).value);
        document.getElementById('deltaValue').innerText = value.toFixed(1);
        if (surfelViewer && surfelViewer.viewer && surfelViewer.viewer.splatMesh && surfelViewer.viewer.splatMesh.material) {
            surfelViewer.viewer.splatMesh.material.polygonOffsetUnits = value;
        }
    });
    
    // Module A: Isolate Surfels vs Gaussians
    document.getElementById('toggleSurfels').addEventListener('change', (e) => {       
        if (surfelViewer) surfelViewer.visible = (e.target as HTMLInputElement).checked;
    });

    document.getElementById('toggleGaussians').addEventListener('change', (e) => {     
        if (gaussianViewer) gaussianViewer.visible = (e.target as HTMLInputElement).checked;
    });

    // The Ultimate Proof: Show what happens when depth-culling is removed!
    document.getElementById('toggleDepthTest').addEventListener('change', (e) => {     
        if (gaussianViewer && gaussianViewer.viewer.splatMesh) {
            // Turning this off instantly causes Gaussians to leak through all walls   
            gaussianViewer.viewer.splatMesh.material.depthTest = (e.target as HTMLInputElement).checked;
        }
    });

    
    function animate() {
        requestAnimationFrame(animate);
        controls.update();

        // render pass 1,2
        renderer.setRenderTarget(renderTarget)
        // renderer.setRenderTarget(null)
        renderer.clear();
        renderer.render(scene, camera);

        // to screen
        renderer.setRenderTarget(null);
        renderer.render(compositeScene, compositeCamera);
    }
    animate();
    window.addEventListener("resize", () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderTarget.setSize(window.innerWidth, window.innerHeight);
    });
}

init();