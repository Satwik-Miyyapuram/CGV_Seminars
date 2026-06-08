import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";

async function init() {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 5);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x000000, 0.0); // CRITICAL: Clear alpha MUST be 0 for additive alpha accumulation!
    
    const container = document.getElementById('viewer');
    if (container) {
        container.appendChild(renderer.domElement);
    } else {
        document.body.appendChild(renderer.domElement);
    }
    
    const controls = new OrbitControls(camera, renderer.domElement);

    const renderTarget = new THREE.WebGLRenderTarget(window.innerWidth, window.innerHeight, {
        type: THREE.HalfFloatType,
        format: THREE.RGBAFormat,
        depthBuffer: true,
    });

    let surfelViewer = null;
    let gaussianViewer = null;

    // Helper to update UI labels
    function updateFileLabel(labelId: string, statusId: string, name: string) {
        const labelEl = document.getElementById(labelId);
        const statusEl = document.getElementById(statusId);
        if (labelEl && statusEl) {
            labelEl.textContent = name;
            labelEl.classList.add('loaded');
            statusEl.textContent = "Loaded";
            statusEl.classList.add('active');
        }
    }

    // SURFEL LOADING FUNCTION
    async function loadSurfels(url: string) {
        if (surfelViewer) {
            scene.remove(surfelViewer);
            surfelViewer = null;
        }

        console.log("Loading surfels from:", url);

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
        material.alphaTest = 0.001; // Do NOT cut off translucent surfels at 0.5! We cull precisely in fragment shader.
        material.transparent = true; // Use transparent blending to handle background splats and boundary smoothing beautifully
        material.blending = THREE.CustomBlending;
        material.blendEquation = THREE.AddEquation;
        material.blendSrc = THREE.OneFactor; // Our fragment shader outputs premultiplied color: color.rgb * opacity
        material.blendDst = THREE.OneMinusSrcAlphaFactor;
        material.blendSrcAlpha = THREE.OneFactor;
        material.blendDstAlpha = THREE.OneMinusSrcAlphaFactor;
        material.polygonOffset = true;
        material.polygonOffsetUnits = parseFloat((document.getElementById('deltaSlider') as HTMLInputElement).value);        

        // CRITICAL FIX: The library's SplatMesh vertex shader projects quads to Z=0.0 on the screen plane 
        // for GPU sorting efficiency. This destroys hardware depth-testing. We must declare "varying float ndcDepth" 
        // in BOTH vertex and fragment shaders, compute it in the vertex shader, and write the correct 3D depth to 
        // "gl_FragDepth" in the fragment shader.
        // Additionally, we replace standard 3DGS Gaussian soft alpha with 2DGS/GES circular disc opacity (Eq. 6-7).
        material.onBeforeCompile = (shader) => {
            // Set up uniforms for interactive control
            shader.uniforms.uUseBiScale = { value: (document.getElementById('toggleBiScale') as HTMLInputElement).checked ? 1.0 : 0.0 };
            shader.uniforms.uOpacityCap = { value: parseFloat((document.getElementById('opacityCapSlider') as HTMLInputElement).value) };
            shader.uniforms.uDiscardThreshold = { value: parseFloat((document.getElementById('discardThresholdSlider') as HTMLInputElement).value) };
            
            // Expose uniforms on material userData so they can be changed dynamically in JS
            material.userData.uUseBiScale = shader.uniforms.uUseBiScale;
            material.userData.uOpacityCap = shader.uniforms.uOpacityCap;
            material.userData.uDiscardThreshold = shader.uniforms.uDiscardThreshold;

            // 1. Declare ndcDepth varying in the vertex shader
            shader.vertexShader = shader.vertexShader.replace(
                /void\s+main\s*\(\s*\)\s*\{/g,
                `varying float ndcDepth;
                void main() {`
            );
            
            // 2. Assign ndcDepth at the very end of the vertex shader's main()
            const lastBraceIndex = shader.vertexShader.lastIndexOf('}');
            if (lastBraceIndex !== -1) {
                shader.vertexShader = 
                    shader.vertexShader.substring(0, lastBraceIndex) + 
                    `    ndcDepth = gl_Position.z / gl_Position.w;
                    }` + 
                    shader.vertexShader.substring(lastBraceIndex + 1);
            }

            // 3. Declare varying and uniforms in the fragment shader
            shader.fragmentShader = shader.fragmentShader.replace(
                /varying\s+vec2\s+vPosition\s*;/g,
                `varying vec2 vPosition;
                varying float ndcDepth;
                uniform float uUseBiScale;
                uniform float uOpacityCap;
                uniform float uDiscardThreshold;`
            );
            
            // 4. Apply precise bi-scale disc opacity vs standard 2DGS Gaussian opacity based on uUseBiScale
            shader.fragmentShader = shader.fragmentShader.replace(
                /float\s+opacity\s*=\s*exp\(\s*-0\.5\s*\*\s*A\s*\)\s*\*\s*vColor\.a\s*;/g,
                `float g = exp(-0.5 * A);
                float opacity;
                if (uUseBiScale > 0.5) {
                    // Paper Eq. 6-7: α_i = min(τ_i, w_i · G(x))
                    // vColor.a = sigmoid(trained opacity logit) ∈ [0,1] (loaded correctly now)
                    // uOpacityCap controls the τ parameter interactively
                    opacity = min(uOpacityCap, vColor.a * g);
                    if (opacity < uDiscardThreshold || g < 1.0 / 255.0) discard;
                } else {
                    // Standard 3DGS/2DGS Gaussian formula: opacity = o_i * G(x)
                    opacity = vColor.a * g;
                    if (opacity < uDiscardThreshold || g < 1.0 / 255.0) discard;
                }`
            );
            // 5. CRITICAL MATHEMATICAL FIX: Write PREMULTIPLIED alpha to the framebuffer!
            // Since NoBlending is disabled, the GPU does not multiply color by alpha automatically.
            // The fragment shader itself must output "color.rgb * opacity" to ensure the accumulated
            // framebuffer contains correct premultiplied colors (C_S * W_S) for the final compositing step!
            shader.fragmentShader = shader.fragmentShader.replace(
                /gl_FragColor\s*=\s*vec4\s*\(\s*color\.rgb\s*,\s*opacity\s*\)\s*;/g,
                `gl_FragColor = vec4(color.rgb * opacity, opacity);
                gl_FragDepth = (ndcDepth + 1.0) / 2.0;`
            );
        };
        material.needsUpdate = true;

        surfelViewer.renderOrder = 0;
        // Respect current visibility checkbox
        surfelViewer.visible = (document.getElementById('toggleSurfels') as HTMLInputElement).checked;
        scene.add(surfelViewer);
        console.log("Surfels loaded and shaders compiled!");
    }

    // GAUSSIAN LOADING FUNCTION
    async function loadGaussians(url: string) {
        if (gaussianViewer) {
            scene.remove(gaussianViewer);
            gaussianViewer = null;
        }

        console.log("Loading gaussians from:", url);

        gaussianViewer = new GaussianSplats3D.DropInViewer({
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
        material.blendSrc = THREE.SrcAlphaFactor; // Shader outputs unpremultiplied color, GPU premultiplies by alpha dynamically
        material.blendDst = THREE.OneFactor;
        material.blendSrcAlpha = THREE.OneFactor;
        material.blendDstAlpha = THREE.OneFactor;

        // CRITICAL FIX: The Gaussians must also write their correct 3D depth to gl_FragDepth 
        // so they are properly depth-tested against the Z-buffer depth map written by the surfels!
        material.onBeforeCompile = (shader) => {
            // 1. Declare ndcDepth varying in the vertex shader
            shader.vertexShader = shader.vertexShader.replace(
                /void\s+main\s*\(\s*\)\s*\{/g,
                `varying float ndcDepth;
                void main() {`
            );
            
            // 2. Assign ndcDepth at the very end of the vertex shader's main()
            const lastBraceIndex = shader.vertexShader.lastIndexOf('}');
            if (lastBraceIndex !== -1) {
                shader.vertexShader = 
                    shader.vertexShader.substring(0, lastBraceIndex) + 
                    `    ndcDepth = gl_Position.z / gl_Position.w;
                    }` + 
                    shader.vertexShader.substring(lastBraceIndex + 1);
            }

            // 3. Declare ndcDepth varying in the fragment shader
            shader.fragmentShader = shader.fragmentShader.replace(
                /varying\s+vec2\s+vPosition\s*;/g,
                `varying vec2 vPosition;
                varying float ndcDepth;`
            );
            
            // 4. Write actual 3D depth to gl_FragDepth for correct depth testing
            shader.fragmentShader = shader.fragmentShader.replace(
                /gl_FragColor\s*=\s*vec4\s*\(\s*color\.rgb\s*,\s*opacity\s*\)\s*;/g,
                `gl_FragColor = vec4(color.rgb, opacity);
                gl_FragDepth = (ndcDepth + 1.0) / 2.0;`
            );
        };
        material.needsUpdate = true;

        gaussianViewer.renderOrder = 1;
        
        // Respect current checkboxes
        gaussianViewer.visible = (document.getElementById('toggleGaussians') as HTMLInputElement).checked;
        if (gaussianViewer.viewer && gaussianViewer.viewer.splatMesh) {
            gaussianViewer.viewer.splatMesh.material.depthTest = (document.getElementById('toggleDepthTest') as HTMLInputElement).checked;
        }

        scene.add(gaussianViewer);
        console.log("Gaussians loaded and shaders compiled!");
    }

    // Bind surfel file input
    document.getElementById('surfelInput').addEventListener('change', async (e) => {   
        const file = (e.target as HTMLInputElement).files?.[0];
        if (file) {
            const url = URL.createObjectURL(file);
            try {
                await loadSurfels(url);
                updateFileLabel('surfelLabel', 'surfelLoaded', file.name);
            } catch (err) {
                console.error("Failed to load surfels:", err);
                alert("Error loading surfels. Make sure it is a valid 2DGS/surfel PLY file.");
            }
        }
    });

    // Bind gaussian file input
    document.getElementById('gaussianInput').addEventListener('change', async (e) => { 
        const file = (e.target as HTMLInputElement).files?.[0];
        if (file) {
            const url = URL.createObjectURL(file);
            try {
                await loadGaussians(url);
                updateFileLabel('gaussianLabel', 'gaussianLoaded', file.name);
            } catch (err) {
                console.error("Failed to load gaussians:", err);
                alert("Error loading Gaussians. Make sure it is a valid 3DGS PLY file.");
            }
        }
    });

    // Bind config file input
    const configInput = document.getElementById('configInput') as HTMLInputElement;
    if (configInput) {
        configInput.addEventListener('change', async (e) => {
            const file = (e.target as HTMLInputElement).files?.[0];
            if (file) {
                const text = await file.text();
                try {
                    const config = JSON.parse(text);
                    if (config.background_color) {
                        const bg = config.background_color;
                        compositeMaterial.uniforms.uBackgroundColor.value.setRGB(bg[0], bg[1], bg[2]);
                        console.log("Loaded background color from config:", bg);
                        updateFileLabel('configLabel', 'configLoaded', file.name);
                    }
                } catch (err) {
                    console.error("Failed to parse config.json", err);
                    alert("Invalid config.json format!");
                }
            }
        });
    }

    // COMPOSITING & TWO-PASS RESOLUTION
    const compositeMaterial = new THREE.ShaderMaterial({
        uniforms: { 
            tDiffuse: {value: renderTarget.texture},
            uBackgroundColor: {value: new THREE.Color(1.0, 1.0, 1.0)} // Default to white
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
                
                // (C_S+C_G)/(W_S+W_G+epsilon)
                vec3 foregroundColor = texel.rgb / max(texel.a, 0.0001);
                
                // CRITICAL FIX: Clamp foreground color!
                // Additive blending can cause accumulated colors to exceed 1.0 if the 
                // raw SH coefficients mapped to colors > 1.0. Clamping prevents glowing spikes.
                foregroundColor = clamp(foregroundColor, 0.0, 1.0);
                
                float finalAlpha = clamp(texel.a, 0.0, 1.0);
                
                // Correctly blend the un-premultiplied color with the background color
                vec3 finalColor = mix(uBackgroundColor, foregroundColor, finalAlpha);
                
                gl_FragColor = vec4(finalColor, 1.0);
            }
        `,
        depthWrite: false,
        depthTest: false,
    });

    const compositeScene = new THREE.Scene();
    const compositeCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const fullScreenQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), compositeMaterial);
    compositeScene.add(fullScreenQuad);

    // PARAMETER CONTROLS BINDING
    document.getElementById('deltaSlider').addEventListener('input', (event) => {
        const value = parseFloat((event.target as HTMLInputElement).value);
        const valueEl = document.getElementById('deltaValue');
        if (valueEl) valueEl.innerText = value.toFixed(1);
        
        if (surfelViewer && surfelViewer.viewer?.splatMesh?.material) {
            surfelViewer.viewer.splatMesh.material.polygonOffsetUnits = value;
        }
    });
    document.getElementById('opacityCapSlider').addEventListener('input', (event) => {
        const value = parseFloat((event.target as HTMLInputElement).value);
        const valueEl = document.getElementById('opacityCapValue');
        if (valueEl) valueEl.innerText = value.toFixed(2);
        if (surfelViewer && surfelViewer.viewer?.splatMesh?.material?.userData?.uOpacityCap) {
            surfelViewer.viewer.splatMesh.material.userData.uOpacityCap.value = value;
        }
    });

    document.getElementById('discardThresholdSlider').addEventListener('input', (event) => {
        const value = parseFloat((event.target as HTMLInputElement).value);
        const valueEl = document.getElementById('discardThresholdValue');
        if (valueEl) valueEl.innerText = value.toFixed(3);

        if (surfelViewer && surfelViewer.viewer?.splatMesh?.material?.userData?.uDiscardThreshold) {
            surfelViewer.viewer.splatMesh.material.userData.uDiscardThreshold.value = value;
        }
    });

    document.getElementById('toggleBiScale').addEventListener('change', (event) => {
        const checked = (event.target as HTMLInputElement).checked;
        if (surfelViewer && surfelViewer.viewer?.splatMesh?.material?.userData?.uUseBiScale) {
            
            surfelViewer.viewer.splatMesh.material.userData.uUseBiScale.value = checked ? 1.0 : 0.0;
            console.log("Toggled Bi-scale opacity rendering to:", checked);
        }
    });
    
    // VIEW OPTIONS BINDING
    document.getElementById('toggleSurfels').addEventListener('change', (e) => {       
        if (surfelViewer) surfelViewer.visible = (e.target as HTMLInputElement).checked;
    });

    document.getElementById('toggleGaussians').addEventListener('change', (e) => {     
        if (gaussianViewer) gaussianViewer.visible = (e.target as HTMLInputElement).checked;
    });

    document.getElementById('toggleDepthTest').addEventListener('change', (e) => {     
        const checked = (e.target as HTMLInputElement).checked;
        if (gaussianViewer && gaussianViewer.viewer?.splatMesh?.material) {
            gaussianViewer.viewer.splatMesh.material.depthTest = checked;
        }
    });

    // AUTO-LOAD DEFAULT ASSETS
    async function autoLoadDefaultAssets() {
        const autoStatus = document.getElementById('autoLoadStatus');
        if (autoStatus) {
            autoStatus.textContent = "Loading...";
            autoStatus.classList.add('active');
        }

        let loadedCount = 0;
        
        // 1. Fetch Config
        try {
            console.log("Checking for config.json...");
            const res = await fetch('/web_assets/config.json');
            if (res.ok) {
                const config = await res.json();
                if (config.background_color) {
                    const bg = config.background_color;
                    compositeMaterial.uniforms.uBackgroundColor.value.setRGB(bg[0], bg[1], bg[2]);
                    updateFileLabel('configLabel', 'configLoaded', 'config.json');
                    console.log("Auto-loaded background color:", bg);
                    loadedCount++;
                }
            }
        } catch (err) {
            console.log("config.json auto-load skipped:", err);
        }

        // 2. Fetch Surfels
        try {
            console.log("Checking for surfels.ply...");
            const res = await fetch('/web_assets/surfels.ply');
            if (res.ok) {
                await loadSurfels('/web_assets/surfels.ply');
                updateFileLabel('surfelLabel', 'surfelLoaded', 'surfels.ply');
                loadedCount++;
            }
        } catch (err) {
            console.log("surfels.ply auto-load skipped:", err);
        }

        // 3. Fetch Gaussians
        try {
            console.log("Checking for gaussians.ply...");
            const res = await fetch('/web_assets/gaussians.ply');
            if (res.ok) {
                await loadGaussians('/web_assets/gaussians.ply');
                updateFileLabel('gaussianLabel', 'gaussianLoaded', 'gaussians.ply');
                loadedCount++;
            }
        } catch (err) {
            console.log("gaussians.ply auto-load skipped:", err);
        }

        if (autoStatus) {
            if (loadedCount > 0) {
                autoStatus.textContent = "Success";
                autoStatus.style.background = "rgba(0, 255, 135, 0.2)";
                autoStatus.style.color = "#00ff87";
            } else {
                autoStatus.textContent = "No Assets Found";
            }
        }
    }

    // Run auto-loader
     // autoLoadDefaultAssets();

    // RENDER LOOP
    function animate() {
        requestAnimationFrame(animate);
        controls.update();

        // Pass 1 & 2: Render splats into Half Float Render Target
        renderer.setRenderTarget(renderTarget);
        renderer.clear();
        renderer.render(scene, camera);

        // Pass 3: Composite over background to full screen
        renderer.setRenderTarget(null);
        renderer.render(compositeScene, compositeCamera);
    }
    
    // Tab switching logic
    const btnViewer = document.getElementById('btn-viewer');
    const btnComparison = document.getElementById('btn-comparison');
    const uiPanel = document.getElementById('ui');
    const viewerEl = document.getElementById('viewer');
    const comparisonContainer = document.getElementById('comparison-container');

    if (btnViewer && btnComparison && uiPanel && viewerEl && comparisonContainer) {
        btnViewer.addEventListener('click', () => {
            btnViewer.classList.add('active');
            btnComparison.classList.remove('active');
            viewerEl.style.display = 'block';
            uiPanel.style.display = 'block';
            comparisonContainer.style.display = 'none';
        });

        btnComparison.addEventListener('click', () => {
            btnViewer.classList.remove('active');
            btnComparison.classList.add('active');
            viewerEl.style.display = 'none';
            uiPanel.style.display = 'none';
            comparisonContainer.style.display = 'grid';
            
            // Trigger layout resize for comparison 3D viewports
            window.dispatchEvent(new Event('resize'));
        });
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