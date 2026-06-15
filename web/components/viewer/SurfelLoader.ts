import * as THREE from "three";
import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";

/**
 * SurfelLoader is responsible for loading the Surfels (2D Splats) PLY file,
 * adding it to the Three.js scene, and compiling the custom shaders that implement:
 * 1. GES Bi-scale Disc Opacity (Eq. 6-7 in the paper)
 * 2. Proper depth-buffer output (gl_FragDepth) for hardware depth-testing
 * 3. Premultiplied alpha output for correct subsequent blending
 */
export class SurfelLoader {
    private scene: THREE.Scene;
    public viewer: any | null = null;
    
    // Shader uniforms exposed for dynamic updates
    private uniforms: {
        uUseBiScale: { value: number };
        uOpacityCap: { value: number };
        uDiscardThreshold: { value: number };
    };

    constructor(scene: THREE.Scene) {
        this.scene = scene;
        this.uniforms = {
            uUseBiScale: { value: 1.0 },       // Enabled by default
            uOpacityCap: { value: 1.0 },       // Fully opaque cap by default
            uDiscardThreshold: { value: 0.002 } // Low opacity discard threshold
        };
    }

    /**
     * Load surfels from a URL (either local file object URL or path).
     */
    public async load(url: string, initialDelta: number = 0.5) {
        this.unload();

        console.log("[SurfelLoader] Loading surfels from:", url);

        // Instantiate GaussianSplats3D DropInViewer
        // Disable GPU accelerated sort because we need standard sorting or custom depth handling
        this.viewer = new GaussianSplats3D.DropInViewer({
            'gpuAcceleratedSort': false
        });

        // Add the PLY scene
        await this.viewer.addSplatScenes([{
            'path': url,
            'format': GaussianSplats3D.SceneFormat.Ply
        }]);

        this.configureMaterial(initialDelta);
        this.viewer.renderOrder = 0; // Surfels render first (Pass 1)
        this.scene.add(this.viewer);
        
        console.log("[SurfelLoader] Surfels loaded and configured.");
    }

    /**
     * Unload current surfels from the scene.
     */
    public unload() {
        if (this.viewer) {
            this.scene.remove(this.viewer);
            this.viewer = null;
        }
    }

    /**
     * Inject custom depth writing and bi-scale equations into the splat material shaders.
     */
    private configureMaterial(delta: number) {
        if (!this.viewer) return;

        const material = this.viewer.viewer.splatMesh.material;
        
        // Enable writing to the depth buffer for subsequent culling of Gaussians
        material.depthWrite = true;
        material.depthTest = true;
        
        // Low alpha test so we don't cull early. We handle discarding in the fragment shader.
        material.alphaTest = 0.001; 
        material.transparent = true;

        // Custom blending configuration for premultiplied alpha accumulation
        material.blending = THREE.CustomBlending;
        material.blendEquation = THREE.AddEquation;
        material.blendSrc = THREE.OneFactor; // Fragment shader outputs premultiplied color: color * opacity
        material.blendDst = THREE.OneMinusSrcAlphaFactor;
        material.blendSrcAlpha = THREE.OneFactor;
        material.blendDstAlpha = THREE.OneMinusSrcAlphaFactor;

        // Polygon offset moves depth slightly to resolve overlapping artifacts
        material.polygonOffset = true;
        material.polygonOffsetUnits = delta;

        // Shader modification hooks
        material.onBeforeCompile = (shader: THREE.Shader) => {
            // Bind our local control uniforms to the shader
            shader.uniforms.uUseBiScale = this.uniforms.uUseBiScale;
            shader.uniforms.uOpacityCap = this.uniforms.uOpacityCap;
            shader.uniforms.uDiscardThreshold = this.uniforms.uDiscardThreshold;

            // 1. Declare ndcDepth varying in the vertex shader to pass projection depth
            shader.vertexShader = shader.vertexShader.replace(
                /void\s+main\s*\(\s*\)\s*\{/g,
                `varying float ndcDepth;
                void main() {`
            );
            
            // 2. Assign ndcDepth at the very end of the vertex shader
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
            
            // 4. Implement GES Bi-scale Disc Opacity (Eq. 6-7) vs standard Gaussian opacity
            shader.fragmentShader = shader.fragmentShader.replace(
                /float\s+opacity\s*=\s*exp\(\s*-0\.5\s*\*\s*A\s*\)\s*\*\s*vColor\.a\s*;/g,
                `float g = exp(-0.5 * A);
                float opacity;
                if (uUseBiScale > 0.5) {
                    // Paper Eq. 6-7: α_i = min(τ_i, w_i · G(x))
                    // vColor.a is trained opacity logit, uOpacityCap represents τ_i
                    opacity = min(uOpacityCap, vColor.a * g);
                    if (opacity < uDiscardThreshold || g < 1.0 / 255.0) discard;
                } else {
                    // Standard Gaussian opacity: o_i * G(x)
                    opacity = vColor.a * g;
                    if (opacity < uDiscardThreshold || g < 1.0 / 255.0) discard;
                }`
            );

            // 5. Output premultiplied color and write the correct 3D depth to gl_FragDepth
            shader.fragmentShader = shader.fragmentShader.replace(
                /gl_FragColor\s*=\s*vec4\s*\(\s*color\.rgb\s*,\s*opacity\s*\)\s*;/g,
                `gl_FragColor = vec4(color.rgb * opacity, opacity);
                gl_FragDepth = (ndcDepth + 1.0) / 2.0;`
            );
        };

        material.needsUpdate = true;
    }

    /**
     * Update the delta (polygon offset) interactively.
     */
    public setDelta(delta: number) {
        if (this.viewer && this.viewer.viewer?.splatMesh?.material) {
            this.viewer.viewer.splatMesh.material.polygonOffsetUnits = delta;
        }
    }

    /**
     * Toggle between GES Bi-scale Disc Opacity and standard Gaussian Opacity.
     */
    public setUseBiScale(enabled: boolean) {
        this.uniforms.uUseBiScale.value = enabled ? 1.0 : 0.0;
    }

    /**
     * Update the opacity capping threshold (tau).
     */
    public setOpacityCap(value: number) {
        this.uniforms.uOpacityCap.value = value;
    }

    /**
     * Update the early discard threshold for transparency culling.
     */
    public setDiscardThreshold(value: number) {
        this.uniforms.uDiscardThreshold.value = value;
    }

    /**
     * Set the visibility of the surfels in the scene.
     */
    public setVisible(visible: boolean) {
        if (this.viewer) {
            this.viewer.visible = visible;
        }
    }
}
