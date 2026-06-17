import * as THREE from "three";
import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";

/**
 * GaussianLoader is responsible for loading the Gaussian Splat PLY file,
 * adding it to the Three.js scene, and compiling the custom shaders that:
 * 1. Output the correct 3D depth to gl_FragDepth so that they are depth-tested
 *    correctly against the Z-buffer written by the Surfels in Pass 1.
 * 2. Configure additive blending (Pass 2) to accumulate colors and opacities.
 */
export class GaussianLoader {
    private sceneManager: any;
    private scene: THREE.Scene;
    public viewer: any | null = null;

    constructor(sceneManager: any) {
        this.sceneManager = sceneManager;
        this.scene = sceneManager.scene;
    }

    /**
     * Load Gaussians from a URL (either local file object URL or path).
     */
    public async load(url: string) {
        this.unload();

        console.log("[GaussianLoader] Loading gaussians from:", url);

        // Instantiate GaussianSplats3D DropInViewer
        this.viewer = new GaussianSplats3D.DropInViewer({
            'gpuAcceleratedSort': false
        });

        // Add the PLY scene
        await this.viewer.addSplatScenes([{
            'path': url,
            'format': GaussianSplats3D.SceneFormat.Ply
        }]);

        this.configureMaterial();
        this.viewer.renderOrder = 1; // Gaussians render after Surfels (Pass 2)
        this.scene.add(this.viewer);
        this.sceneManager.gaussianViewer = this.viewer;

        console.log("[GaussianLoader] Gaussians loaded and configured.");
    }

    /**
     * Unload current gaussians from the scene.
     */
    public unload() {
        if (this.viewer) {
            this.scene.remove(this.viewer);
            this.viewer = null;
            this.sceneManager.gaussianViewer = null;
        }
    }

    /**
     * Inject custom depth testing and additive blending into the splat material shaders.
     */
    private configureMaterial() {
        if (!this.viewer) return;

        const material = this.viewer.viewer.splatMesh.material;
        
        // Depth testing is enabled (to cull Gaussians occluded by Surfels)
        // Depth writing is disabled (Gaussians don't block other Gaussians or Surfels)
        material.depthTest = true;
        material.depthWrite = false;
        material.transparent = true;

        // Custom blending configuration for standard 3DGS premultiplied alpha accumulation
        material.blending = THREE.CustomBlending;
        material.blendEquation = THREE.AddEquation;
        material.blendSrc = THREE.OneFactor; // We will output premultiplied color
        material.blendDst = THREE.OneMinusSrcAlphaFactor;
        material.blendSrcAlpha = THREE.OneFactor;
        material.blendDstAlpha = THREE.OneMinusSrcAlphaFactor;

        // Shader modification hooks
        material.onBeforeCompile = (shader: THREE.Shader) => {
            // Add custom uniform for depth culling delta
            shader.uniforms.uDepthCullDelta = { value: 0.05 };

            // 1. Declare ndcDepth and uniform in the vertex shader
            shader.vertexShader = shader.vertexShader.replace(
                /void\s+main\s*\(\s*\)\s*\{/g,
                `varying float ndcDepth;
                uniform float uDepthCullDelta;
                void main() {`
            );
            
            // 2. Assign ndcDepth at the very end of the vertex shader
            const lastBraceIndex = shader.vertexShader.lastIndexOf('}');
            if (lastBraceIndex !== -1) {
                shader.vertexShader = 
                    shader.vertexShader.substring(0, lastBraceIndex) + 
                    `    
                    // The standard gl_Position is already computed.
                    // We calculate a modified z_clip and w_clip using a view-space offset.
                    // In Three.js, view-space z is negative. Pushing it forward means adding delta.
                    
                    // We need the original view-space Z.
                    // Since gl_Position.w = -viewZ (for perspective projection), viewZ = -gl_Position.w
                    float viewZ = -gl_Position.w;
                    float newViewZ = viewZ + uDepthCullDelta; // Closer to camera
                    
                    // Project the new viewZ using the projection matrix elements
                    // projectionMatrix[2][2] corresponds to the Z-scaling
                    // projectionMatrix[3][2] corresponds to the Z-translation
                    float newClipZ = projectionMatrix[2][2] * newViewZ + projectionMatrix[3][2];
                    float newClipW = -newViewZ;
                    
                    ndcDepth = newClipZ / newClipW;
                    }` + 
                    shader.vertexShader.substring(lastBraceIndex + 1);
            }

            // 3. Declare varying in the fragment shader
            shader.fragmentShader = shader.fragmentShader.replace(
                /varying\s+vec2\s+vPosition\s*;/g,
                `varying vec2 vPosition;
                varying float ndcDepth;`
            );
            
            // 4. Output premultiplied color and write modified 3D depth to gl_FragDepth
            shader.fragmentShader = shader.fragmentShader.replace(
                /gl_FragColor\s*=\s*vec4\s*\(\s*color\.rgb\s*,\s*opacity\s*\)\s*;/g,
                `gl_FragColor = vec4(color.rgb * opacity, opacity);
                gl_FragDepth = (ndcDepth + 1.0) / 2.0;`
            );
        };

        material.needsUpdate = true;
    }

    /**
     * Enable/disable depth culling against the Z-buffer.
     */
    public setDepthTest(enabled: boolean) {
        if (this.viewer && this.viewer.viewer?.splatMesh?.material) {
            this.viewer.viewer.splatMesh.material.depthTest = enabled;
        }
    }

    /**
     * Set the visibility of the gaussians in the scene.
     */
    public setVisible(visible: boolean) {
        if (this.viewer) {
            this.viewer.visible = visible;
        }
    }
}
