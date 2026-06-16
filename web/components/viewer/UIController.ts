import { SceneManager } from "./SceneManager";
import { SurfelLoader } from "./SurfelLoader";
import { GaussianLoader } from "./GaussianLoader";
import { ConfigLoader } from "./ConfigLoader";
import { ScenePlyLoader } from "./ScenePlyLoader";

/**
 * UIController encapsulates all user interface event listener bindings,
 * updating DOM labels, and dispatching parameter changes to the loaders
 * and the scene manager.
 */
export class UIController {
    private sceneManager: SceneManager;
    private surfelLoader: SurfelLoader;
    private gaussianLoader: GaussianLoader;

    constructor(
        sceneManager: SceneManager,
        surfelLoader: SurfelLoader,
        gaussianLoader: GaussianLoader
    ) {
        this.sceneManager = sceneManager;
        this.surfelLoader = surfelLoader;
        this.gaussianLoader = gaussianLoader;
    }

    /**
     * Bind event listeners to all interactive HTML control elements.
     */
    public bindEvents() {
        this.bindFileInputs();
        this.bindParameterSliders();
        this.bindViewOptions();
        this.bindTabNavigation();
    }

    /**
     * File inputs for custom Surfels, Gaussians, and Config JSON files.
     */
    private bindFileInputs() {
        // 0. Combined Scene File Input (single scene.ply with surfels + gaussians)
        const sceneInput = document.getElementById('sceneInput') as HTMLInputElement;
        if (sceneInput) {
            sceneInput.addEventListener('change', async (e) => {
                const file = (e.target as HTMLInputElement).files?.[0];
                if (file) {
                    try {
                        const buffer = await file.arrayBuffer();
                        const split = ScenePlyLoader.splitSceneBuffer(buffer);

                        if (split.surfelUrl) {
                            const deltaVal = parseFloat((document.getElementById('deltaSlider') as HTMLInputElement).value);
                            await this.surfelLoader.load(split.surfelUrl, deltaVal);
                            this.surfelLoader.setUseBiScale((document.getElementById('toggleBiScale') as HTMLInputElement).checked);
                            this.surfelLoader.setVisible((document.getElementById('toggleSurfels') as HTMLInputElement).checked);
                            ConfigLoader.updateFileLabel('surfelLabel', 'surfelLoaded', `${file.name} (surfels)`);
                        }
                        if (split.gaussianUrl) {
                            await this.gaussianLoader.load(split.gaussianUrl);
                            this.gaussianLoader.setVisible((document.getElementById('toggleGaussians') as HTMLInputElement).checked);
                            this.gaussianLoader.setDepthTest((document.getElementById('toggleDepthTest') as HTMLInputElement).checked);
                            ConfigLoader.updateFileLabel('gaussianLabel', 'gaussianLoaded', `${file.name} (gaussians)`);
                        }

                        ConfigLoader.updateFileLabel('sceneLabel', 'sceneLoaded', file.name);
                        console.log(`[UIController] Loaded combined scene: ${split.surfelCount} surfels, ${split.gaussianCount} gaussians.`);
                    } catch (err) {
                        console.error("Failed to load combined scene:", err);
                        alert("Error loading combined scene. Make sure it is a GES scene.ply with a prim_type field.");
                    }
                }
            });
        }

        // 1. Surfels File Input
        const surfelInput = document.getElementById('surfelInput') as HTMLInputElement;
        if (surfelInput) {
            surfelInput.addEventListener('change', async (e) => {   
                const file = (e.target as HTMLInputElement).files?.[0];
                if (file) {
                    const url = URL.createObjectURL(file);
                    try {
                        const deltaVal = parseFloat((document.getElementById('deltaSlider') as HTMLInputElement).value);
                        await this.surfelLoader.load(url, deltaVal);
                        
                        // Sync with current bi-scale toggle
                        const useBiScale = (document.getElementById('toggleBiScale') as HTMLInputElement).checked;
                        this.surfelLoader.setUseBiScale(useBiScale);
                        
                        // Sync visibility checkbox
                        const isVisible = (document.getElementById('toggleSurfels') as HTMLInputElement).checked;
                        this.surfelLoader.setVisible(isVisible);

                        ConfigLoader.updateFileLabel('surfelLabel', 'surfelLoaded', file.name);
                    } catch (err) {
                        console.error("Failed to load surfels:", err);
                        alert("Error loading surfels. Make sure it is a valid 2DGS/surfel PLY file.");
                    }
                }
            });
        }

        // 2. Gaussians File Input
        const gaussianInput = document.getElementById('gaussianInput') as HTMLInputElement;
        if (gaussianInput) {
            gaussianInput.addEventListener('change', async (e) => { 
                const file = (e.target as HTMLInputElement).files?.[0];
                if (file) {
                    const url = URL.createObjectURL(file);
                    try {
                        await this.gaussianLoader.load(url);
                        
                        // Sync visibility checkbox
                        const isVisible = (document.getElementById('toggleGaussians') as HTMLInputElement).checked;
                        this.gaussianLoader.setVisible(isVisible);

                        // Sync depth culling checkbox
                        const isDepthTest = (document.getElementById('toggleDepthTest') as HTMLInputElement).checked;
                        this.gaussianLoader.setDepthTest(isDepthTest);

                        ConfigLoader.updateFileLabel('gaussianLabel', 'gaussianLoaded', file.name);
                    } catch (err) {
                        console.error("Failed to load gaussians:", err);
                        alert("Error loading Gaussians. Make sure it is a valid 3DGS PLY file.");
                    }
                }
            });
        }

        // 3. Config JSON File Input
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
                            this.sceneManager.setBackgroundColor(new THREE.Color(bg[0], bg[1], bg[2]));
                            ConfigLoader.updateFileLabel('configLabel', 'configLoaded', file.name);
                            console.log("[UIController] Loaded background color from config:", bg);
                        }
                    } catch (err) {
                        console.error("Failed to parse config.json", err);
                        alert("Invalid config.json format!");
                    }
                }
            });
        }
    }

    /**
     * Sliders for Delta, Opacity Capping (Tau), and Discard Threshold.
     */
    private bindParameterSliders() {
        // 1. Delta Slider (Surfel depth culling offset)
        const deltaSlider = document.getElementById('deltaSlider') as HTMLInputElement;
        if (deltaSlider) {
            deltaSlider.addEventListener('input', (event) => {
                const value = parseFloat((event.target as HTMLInputElement).value);
                const valueEl = document.getElementById('deltaValue');
                if (valueEl) valueEl.innerText = value.toFixed(1);
                this.surfelLoader.setDelta(value);
            });
        }

        // 2. Opacity Capping (Tau) Slider
        const opacityCapSlider = document.getElementById('opacityCapSlider') as HTMLInputElement;
        if (opacityCapSlider) {
            opacityCapSlider.addEventListener('input', (event) => {
                const value = parseFloat((event.target as HTMLInputElement).value);
                const valueEl = document.getElementById('opacityCapValue');
                if (valueEl) valueEl.innerText = value.toFixed(2);
                this.surfelLoader.setOpacityCap(value);
            });
        }

        // 3. Early Discard Threshold Slider
        const discardThresholdSlider = document.getElementById('discardThresholdSlider') as HTMLInputElement;
        if (discardThresholdSlider) {
            discardThresholdSlider.addEventListener('input', (event) => {
                const value = parseFloat((event.target as HTMLInputElement).value);
                const valueEl = document.getElementById('discardThresholdValue');
                if (valueEl) valueEl.innerText = value.toFixed(3);
                this.surfelLoader.setDiscardThreshold(value);
            });
        }
    }

    /**
     * Checkboxes for toggling view features.
     */
    private bindViewOptions() {
        // 1. Toggle Bi-Scale Opacity
        const toggleBiScale = document.getElementById('toggleBiScale') as HTMLInputElement;
        if (toggleBiScale) {
            toggleBiScale.addEventListener('change', (event) => {
                const checked = (event.target as HTMLInputElement).checked;
                this.surfelLoader.setUseBiScale(checked);
            });
        }

        // 2. Show/Hide Surfels
        const toggleSurfels = document.getElementById('toggleSurfels') as HTMLInputElement;
        if (toggleSurfels) {
            toggleSurfels.addEventListener('change', (e) => {
                const checked = (e.target as HTMLInputElement).checked;
                this.surfelLoader.setVisible(checked);
            });
        }

        // 3. Show/Hide Gaussians
        const toggleGaussians = document.getElementById('toggleGaussians') as HTMLInputElement;
        if (toggleGaussians) {
            toggleGaussians.addEventListener('change', (e) => {
                const checked = (e.target as HTMLInputElement).checked;
                this.gaussianLoader.setVisible(checked);
            });
        }

        // 4. Enable/Disable Depth Culling (depthTest for Gaussians)
        const toggleDepthTest = document.getElementById('toggleDepthTest') as HTMLInputElement;
        if (toggleDepthTest) {
            toggleDepthTest.addEventListener('change', (e) => {
                const checked = (e.target as HTMLInputElement).checked;
                this.gaussianLoader.setDepthTest(checked);
            });
        }
    }

    /**
     * Tabs switching logic between GES Web Viewer and Artifact Comparison.
     */
    private bindTabNavigation() {
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
                comparisonContainer.style.display = 'block';

                // Dispatch layout resize event for comparison viewers
                window.dispatchEvent(new Event('resize'));
            });
        }
    }
}
