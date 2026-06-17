import { ScenePlyLoader } from "./ScenePlyLoader";
import { ConfigLoader } from "./ConfigLoader";

/**
 * UIController wires the glassmorphic #ui control panel to the viewer. It is UI-only: it never
 * touches the renderer directly, it just dispatches window events the viewer listens for:
 *   - `sceneSplit`       { surfelUrl, gaussianUrl, center, radius }  (load a scene)
 *   - `viewerMode`       { surfels, gaussians }                       (show/hide)
 *   - `viewerBackground` [r, g, b]                                    (background colour)
 */
export class UIController {
    public bindEvents() {
        this.bindFileInputs();
        this.bindParameterSliders();
        this.bindViewOptions();
        this.bindTabNavigation();
        this.bindPanelCollapse();
    }

    /** Collapse the control overlay into a small "Controls" chip that re-expands on click. */
    private bindPanelCollapse() {
        const ui = document.getElementById('ui');
        const hideBtn = document.getElementById('ui-hide');
        const showBtn = document.getElementById('ui-show');
        if (!ui || !hideBtn || !showBtn) return;

        const setCollapsed = (collapsed: boolean) => {
            ui.classList.toggle('collapsed', collapsed);
            showBtn.classList.toggle('visible', collapsed);
        };
        hideBtn.addEventListener('click', () => setCollapsed(true));
        showBtn.addEventListener('click', () => setCollapsed(false));
    }

    /** Scene (.ply) and Config (.json) file inputs. */
    private bindFileInputs() {
        const sceneInput = document.getElementById('sceneInput') as HTMLInputElement;
        if (sceneInput) {
            sceneInput.addEventListener('change', async (e) => {
                const file = (e.target as HTMLInputElement).files?.[0];
                if (!file) return;
                try {
                    const buffer = await file.arrayBuffer();
                    const split = ScenePlyLoader.splitSceneBuffer(buffer);
                    window.dispatchEvent(new CustomEvent('sceneSplit', { detail: split }));
                    ConfigLoader.updateFileLabel('sceneLabel', 'sceneLoaded', file.name);
                    console.log(`[UIController] Scene: ${split.surfelCount} surfels, ${split.gaussianCount} gaussians.`);
                } catch (err) {
                    console.error("Failed to load scene:", err);
                    alert("Error loading scene. Make sure it is a GES scene.ply with a prim_type field.");
                }
            });
        }

        const configInput = document.getElementById('configInput') as HTMLInputElement;
        if (configInput) {
            configInput.addEventListener('change', async (e) => {
                const file = (e.target as HTMLInputElement).files?.[0];
                if (!file) return;
                try {
                    const config = JSON.parse(await file.text());
                    if (config.background_color) {
                        window.dispatchEvent(new CustomEvent('viewerBackground', { detail: config.background_color }));
                    }
                    ConfigLoader.updateFileLabel('configLabel', 'configLoaded', file.name);
                } catch (err) {
                    console.error("Failed to parse config.json", err);
                    alert("Invalid config.json format!");
                }
            });
        }
    }

    /** Sliders just keep their value labels in sync (no effect on the z-buffer render path). */
    private bindParameterSliders() {
        const bindLabel = (sliderId: string, labelId: string, digits: number) => {
            const slider = document.getElementById(sliderId) as HTMLInputElement | null;
            const label = document.getElementById(labelId);
            if (slider && label) {
                slider.addEventListener('input', () => {
                    label.innerText = parseFloat(slider.value).toFixed(digits);
                });
            }
        };
    }

    /** Show/hide surfels and gaussians → broadcast the current mode to the viewer. */
    private bindViewOptions() {
        const toggleSurfels = document.getElementById('toggleSurfels') as HTMLInputElement | null;
        const toggleGaussians = document.getElementById('toggleGaussians') as HTMLInputElement | null;

        const dispatchMode = () => {
            window.dispatchEvent(new CustomEvent('viewerMode', {
                detail: {
                    surfels: toggleSurfels ? toggleSurfels.checked : true,
                    gaussians: toggleGaussians ? toggleGaussians.checked : true,
                },
            }));
        };
        toggleSurfels?.addEventListener('change', dispatchMode);
        toggleGaussians?.addEventListener('change', dispatchMode);
    }

    /** Tab switching between the GES Web Viewer and the Artifact Comparison. */
    private bindTabNavigation() {
        const btnViewer = document.getElementById('btn-viewer');
        const btnComparison = document.getElementById('btn-comparison');
        const uiPanel = document.getElementById('ui');
        const viewerEl = document.getElementById('viewer');
        const comparisonContainer = document.getElementById('comparison-container');
        if (!btnViewer || !btnComparison || !uiPanel || !viewerEl || !comparisonContainer) return;

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
            window.dispatchEvent(new Event('resize'));
        });
    }
}
