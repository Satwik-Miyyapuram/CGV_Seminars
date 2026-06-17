import { GesViewer } from "./components/viewer/GesViewer";
import { UIController } from "./components/viewer/UIController";
import { ConfigLoader } from "./components/viewer/ConfigLoader";
import { ComparisonTab } from "./components/comparison/ComparisonTab";

/**
 * Single entry point. On load it wires the three pieces that talk to each other only through
 * window events (sceneSplit / viewerMode / viewerBackground):
 *   - GesViewer    — the 3D renderer (GES Web Viewer tab)
 *   - UIController — the #ui control panel + tab switching
 *   - ComparisonTab — the Artifact Comparison tab
 * then autoloads the default scene. The renderer is constructed first so its `sceneSplit`
 * listener is registered before the autoloader dispatches that event.
 */
document.addEventListener("DOMContentLoaded", async () => {
    new GesViewer();

    const ui = new UIController();
    ui.bindEvents();

    const comparison = new ComparisonTab();
    // Lay out the comparison canvases once the DOM has real dimensions.
    setTimeout(() => comparison.handleResize(), 150);

    try {
        await ConfigLoader.autoLoadDefaultAssets({
            onConfigLoaded: (config) => {
                if (config.background_color) {
                    window.dispatchEvent(new CustomEvent("viewerBackground", { detail: config.background_color }));
                }
            },
            // The renderer loads from the `sceneSplit` event the autoloader dispatches; these
            // per-layer callbacks are intentionally no-ops here.
            onSurfelsLoaded: async () => {},
            onGaussiansLoaded: async () => {},
        });
    } catch (error) {
        console.error("[main] Failed to initialize GES-Explorer:", error);
    }
});
