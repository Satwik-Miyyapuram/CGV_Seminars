import { UIController } from "./components/viewer/UIController";
import { ConfigLoader } from "./components/viewer/ConfigLoader";

/**
 * GES Web Viewer entry point. The actual rendering + camera controls live in
 * combined_viewer.ts (mounted in #viewer); this file only wires the #ui control panel and
 * runs the default-asset autoload. Both communicate with the renderer through window events
 * (sceneSplit / viewerMode / viewerBackground) dispatched by UIController and ConfigLoader.
 */
async function main() {
    try {
        const uiController = new UIController();
        uiController.bindEvents();

        // Autoload default assets. autoLoadDefaultAssets dispatches the `sceneSplit` event the
        // renderer listens for; the surfel/gaussian callbacks are no-ops here (the renderer
        // loads from that event), and the config callback forwards the background colour.
        await ConfigLoader.autoLoadDefaultAssets({
            onConfigLoaded: (config) => {
                if (config.background_color) {
                    window.dispatchEvent(new CustomEvent('viewerBackground', { detail: config.background_color }));
                }
            },
            onSurfelsLoaded: async () => {},
            onGaussiansLoaded: async () => {},
        });
    } catch (error) {
        console.error("[main] Failed to initialize GES-Explorer:", error);
    }
}

document.addEventListener("DOMContentLoaded", main);
