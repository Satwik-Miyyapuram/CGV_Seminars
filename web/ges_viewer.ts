import * as THREE from "three";
import { SceneManager } from "./components/viewer/SceneManager";
import { SurfelLoader } from "./components/viewer/SurfelLoader";
import { GaussianLoader } from "./components/viewer/GaussianLoader";
import { UIController } from "./components/viewer/UIController";
import { ConfigLoader } from "./components/viewer/ConfigLoader";

/**
 * Main application entry point for the GES Web Viewer.
 * Instantiates the core modules and kicks off rendering and default asset autoloading.
 */
async function main() {
    try {
        // 1. Initialize SceneManager (Controls scene, camera, renderer, and composting passes)
        const sceneManager = new SceneManager("viewer");

        // 2. Initialize loaders for Surfels and Gaussians
        const surfelLoader = new SurfelLoader(sceneManager.scene);
        const gaussianLoader = new GaussianLoader(sceneManager.scene);

        // 3. Initialize UIController to bind sliders, checkboxes and file inputs
        const uiController = new UIController(sceneManager, surfelLoader, gaussianLoader);
        uiController.bindEvents();

        // 4. Start the Three.js rendering animation loop
        sceneManager.start();

        // 5. Optionally, automatically look for default assets in '/web_assets'
        // To enable auto-loading of default assets on startup, uncomment the block below:
        /*
        await ConfigLoader.autoLoadDefaultAssets({
            onConfigLoaded: (config) => {
                if (config.background_color) {
                    const bg = config.background_color;
                    sceneManager.setBackgroundColor(new THREE.Color(bg[0], bg[1], bg[2]));
                }
            },
            onSurfelsLoaded: async (url) => {
                const deltaVal = parseFloat((document.getElementById('deltaSlider') as HTMLInputElement).value);
                await surfelLoader.load(url, deltaVal);
                
                // Sync with current UI state
                surfelLoader.setUseBiScale((document.getElementById('toggleBiScale') as HTMLInputElement).checked);
                surfelLoader.setVisible((document.getElementById('toggleSurfels') as HTMLInputElement).checked);
            },
            onGaussiansLoaded: async (url) => {
                await gaussianLoader.load(url);
                
                // Sync with current UI state
                gaussianLoader.setVisible((document.getElementById('toggleGaussians') as HTMLInputElement).checked);
                gaussianLoader.setDepthTest((document.getElementById('toggleDepthTest') as HTMLInputElement).checked);
            }
        });
        */

    } catch (error) {
        console.error("[main] Failed to initialize GES-Explorer:", error);
    }
}

// Kickstart application on DOM load
document.addEventListener("DOMContentLoaded", main);