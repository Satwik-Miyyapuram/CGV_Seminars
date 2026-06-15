import * as THREE from "three";

/**
 * ConfigLoader is responsible for loading the configuration file (config.json)
 * and managing the auto-loading of default assets (surfels, gaussians, and configs).
 */
export class ConfigLoader {
    /**
     * Helper to update UI labels indicating whether a file has loaded.
     */
    public static updateFileLabel(labelId: string, statusId: string, name: string) {
        const labelEl = document.getElementById(labelId);
        const statusEl = document.getElementById(statusId);
        if (labelEl && statusEl) {
            labelEl.textContent = name;
            labelEl.classList.add('loaded');
            statusEl.textContent = "Loaded";
            statusEl.classList.add('active');
        }
    }

    /**
     * Attempts to auto-load default assets from `/web_assets/`.
     * Invokes callbacks for loaded configurations, surfels, and gaussians.
     */
    public static async autoLoadDefaultAssets(callbacks: {
        onConfigLoaded: (config: any) => void;
        onSurfelsLoaded: (url: string) => Promise<void>;
        onGaussiansLoaded: (url: string) => Promise<void>;
    }) {
        const autoStatus = document.getElementById('autoLoadStatus');
        if (autoStatus) {
            autoStatus.textContent = "Loading...";
            autoStatus.classList.add('active');
        }

        let loadedCount = 0;

        // 1. Fetch Config
        try {
            console.log("[ConfigLoader] Checking for default config.json...");
            const res = await fetch('/web_assets/config.json');
            if (res.ok) {
                const config = await res.json();
                callbacks.onConfigLoaded(config);
                this.updateFileLabel('configLabel', 'configLoaded', 'config.json');
                console.log("[ConfigLoader] Auto-loaded config.json successfully.");
                loadedCount++;
            }
        } catch (err) {
            console.log("[ConfigLoader] config.json auto-load skipped:", err);
        }

        // 2. Fetch Surfels
        try {
            console.log("[ConfigLoader] Checking for default surfels.ply...");
            const res = await fetch('/web_assets/surfels.ply');
            if (res.ok) {
                await callbacks.onSurfelsLoaded('/web_assets/surfels.ply');
                this.updateFileLabel('surfelLabel', 'surfelLoaded', 'surfels.ply');
                console.log("[ConfigLoader] Auto-loaded surfels.ply successfully.");
                loadedCount++;
            }
        } catch (err) {
            console.log("[ConfigLoader] surfels.ply auto-load skipped:", err);
        }

        // 3. Fetch Gaussians
        try {
            console.log("[ConfigLoader] Checking for default gaussians.ply...");
            const res = await fetch('/web_assets/gaussians.ply');
            if (res.ok) {
                await callbacks.onGaussiansLoaded('/web_assets/gaussians.ply');
                this.updateFileLabel('gaussianLabel', 'gaussianLoaded', 'gaussians.ply');
                console.log("[ConfigLoader] Auto-loaded gaussians.ply successfully.");
                loadedCount++;
            }
        } catch (err) {
            console.log("[ConfigLoader] gaussians.ply auto-load skipped:", err);
        }

        // Update main auto-load status badge in UI
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
}
