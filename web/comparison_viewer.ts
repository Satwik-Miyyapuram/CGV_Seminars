import { ComparisonManager } from "./components/comparison/ComparisonManager";

// Entry point for the Comparison tab, initializing the main manager on DOM load
document.addEventListener("DOMContentLoaded", () => {
    const comparisonManager = new ComparisonManager();

    // Trigger an initial layout resize call shortly after loading to ensure all
    // canvas dimensions align correctly (since container displays might change)
    setTimeout(() => {
        comparisonManager.handleResize();
    }, 150);
});