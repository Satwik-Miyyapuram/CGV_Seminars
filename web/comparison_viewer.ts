import { ComparisonTab } from "./components/comparison/ComparisonTab";

// Entry point for the redesigned Comparison tab (Tab B). ComparisonTab builds the tab's
// markup and wires up all the diagrams/3D viewers via ComparisonManager.
document.addEventListener("DOMContentLoaded", () => {
    const tab = new ComparisonTab();

    // Trigger an initial layout pass shortly after load so every canvas measures correctly
    // once the container has been laid out.
    setTimeout(() => tab.handleResize(), 150);
});
