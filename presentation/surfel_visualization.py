import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# 1. Setup Data and Grid
axis_limit = 5.0
X_1D = np.linspace(-axis_limit, axis_limit, 800)
X_2D = np.linspace(-axis_limit, axis_limit, 500)
Y_2D = np.linspace(-axis_limit, axis_limit, 500)
XX, YY = np.meshgrid(X_2D, Y_2D)
R2 = XX**2 + YY**2

# Paper's specific math constants
threshold = 1.0 / 255.0
max_radius = np.sqrt(2 * np.log(255))  # ≈ 3.33

# Logarithmic growth from 0.1 to 255, then smoothly back down
wi_up = np.logspace(np.log10(0.1), np.log10(255), 75)
wi_down = np.logspace(np.log10(255), np.log10(0.1), 75)
wi_values = np.concatenate([wi_up, wi_down])
bg_color = "#ffffff"  # Updated global background color
# Light Theme Cyan Colormap - Updated to fade into #f8fafc
colors = [bg_color, "#00A6D6"]
cyan_cmap = LinearSegmentedColormap.from_list("custom_cyan_light", colors)

# 2. Setup Figure and Styling
plt.style.use("default")
fig, (ax_2d, ax_1d) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 1]})
fig.patch.set_facecolor(bg_color)  # Updated global bg color
fig.subplots_adjust(top=0.82, bottom=0.15, wspace=0.3)
# fig.suptitle('Coarse-to-Fine Surfel Hardening', fontsize=24, color='#000000', fontweight='bold')

# --- 2D Top-Down View Setup ---
ax_2d.set_facecolor(bg_color)  # Updated 2D plot bg color
ax_2d.axis("off")
ax_2d.set_title("Top-Down View", color="#333333", pad=20, fontsize=16)
im = ax_2d.imshow(
    np.zeros_like(R2),
    cmap=cyan_cmap,
    vmin=0,
    vmax=1,
    origin="lower",
    extent=[-axis_limit, axis_limit, -axis_limit, axis_limit],
)

# Draw a faint circle showing the theoretical max radius (r ≈ 3.3)
circle = plt.Circle((0, 0), max_radius, color="#cbd5e1", fill=False, linestyle="--", linewidth=1.5)
ax_2d.add_patch(circle)
ax_2d.text(
    0, max_radius + 0.2, "Max Opaque Bound (r ≈ 3.3)", color="#64748b", ha="center", fontsize=10
)

# --- 1D Graph Setup ---
ax_1d.set_facecolor(bg_color)
ax_1d.set_ylim(0, 1.2)  # Focused Y-axis to show the clipping clearly
ax_1d.set_xlim(-axis_limit, axis_limit)
ax_1d.set_title("Opacity Clipping Profile", color="#333333", pad=20, fontsize=16)
ax_1d.set_xlabel("Distance from center (r)", color="#333333", fontsize=12)
ax_1d.set_ylabel("Opacity (\u03b1)", color="#333333", fontsize=12)
ax_1d.grid(color="#cbd5e1", linestyle="--", alpha=0.8)

# The Math Guide Lines
ax_1d.axhline(1.0, color="#64748b", linestyle="--", label="Max Opacity (1.0)")
ax_1d.axhline(threshold, color="#ef4444", linestyle=":", label="Cutoff (1/255)", alpha=0.5)

# The Dynamic Line
(line_clipped,) = ax_1d.plot([], [], color="#00A6D6", linewidth=3, label="Surfel Opacity (α)")
ax_1d.legend(loc="upper right", facecolor=bg_color, edgecolor="#cbd5e1")  # Updated legend bg color

# Dynamic Text for w_i
wi_text = ax_1d.text(
    0.05, 0.85, "", transform=ax_1d.transAxes, fontsize=16, color="#000000", fontweight="bold"
)

# Pre-calculate base Gaussian values to save time in loop
G_2D = np.exp(-R2 / 2.0)
G_1D = np.exp(-(X_1D**2) / 2.0)


# 3. Animation Update Function
def update(frame):
    wi = wi_values[frame]

    # --- 2D Math ---
    alpha_2D = np.clip(wi * G_2D, 0, 1)
    # The paper's hard cutoff: clamped to 0 when min(alpha, G) < 1/255
    clamp_mask_2D = np.minimum(alpha_2D, G_2D) < threshold
    alpha_2D[clamp_mask_2D] = 0.0
    im.set_array(alpha_2D)

    # --- 1D Math ---
    alpha_1D = np.clip(wi * G_1D, 0, 1)
    # Apply identical cutoff to the 1D graph
    clamp_mask_1D = np.minimum(alpha_1D, G_1D) < threshold
    alpha_1D[clamp_mask_1D] = 0.0
    line_clipped.set_data(X_1D, alpha_1D)

    # Update Text
    wi_text.set_text(f"w_i = {wi:.1f}")

    return im, line_clipped, wi_text


# 4. Compile and Save Animation
print("Generating mathematically precise animation...")
ani = animation.FuncAnimation(fig, update, frames=len(wi_values), interval=50, blit=True)

output_filename = "surfel_hardening_exact_math.gif"
ani.save(output_filename, writer="pillow", fps=24)
print(f"Animation successfully saved as {output_filename}!")
