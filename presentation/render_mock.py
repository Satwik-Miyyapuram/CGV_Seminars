import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend so it saves in the background without popping up windows
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse

# --- 1. Canvas Setup ---
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('#f8f9fa')
ax.set_facecolor('#ffffff')
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

# UI Elements
title_text = ax.text(50, 93, "Gaussian-Surfel Joint Optimization Pipeline", fontsize=16, fontweight='bold', ha='center', color='#212529')
iter_text = ax.text(5, 87, "", fontsize=14, fontweight='bold', color='#0077b6')
phase_text = ax.text(5, 82, "", fontsize=12, color='#495057')
action_text = ax.text(5, 77, "", fontsize=12, color='#e63946', fontweight='bold')

ax.plot([5, 95], [5, 5], color='#dee2e6', lw=4, zorder=1)
progress_fill, = ax.plot([], [], color='#0077b6', lw=4, zorder=2)

# --- 2. Master Data Arrays (State is preserved across all animations!) ---
np.random.seed(42)
MAX_SURFELS = 100
MAX_GAUSSIANS = 150

s_active = np.zeros(MAX_SURFELS, dtype=bool)
s_x = np.zeros(MAX_SURFELS); s_y = np.zeros(MAX_SURFELS)
s_w = np.zeros(MAX_SURFELS); s_h = np.zeros(MAX_SURFELS)
s_angle = np.zeros(MAX_SURFELS); s_wi = np.zeros(MAX_SURFELS)

start_count = 40
s_active[:start_count] = True
s_x[:start_count] = np.random.uniform(20, 80, start_count)
s_y[:start_count] = np.random.uniform(20, 65, start_count)
s_w[:start_count] = np.random.uniform(5, 12, start_count)
s_h[:start_count] = np.random.uniform(2, 6, start_count)
s_angle[:start_count] = np.random.uniform(0, 360, start_count)
s_wi[:start_count] = np.random.uniform(0.1, 15.0, start_count)

s_base_colors = plt.get_cmap('coolwarm')(np.linspace(0.1, 0.9, MAX_SURFELS))
discarded_x, discarded_y = [], []

g_active = np.zeros(MAX_GAUSSIANS, dtype=bool)
g_x = np.zeros(MAX_GAUSSIANS); g_y = np.zeros(MAX_GAUSSIANS)
g_w = np.zeros(MAX_GAUSSIANS); g_h = np.zeros(MAX_GAUSSIANS)
g_angle = np.zeros(MAX_GAUSSIANS)
g_base_colors = plt.get_cmap('viridis')(np.linspace(0.2, 0.9, MAX_GAUSSIANS))

# --- 3. Graphics Patch Setup ---
surfel_patches_out = [Ellipse((0,0),0,0, angle=0, zorder=3) for _ in range(MAX_SURFELS)]
surfel_patches_in = [Ellipse((0,0),0,0, angle=0, zorder=4) for _ in range(MAX_SURFELS)]
for po, pi in zip(surfel_patches_out, surfel_patches_in): ax.add_patch(po); ax.add_patch(pi)

gaussian_patches_out = [Ellipse((0,0),0,0, angle=0, zorder=5) for _ in range(MAX_GAUSSIANS)]
gaussian_patches_in = [Ellipse((0,0),0,0, angle=0, zorder=6) for _ in range(MAX_GAUSSIANS)]
for po, pi in zip(gaussian_patches_out, gaussian_patches_in): ax.add_patch(po); ax.add_patch(pi)

# Helper function to draw the current mathematical state
def draw_state(iteration, jitter=0, g_jitter=0):
    iter_text.set_text(f"Iteration: {int(iteration):,}")
    progress_fill.set_data([5, 5 + (iteration/30000)*90], [5, 5])
    
    for i in range(MAX_SURFELS):
        if s_active[i]:
            surfel_patches_out[i].set_center((s_x[i] + jitter, s_y[i] + jitter))
            surfel_patches_out[i].set_width(s_w[i] * 2.5); surfel_patches_out[i].set_height(s_h[i] * 2.5)
            surfel_patches_out[i].set_angle(s_angle[i])
            surfel_patches_in[i].set_center((s_x[i] + jitter, s_y[i] + jitter))
            surfel_patches_in[i].set_width(s_w[i]); surfel_patches_in[i].set_height(s_h[i])
            surfel_patches_in[i].set_angle(s_angle[i])
            c = list(s_base_colors[i][:3])
            
            if s_wi[i] < 30: # Phase 1 styling
                surfel_patches_out[i].set_facecolor(c + [0.1])
                surfel_patches_in[i].set_facecolor(c + [0.3])
                surfel_patches_in[i].set_edgecolor('none')
            else: # Phase 2 & 3 styling
                alpha = np.clip(s_wi[i] / 255.0, 0.3, 1.0)
                surfel_patches_out[i].set_facecolor(c + [max(0.0, 0.1 - (alpha-0.3))])
                surfel_patches_in[i].set_facecolor(c + [alpha])
                if s_wi[i] >= 255.0:
                    surfel_patches_in[i].set_edgecolor('#212529')
                    surfel_patches_in[i].set_linewidth(1.5)
                else:
                    surfel_patches_in[i].set_edgecolor('none')
        else:
            surfel_patches_out[i].set_facecolor([0,0,0,0])
            surfel_patches_in[i].set_facecolor([0,0,0,0])
            surfel_patches_in[i].set_edgecolor('none')

    for i in range(MAX_GAUSSIANS):
        if g_active[i]:
            c = list(g_base_colors[i][:3])
            gaussian_patches_out[i].set_center((g_x[i] + g_jitter, g_y[i] + g_jitter))
            gaussian_patches_out[i].set_width(g_w[i] * 2.5); gaussian_patches_out[i].set_height(g_h[i] * 2.5)
            gaussian_patches_out[i].set_angle(g_angle[i])
            gaussian_patches_out[i].set_facecolor(c + [0.3])
            gaussian_patches_out[i].set_edgecolor('none')
            gaussian_patches_in[i].set_center((g_x[i] + g_jitter, g_y[i] + g_jitter))
            gaussian_patches_in[i].set_width(g_w[i]); gaussian_patches_in[i].set_height(g_h[i])
            gaussian_patches_in[i].set_angle(g_angle[i])
            gaussian_patches_in[i].set_facecolor(c + [0.9])
            gaussian_patches_in[i].set_edgecolor('none')
        else:
            gaussian_patches_out[i].set_facecolor([0,0,0,0])
            gaussian_patches_in[i].set_facecolor([0,0,0,0])
            
    return [iter_text, phase_text, action_text, progress_fill] + surfel_patches_out + surfel_patches_in + gaussian_patches_out + gaussian_patches_in


# --- 4. Animation Generators ---
# Since we do not reset the arrays, each phase automatically picks up where the last one left off!
frames_per_phase = 150

def update_phase1(frame):
    iteration = frame * (10000 / frames_per_phase)
    phase_text.set_text("Phase 1: 0 - 10K Iterations"); phase_text.set_color('#00a8e8')
    action_text.set_text("Action: Densifying & optimizing translucent 2D Surfels")
    progress_fill.set_color('#00a8e8')
    
    if frame % 4 == 0 and np.sum(s_active) < MAX_SURFELS - 1:
        active_idx = np.where(s_active)[0]
        if len(active_idx) > 0:
            parent = np.random.choice(active_idx)
            new_idx = np.where(~s_active)[0][0]
            s_active[new_idx] = True
            s_x[new_idx] = s_x[parent] + np.random.uniform(-3, 3)
            s_y[new_idx] = np.clip(s_y[parent] + np.random.uniform(-3, 3), 15, 65)
            s_w[new_idx] = s_w[parent] * 0.8; s_h[new_idx] = s_h[parent] * 0.8
            s_angle[new_idx] = s_angle[parent]; s_wi[new_idx] = s_wi[parent]
            
    s_wi[s_active] += np.random.normal(0, 0.5, np.sum(s_active))
    s_wi[s_active] = np.clip(s_wi[s_active], 0.1, 29.9)
    return draw_state(iteration, jitter=np.sin(frame * 0.2) * 0.3)


def update_phase2(frame):
    iteration = 10000 + frame * (10000 / frames_per_phase)
    phase_text.set_text("Phase 2: 10K - 20K Iterations"); phase_text.set_color('#0077b6')
    action_text.set_text("Action: Discarding points, forcing w_i to opaque 2D discs")
    progress_fill.set_color('#0077b6')
    
    if frame == 1 and len(discarded_x) == 0:
        active_idx = np.where(s_active)[0]
        for idx in active_idx:
            if np.random.random() < 0.2: s_wi[idx] = 0.5 
            if s_wi[idx] < 0.8:
                s_active[idx] = False
                discarded_x.append(s_x[idx]); discarded_y.append(s_y[idx])
    
    t = frame / float(frames_per_phase)
    s_wi[s_active] = 30 + (t * 225) 
    return draw_state(iteration)


def update_phase3(frame):
    iteration = 20000 + frame * (10000 / frames_per_phase)
    phase_text.set_text("Phase 3: 20K+ Iterations"); phase_text.set_color('#2a9d8f')
    action_text.set_text("Action: Geometry Locked. 3D Gaussians spawned in voids.")
    progress_fill.set_color('#2a9d8f')
    
    if frame == 1 and np.sum(g_active) == 0:
        s_wi[s_active] = 255.0 
        for i, (dx, dy) in enumerate(zip(discarded_x, discarded_y)):
            if i < MAX_GAUSSIANS:
                g_active[i] = True
                g_x[i] = dx + np.random.normal(0, 1.5); g_y[i] = dy + np.random.normal(0, 1.5)
                g_w[i] = np.random.uniform(1.5, 4.0); g_h[i] = np.random.uniform(0.5, 2.0)
                g_angle[i] = np.random.uniform(0, 360)
    
    if frame % 5 == 0 and np.sum(g_active) < MAX_GAUSSIANS - 1 and np.sum(g_active) > 0:
        active_g = np.where(g_active)[0]
        parent = np.random.choice(active_g)
        new_idx = np.where(~g_active)[0][0]
        g_active[new_idx] = True
        g_x[new_idx] = g_x[parent] + np.random.uniform(-2, 2)
        g_y[new_idx] = np.clip(g_y[parent] + np.random.uniform(-2, 2), 15, 65)
        g_w[new_idx] = g_w[parent] * 0.8; g_h[new_idx] = g_h[parent] * 0.8
        g_angle[new_idx] = g_angle[parent]

    return draw_state(iteration, g_jitter=np.sin(frame * 0.4) * 0.2)

# --- 5. Export Sequentially ---
writer = animation.PillowWriter(fps=30)

print("Rendering Phase 1 (0-10K)...")
ani1 = animation.FuncAnimation(fig, update_phase1, frames=frames_per_phase, blit=True)
ani1.save('Phase1_0_10k.gif', writer=writer)

print("Rendering Phase 2 (10K-20K)...")
ani2 = animation.FuncAnimation(fig, update_phase2, frames=frames_per_phase, blit=True)
ani2.save('Phase2_10k_20k.gif', writer=writer)

print("Rendering Phase 3 (20K+)...")
ani3 = animation.FuncAnimation(fig, update_phase3, frames=frames_per_phase, blit=True)
ani3.save('Phase3_20k_plus.gif', writer=writer)

print("All 3 GIFs saved successfully! They mathematically perfectly connect to each other.")