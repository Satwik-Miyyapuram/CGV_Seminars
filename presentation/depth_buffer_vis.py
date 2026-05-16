import matplotlib
matplotlib.use('TkAgg') # Forces the interactive window to open
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Rectangle, Polygon

# --- 1. Setup the Expanded Canvas ---
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#f8f9fa')
ax.set_facecolor('#f8f9fa')
ax.set_xlim(0, 120)
ax.set_ylim(-30, 60) 
ax.axis('off')

ax.axhline(5, color='#dee2e6', lw=2, zorder=1)
ax.text(2, -2, "1D Depth Sort Buffer (Distance from Camera)", fontsize=12, fontweight='bold', color='#495057')

# --- 2. Draw the Camera ---
cx, cy = 10, 35
camera_body = Rectangle((2, 30), 8, 10, edgecolor='#343a40', facecolor='#e9ecef', lw=2, zorder=10)
camera_lens = Polygon([[10, 32], [15, 27], [15, 43], [10, 38]], edgecolor='#343a40', facecolor='#e9ecef', lw=2, zorder=10)
ax.add_patch(camera_body)
ax.add_patch(camera_lens)

# --- 3. Draw the Numbered Depth Axis ---
depth_y = -18 
ax.plot([cx, cx + 100], [depth_y, depth_y], color='#ced4da', lw=2, zorder=1) 

for dist in range(0, 101, 20):
    ax.plot([cx + dist, cx + dist], [depth_y, depth_y - 1.5], color='#adb5bd', lw=2, zorder=1)
    ax.text(cx + dist, depth_y - 4, str(dist), color='#6c757d', fontsize=10, ha='center', va='center')

# --- 4. Generate the Gaussian Ellipses & Text Objects ---
np.random.seed(42)
num_splats = 40
splat_x = np.random.uniform(25, 110, num_splats)
splat_y = np.random.uniform(15, 55, num_splats)

splat_w = np.random.uniform(5, 12, num_splats) 
splat_h = np.random.uniform(3, 7, num_splats)  
splat_angles = np.random.uniform(0, 360, num_splats) 

base_colors = plt.get_cmap('Set2')(np.linspace(0, 1, num_splats))

scene_outer, scene_inner, scene_texts = [], [], []
depth_outer, depth_inner, depth_texts = [], [], []

for i in range(num_splats):
    e_out = Ellipse((splat_x[i], splat_y[i]), splat_w[i]*2.5, splat_h[i]*2.5, angle=splat_angles[i], zorder=5)
    e_in = Ellipse((splat_x[i], splat_y[i]), splat_w[i], splat_h[i], angle=splat_angles[i], zorder=6)
    t_scene = ax.text(splat_x[i], splat_y[i], "", ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=7, alpha=0)
    
    d_out = Ellipse((0, depth_y), splat_w[i]*2.5, splat_h[i]*2.5, angle=splat_angles[i], zorder=5, facecolor=[0,0,0,0])
    d_in = Ellipse((0, depth_y), splat_w[i], splat_h[i], angle=splat_angles[i], zorder=6, facecolor=[0,0,0,0])
    t_depth = ax.text(0, depth_y, "", ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=7, alpha=0)
    
    e_out.set_edgecolor('none'); e_in.set_edgecolor('none')
    d_out.set_edgecolor('none'); d_in.set_edgecolor('none')
    
    ax.add_patch(e_out); ax.add_patch(e_in)
    ax.add_patch(d_out); ax.add_patch(d_in)
    
    scene_outer.append(e_out); scene_inner.append(e_in); scene_texts.append(t_scene)
    depth_outer.append(d_out); depth_inner.append(d_in); depth_texts.append(t_depth)

# --- 5. Create the Ray Lines ---
ray_line, = ax.plot([], [], color='#ff4757', lw=3, zorder=4)
depth_ray_line, = ax.plot([], [], color='#ff4757', lw=2, zorder=4, linestyle='--')

angles_rad = np.radians(splat_angles)
cos_a = np.cos(angles_rad)
sin_a = np.sin(angles_rad)

# --- 6. The Animation Loop ---
def update(frame):
    cycle_frame = frame % 80
    cycle_num = frame // 80
    
    angles = [12, -8, 18, -3, -15]
    ray_angle_rad = np.radians(angles[cycle_num % len(angles)])
    dx = np.cos(ray_angle_rad)
    dy = np.sin(ray_angle_rad)
    max_ray_len = 100
    
    if cycle_frame < 50:
        current_ray_len = (cycle_frame / 50.0) * max_ray_len
        ray_alpha = 1.0
    else:
        current_ray_len = max_ray_len
        ray_alpha = max(0, 1.0 - (cycle_frame - 50) / 15.0) 
        
    end_x = cx + dx * current_ray_len
    end_y = cy + dy * current_ray_len
    
    ray_line.set_data([cx, end_x], [cy, end_y])
    ray_line.set_alpha(ray_alpha)
    depth_ray_line.set_data([cx, cx + current_ray_len], [depth_y, depth_y])
    depth_ray_line.set_alpha(ray_alpha)
    
    # --- INTERSECTION MATH ---
    ray_px = cx + np.linspace(0, current_ray_len, 100) * dx
    ray_py = cy + np.linspace(0, current_ray_len, 100) * dy
    
    rx = ray_px[:, np.newaxis] - splat_x[np.newaxis, :]
    ry = ray_py[:, np.newaxis] - splat_y[np.newaxis, :]
    
    u = rx * cos_a + ry * sin_a
    v = -rx * sin_a + ry * cos_a
    
    inside = (u**2 / (splat_w*1.25)**2) + (v**2 / (splat_h*1.25)**2) <= 1.0
    is_hit = np.any(inside, axis=0) 
    
    if cycle_frame > 65:
        is_hit[:] = False 

    # --- DYNAMIC DEPTH SORTING MATH ---
    # 1. Calculate distances for all splats
    forward_dists = (splat_x - cx) * dx + (splat_y - cy) * dy
    
    # 2. Extract hits and sort them by distance
    hits = [(i, forward_dists[i]) for i in range(num_splats) if is_hit[i]]
    hits.sort(key=lambda x: x[1]) # Sort by distance
    
    # 3. Create a mapping of splat_index -> depth_rank (1, 2, 3...)
    rank_map = {hit[0]: rank for rank, hit in enumerate(hits, 1)}
        
    # --- Update Visuals & Text ---
    grey_out = [0.6, 0.6, 0.6, 0.15]
    grey_in  = [0.5, 0.5, 0.5, 0.25]
    
    for i in range(num_splats):
        if is_hit[i]:
            rank = rank_map[i]
            dist = forward_dists[i]
            
            c_out = list(base_colors[i]); c_out[3] = 0.6 
            c_in = list(base_colors[i]); c_in[3] = 1.0  
            
            # Show in Scene with Dynamic Rank
            scene_outer[i].set_facecolor(c_out)
            scene_inner[i].set_facecolor(c_in)
            scene_texts[i].set_text(str(rank))
            scene_texts[i].set_alpha(1.0) 
            
            # Project onto Depth Buffer with Dynamic Rank
            depth_outer[i].set_center((cx + dist, depth_y))
            depth_inner[i].set_center((cx + dist, depth_y))
            depth_outer[i].set_facecolor(c_out)
            depth_inner[i].set_facecolor(c_in)
            
            depth_texts[i].set_position((cx + dist, depth_y))
            depth_texts[i].set_text(str(rank))
            depth_texts[i].set_alpha(1.0)
        else:
            scene_outer[i].set_facecolor(grey_out)
            scene_inner[i].set_facecolor(grey_in)
            scene_texts[i].set_alpha(0) 
            
            depth_outer[i].set_facecolor([0,0,0,0])
            depth_inner[i].set_facecolor([0,0,0,0])
            depth_texts[i].set_alpha(0) 
            
    return [ray_line, depth_ray_line] + scene_outer + scene_inner + scene_texts + depth_outer + depth_inner + depth_texts

# --- 7. Render ---
ani = animation.FuncAnimation(fig, update, frames=400, interval=30, blit=True)
# UNCOMMENT BELOW TO SAVE AS GIF
print("Saving raycast_depth_sort.gif...")
writer = animation.PillowWriter(fps=30)
ani.save("raycast_depth_sort.gif", writer=writer)
print("Done!")
print("Opening Rank-Sorted Depth Buffer window...")
plt.show(block=True)