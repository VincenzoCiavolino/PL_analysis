import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, gaussian_filter
from skimage.feature import peak_local_max
import matplotlib.cm as cm
import tkinter as tk
from tkinter import filedialog
from matplotlib.patches import Rectangle



def load_dat_file(path):
    data = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    values = [float(val) for val in line.split('\t')]
                    data.append(values)
                except ValueError:
                    continue
    return np.array(data)

# Open file selection dialog
root = tk.Tk()
root.withdraw()
file_paths = filedialog.askopenfilenames(
    title="Select 3 .dat files",
    filetypes=[("DAT files", "*.dat")])

# Check correct number of files
if len(file_paths) != 3:
    raise ValueError("You must select exactly 3 .dat files.")

# Load and concatenate
arrays = [load_dat_file(path).T for path in file_paths]
data_array = np.hstack(arrays)

num_columns = 27
column_width = data_array.shape[1] // num_columns

block_height = arrays[0].shape[0]  # assuming all arrays have same height
all_peaks = []

for i in range(num_columns):  # 30 columns
    x_start = i * column_width
    x_end = min((i + 1) * column_width, data_array.shape[1])
    region = data_array[:, x_start:x_end]  # shape (500, ~50)

    coordinates = peak_local_max(
        region,
        min_distance=5,
        threshold_abs=np.percentile(region, 95),
        num_peaks=10
    )
    # Adjust row indices to account for stacking
    adjusted_coordinates = coordinates.copy()
    adjusted_coordinates[:, 1] += x_start
    # Extract intensity values at each detected peak
    peak_values = [data_array[row, col] for row, col in adjusted_coordinates]
    # Combine coordinates with their corresponding values
    peaks_with_values = list(zip(adjusted_coordinates, peak_values))
    all_peaks.extend(peaks_with_values)

# Optional: print shape to confirm
print(f"Combined array shape: {data_array.shape}")


# Double the resolution using bicubic interpolation
doubled_data = zoom(data_array, zoom=2, order=3)
# Smooth the data
smoothed_data = gaussian_filter(doubled_data, sigma=1)


# Prepare for sorting
coordinates_all = np.array([coord for coord, val in all_peaks])
values_all = [val for coord, val in all_peaks]
# Sort coordinates visually: top-to-bottom, then left-to-right
sorted_indices = np.lexsort((coordinates_all[:, 1], coordinates_all[:, 0]))  # (X, Y)
sorted_coordinates = coordinates_all[sorted_indices]
sorted_groups = [sorted_coordinates[i:i + 10] for i in range(0, len(sorted_coordinates), 10)]
sorted_array_list = [group[group[:, 1].argsort()] for group in sorted_groups]
combined_list = np.vstack(sorted_array_list)
sorted_peak_values = [data_array[row, col] for row, col in combined_list]

# Get 10 distinct colors from the 'tab10' colormap
cmap = plt.colormaps.get_cmap('tab10')

# Group visually sorted peaks into 10 groups of 10
groups = [sorted_peak_values[i:i + 10] for i in range(0, 300, 10)]
group_colors = [cmap(i) for i in range(10)]
#means = np.mean(groups, axis=1)

# Plot the peaks on the image
fig1, ax1 = plt.subplots(figsize=(12, 8))
im = ax1.imshow(data_array, cmap="viridis", aspect="equal", vmax=30000)

# Draw a reference line of 5 pixels (in data units)
x0, y0 = 10, 10  # starting point
x1, y1 = x0 + 1, y0  # 5 pixels to the right (horizontal line)
ax1.plot([x0, x1], [y0, y1], 'w-', lw=2, label='5-pixel line')
# Create color array for each peak based on its group
scatter_colors = [group_colors[i // 10] for i in range(100)]

# Plot peaks with assigned colors
# In scatter plot
#ax1.scatter(combined_list[:, 1], combined_list[:, 0], c=scatter_colors, s=20, label='Peaks')

# For annotations:
for i, (row, col) in enumerate(combined_list):
    ax1.text(col+5, row, str(i+1), color='white', fontsize=6, ha='left', va='top')

# For bar plots (loop as before, using `groups[i]` and `group_colors[i]`)

# Adjust width and spacing
shrink_factor = 0.7  # 80% of original width
actual_width = int(column_width * shrink_factor)
side_margin = (actual_width) // 3

for i in range(num_columns):
    x_start = i * column_width + side_margin
    rect = Rectangle(
        (x_start, 0),                     # (x, y)
        actual_width,                    # reduced width
        data_array.shape[0],             # height
        linewidth=1,
        edgecolor='red',
        facecolor='none'
    )
    ax1.add_patch(rect)


plt.colorbar(im, ax=ax1, label="Intensity")
ax1.set_title("Top 100 Intensity Peaks")
ax1.set_xlabel("X (columns)")
ax1.set_ylabel("Y (rows)")
ax1.legend()
fig1.tight_layout()

# Create 10 bar plots
fig2, axs = plt.subplots(2, 5, figsize=(15, 6))
axs = axs.flatten()

for i, ax in enumerate(axs):
    ax.bar(range(1, 11), groups[i])
    ax.set_title(f"Peaks {i*10+1}–{(i+1)*10}")
    ax.set_xlabel("Peak Index")
    ax.set_ylabel("Intensity")
    ax.set_ylim([0, max(peak_values) * 1.1])  # same scale for comparison


#fig3, ax3 = plt.subplots(figsize=(10, 6))
#for line in range(30):
#    ax3.plot(range(30),[row[line] for row in groups],label=f"{line} elements")
#    ax3.legend()

plt.tight_layout()
plt.suptitle("Intensity of Peaks (Grouped by 10s)", fontsize=16, y=1.02)
plt.show()

# Optional: print coordinates
#print("Top 100 peak coordinates (row, col):")
#print(coordinates)



# Print top 10 as a preview
print("Top 10 of 100 Peaks (row, col, value):")
for i, (coord, val) in enumerate(peaks_with_values[:10]):
    print(f"{i+1:2d}: {tuple(coord)} → {val:.3f}")

# Optional: Save to file
# np.savetxt("top_100_peaks.txt", np.column_stack((coordinates, peak_values)), 
#            header="row\tcol\tintensity", fmt="%d\t%d\t%.6f")

