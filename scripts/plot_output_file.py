
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import os

# Open the NetCDF file
filename = input('enter filename: ')

file_path = 'data/' + filename
dataset = nc.Dataset(file_path, mode='r')

# Extract dimensions
n_members = dataset.dimensions['n_members'].size
n_snapshots = dataset.dimensions['n_snapshots'].size
x_size_and_boundary = dataset.dimensions['x_size_and_boundary'].size
n_coupled_and_y_size_and_boundary = dataset.dimensions['n_coupled_and_x_size_and_boundary'].size
n_coupled = dataset.getncattr('n_coupled')

# Extract the data variable
data = dataset.variables['data'][:]

global_min = np.min(data)
global_max = np.max(data)
print(global_min)
print(global_max)



# Create a grid of subplots
fig, axes = plt.subplots(nrows=n_members*n_coupled, ncols=n_snapshots, figsize=(15, 15))
axes = axes.flatten()  # Flatten to make it easier to iterate

for member in range(n_members):
    for snapshot in range(n_snapshots):
        for coupled_idx in range(n_coupled):
            # Calculate the subplot index
            ax_index = member * n_coupled * n_snapshots + coupled_idx * n_snapshots + snapshot
            ax = axes[ax_index]

            # Extract the 2D matrix for the current member, snapshot, and coupled index
            matrix = data[member, snapshot, :, coupled_idx::n_coupled]

            # Plot the matrix
            im = ax.imshow(matrix, cmap='viridis', aspect='equal', vmin=global_min, vmax=global_max)
            ax.set_title(f'Member {member + 1}, Coupled Index {coupled_idx + 1}, Snapshot {snapshot + 1}')
            ax.set_xlabel('n_coupled_and_x_size_and_boundary')
            ax.set_ylabel('x_size_and_boundary')

# Add a colorbar to the figure
cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Data Value')

# Adjust layout to prevent overlap
plt.savefig('out/'+ os.path.splitext(filename)[0] + "_" + str(member) + '.svg')  # Save the figure to a file
plt.show()

# Close the dataset
dataset.close()
