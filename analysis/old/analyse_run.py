import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial

import os
import sys
import argparse

n_coupled = 2

def plot(data):
    global_min = np.min(data)
    global_max = np.max(data)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

    ims = []

    coupled_idx = 0
    # Display the first snapshot initially; this will be updated in the animation
    matrix = data[0, 0, :, coupled_idx::n_coupled]
    im = ax.imshow(
        matrix, cmap="viridis", aspect="equal", vmin=global_min, vmax=global_max
    )
    ims.append(im)
    # ax.set_xlabel("y")
    # ax.set_ylabel("x")

    # Add a colorbar
    # cbar = fig.colorbar(
    #     ims[0], ax=ax, orientation="vertical", fraction=0.02, pad=0.04
    # )
    # cbar.set_label("Data Value")
    return fig, ax, ims

def animate(snapshot, data, ims, ax):
    coupled_idx = 0
    matrix = data[0, snapshot, :, coupled_idx::n_coupled]
    im = ims[coupled_idx]
    im.set_array(matrix)  # Update data for each coupled component
    name = "u" if coupled_idx == 0 else "v"
    # ax.set_title(
    #     f"Snapshot {snapshot + 1}, {name}"
    # )

    return ims

def make_animation(data, param, out_dir):
    fig, axes, ims = plot(data)
    ani = animation.FuncAnimation(
        fig,
        partial(animate, data=data, ims=ims, axes=axes),
        frames=data.shape[1],
        interval=100,
        blit=True,
    )
    out_name = os.path.join(out_dir, f"{param}_output.mp4")
    ani.save(out_name, writer="ffmpeg", dpi=150)
    plt.close(fig)

def save_final_frame(data, param, out_dir):
    fig, axes, ims = plot(data)
    animate(data.shape[1] - 1, data, ims, axes)
    plt.savefig(os.path.join(out_dir, f"{param}_final.png"))
    plt.close()

def convergence_plot(data, param, out_dir, A, B):
    steady_state = np.zeros_like(data[0, 0, :, :])
    steady_state[:, 0::n_coupled] = A
    steady_state[:, 1::n_coupled] = B / A

    l1 = np.zeros((data.shape[1],2))

    for i in range(data.shape[1]):
        l1[i,0] = i
        l1[i,1] = np.linalg.norm(data[0,i,:,:] - steady_state, ord=1) / np.prod(data[0,i,:,:].shape)

    plt.plot(l1[:,0], l1[:,1], label=f"{param}")
    plt.xlabel("Iteration")
    plt.ylabel("|u - u*| + |v - v*|")
    plt.legend()
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--anim", action="store_true", default=False)
    parser.add_argument("--save_final", action="store_true", default=True)
    parser.add_argument("--conv", action="store_true", default=False)

    args = parser.parse_args()
    dir_path = args.data
    out_dir = args.out_dir
    anim = args.anim
    save_final = args.save_final
    conv = args.conv

    dir_name = os.path.basename(dir_path)
    # make out/dir_name if it does not exist
    out_dir = os.path.join(out_dir, dir_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    files = []
    for file in os.listdir(dir_path): 
        if file.endswith("_output.nc"):
            files.append(file)

    # Assuming format: run-name_param-name_param-value_[more params]_output.nc
    files = sorted(
        [(os.path.basename(f).split("_")[2], f) for f in files],
        key=lambda x: float(x[0])
    )

    for (param, f) in files:
        print(f"Processing {f}")
        d = nc.Dataset(os.path.join(dir_path, f))
        data = d["data"][:]
        if anim:
            make_animation(data, param, out_dir)
        if save_final:
            save_final_frame(data, param, out_dir)
    if conv:
        for (param, f) in files:

            if not f.startswith("bruss"):
                raise ValueError("Steady state convergence only supported for Brusselator")
            A = 5
            B = 9

            if f.split("_")[1] == "A":
                A = float(f.split("_")[2])
            if f.split("_")[3] == "B":
                B = float(f.split("_")[4])
            
            convergence_plot(data, param, out_dir, A, B)
        
        plt.savefig(os.path.join(out_dir, "convergence.png"), dpi=150)

if __name__ == "__main__":
    main()