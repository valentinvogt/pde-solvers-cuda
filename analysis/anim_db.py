import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.animation import FuncAnimation

import os
import glob
import pandas as pd
import json
from dotenv import load_dotenv
from functools import partial
import argparse


def get_db(data_dir):
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    data_list = []

    # Iterate through the JSON files and read them
    for file in json_files:
        with open(file, "r") as f:
            data = json.load(f)
            data_list.append(data)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data_list)
    return df


def plot(data, coupled_idx=0):
    global_min = np.min(data)
    global_max = np.max(data)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

    # Display the first snapshot initially; this will be updated in the animation
    matrix = data[0, 0, :, coupled_idx::2]
    im = ax.imshow(
        matrix, cmap="viridis", aspect="equal", vmin=global_min, vmax=global_max
    )
    return fig, ax, im


def animate(snapshot, coupled_idx, data, im, ax):
    matrix = data[0, snapshot, :, coupled_idx::2]
    im.set_array(matrix)  # Update data for each coupled component
    name = "u" if coupled_idx == 0 else "v"
    # ax.set_title(
    #     f"Snapshot {snapshot + 1}, {name}"
    # )
    return [im]


def make_animation(data, name, out_dir, coupled_idx):
    fig, ax, im = plot(data, coupled_idx)
    ani = animation.FuncAnimation(
        fig,
        partial(animate, coupled_idx=coupled_idx, data=data, im=im, ax=ax),
        frames=data.shape[1],
        interval=100,
        blit=True,
    )
    out_name = os.path.join(out_dir, f"{name}_output.mp4")
    ani.save(out_name, writer="ffmpeg", dpi=150)
    plt.close(fig)

def ab_grid_animation(
    df, component_idx=0, sigdigits=2, var1="A", var2="B", file="", fps=10, dpi=100
):
    if len(df) == 0:
        return None

    df = df.sort_values(by=[var1, var2])
    A_count = len(df[var1].unique())
    B_count = int(len(df) / A_count)

    fig = plt.figure(figsize=(15, 12))
    grid = ImageGrid(fig, 111, nrows_ncols=(A_count, B_count), axes_pad=(0.1, 0.3))
    ims = []

    # Preload data and calculate global min/max for normalization
    global_min, global_max = float("inf"), float("-inf")
    for i, row in df.iterrows():
        ds = nc.Dataset(row["filename"])
        data = ds.variables["data"][:]
        ims.append((row, data))
        global_min = min(global_min, data[0, :, :, component_idx::2].min())
        global_max = max(global_max, data[0, :, :, component_idx::2].max())

    # Normalization parameters for consistent color mapping
    norm = plt.Normalize(vmin=global_min, vmax=global_max)

    def update(frame):
        fig.clear()
        grid = ImageGrid(fig, 111, nrows_ncols=(A_count, B_count), axes_pad=(0.1, 0.3))

        for ax, (row, data) in zip(grid, ims):
            im = data[0, frame, :, component_idx::2]
            ax.set_title(
                f"{var1}={row[var1]:.{sigdigits}f}\n{var2} = {row[var2]:.{sigdigits}f}",
                fontsize=6,
            )
            ax.imshow(im, cmap="viridis", norm=norm)
            ax.set_aspect("equal")
            ax.axis("off")

        row = df.iloc[0]
        time = row["dt"] * frame * row["Nt"] / row["n_snapshots"]
        fig.suptitle(
            f"{row['model'].capitalize()}, Nx={row['Nx']}, dx={row['dx']}, dt={row['dt']}, T={time:.2f}",
            fontsize=16,
        )

    anim = FuncAnimation(fig, update, frames=range(df.iloc[0]["n_snapshots"]), interval=1000/fps)

    if file:
        anim.save(file, fps=fps, dpi=dpi)
    else:
        plt.show()

    return anim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bruss")
    parser.add_argument("--subdir", type=str, default="")
    parser.add_argument("--dt", type=float, default=0)
    parser.add_argument("--Nt", type=int, default=0)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--var1", type=str, default="A")
    parser.add_argument("--var2", type=str, default="B")
    parser.add_argument("--outfile", type=str, default="output.gif")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=150)
    
    args = parser.parse_args()
    model = args.model
    subdir = args.subdir
    dt = args.dt
    Nt = args.Nt
    run_id = args.run_id
    var1 = args.var1
    var2 = args.var2
    outfile = args.outfile
    fps = args.fps
    dpi = args.dpi

    load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    output_dir = os.getenv("OUT_DIR")
    output_dir = os.path.join(output_dir, model)
    os.makedirs(output_dir, exist_ok=True)

    if subdir != "":
        path = os.path.join(data_dir, model, subdir)
    else:
        path = os.path.join(data_dir, model)
    df0 = get_db(path)

    df = df0.copy()
    if run_id != "":
        df = df[df["run_id"] == run_id]
    if dt != 0:
        df = df[df["dt"] == dt]
    if Nt != 0:
        df = df[df["Nt"] == Nt]

    # for i, row in df.iterrows():
    #     ds = nc.Dataset(row["filename"])
    #     data = ds.variables["data"][:]
    #     A, B = row["A"], row["B"]
    #     make_animation(data, f"{model}-{A}-{B}", output_dir, coupled_idx=1)
    #     print(f"created ({A},{B})")

    # replace ending of outfile with _u.<ending> and _v.<ending>
    # should work for .gif and .mp4
    file_u = outfile.replace(".", "_u.")
    file_v = outfile.replace(".", "_v.")
    ab_grid_animation(df, 0, sigdigits=2, var1=var1, var2=var2, file=file_u, fps=fps, dpi=dpi)
    ab_grid_animation(df, 1, sigdigits=2, var1=var1, var2=var2, file=file_v, fps=fps, dpi=dpi)
        
if __name__ == "__main__":
    main()