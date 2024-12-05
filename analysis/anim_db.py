import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import ImageGrid

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
        partial(animate, coupled_idx, data=data, im=im, ax=ax),
        frames=data.shape[1],
        interval=100,
        blit=True,
    )
    out_name = os.path.join(out_dir, f"{name}_output.mp4")
    ani.save(out_name, writer="ffmpeg", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bruss")
    parser.add_argument("--dt", type=float, default=0)
    parser.add_argument("--run_id", type=str, default="")

    args = parser.parse_args()
    model = args.model
    dt = args.dt
    run_id = args.run_id

    load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    output_dir = os.getenv("OUT_DIR")
    output_dir = os.path.join(output_dir, model)
    os.makedirs(output_dir, exist_ok=True)

    df0 = get_db(os.path.join(data_dir, model))

    df = df0.copy()
    if run_id != "":
        df = df[df["run_id"] == run_id]
    if dt != 0:
        df = df[df["dt"] == dt]

    for i, row in df.iterrows():
        ds = nc.Dataset(row["filename"])
        data = ds.variables["data"][:]
        A, B = row["A"], row["B"]
        make_animation(data, f"{model}-{A}-{B}", output_dir, coupled_idx=1)
        print(f"created ({A},{B})")
if __name__ == "__main__":
    main()