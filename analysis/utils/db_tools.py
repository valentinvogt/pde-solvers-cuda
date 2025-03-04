import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.animation import FuncAnimation
from typing import Tuple
import plotly.graph_objects as go

import os
import glob
import pandas as pd
import json
from dotenv import load_dotenv
from functools import partial


def get_db_for_model(data_dir, model: str = "all") -> pd.DataFrame:
    """
    Creates a DataFrame from the JSON files in the specified directory.
    If model is 'all', all subdirectories of data_dir are used.
    Else, data_dir/model is used.
    """
    if model == "all":
        # Get all subdirectories
        subdirs = [x[0] for x in os.walk(data_dir)]
        subdirs = subdirs[1:]
        df_all = pd.DataFrame()
        for dir in subdirs:
            df = get_db(dir, dir.split("/")[-1])
            df_all = pd.concat([df_all, df])

        return df_all

    json_files = glob.glob(os.path.join(data_dir, model, "*.json"))
    data_list = []

    # Iterate through the JSON files and read them
    for file in json_files:
        with open(file, "r") as f:
            data = json.load(f)
            data_list.append(data)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data_list)
    return df


def get_db(data_dir) -> pd.DataFrame:
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    data_list = []

    # Iterate through the JSON files and read them
    for file in json_files:
        if os.path.basename(file).startswith("_"):
            continue
        with open(file, "r") as f:
            data = json.load(f)
            data_list.append(data)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data_list)
    return df


def check_data_dir(data_dir):
    """
    Asserts that all JSON files in data_dir point to an
    existing data file.
    """
    valid = True
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".json"):
                d = json.load(open(os.path.join(root, file)))
                if "filename" in d:
                    # check if file exists
                    if not os.path.isfile(d["filename"]):
                        print("File does not exist")
                        valid = False

    return valid


def filter_df(df, A=None, B=None, Du=None, Dv=None):
    """
    Filter by the provided parameters.
    """
    filter_criteria = {}
    if A is not None:
        filter_criteria["A"] = A
    if B is not None:
        filter_criteria["B"] = B
    if Du is not None:
        filter_criteria["Du"] = Du
    if Dv is not None:
        filter_criteria["Dv"] = Dv

    # Filter the dataframe based on provided parameters
    filtered_df = df
    for key, value in filter_criteria.items():
        filtered_df = filtered_df[filtered_df[key] == value]
    return filtered_df


def get_data(row):
    """
    Returns the data array associated with a row or
    a one-row df.
    """
    if isinstance(row, pd.DataFrame):
        if len(row) == 1:
            row = row.iloc[0]
        else:
            raise ValueError("row should be Series or single-row DataFrame")
    ds = nc.Dataset(row["filename"])
    data = ds.variables["data"][:]
    ds.close()
    return data


def plot(data, global_min, global_max):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"wspace": 0.4})
    ims = []
    for coupled_idx, ax in enumerate(axes):
        matrix = data[0, 0, :, 0::2]
        matrix /= np.max(matrix)
        im = ax.imshow(matrix, cmap="viridis", aspect="equal", vmin=0, vmax=1)
        ax.set_title(f"Snapshot 1, {'u' if coupled_idx == 0 else 'v'}")
        ims.append(im)
    return fig, axes, ims


def animate(snapshot, data, ims, axes):
    for coupled_idx, (ax, im) in enumerate(zip(axes, ims)):
        matrix = data[0, snapshot, :, coupled_idx::2]
        matrix /= matrix.max()  # Normalize
        im.set_array(matrix)
        name = "u" if coupled_idx == 0 else "v"
        ax.set_title(f"Snapshot {snapshot + 1}, {name}")
    return ims


def make_animation(data, name, out_dir):
    """
    Creates .gif animation of the data in the specified directory.
    """
    global_min = np.min(data)
    global_max = np.max(data)
    fig, axes, ims = plot(data, global_min, global_max)
    ani = animation.FuncAnimation(
        fig,
        partial(animate, data=data, ims=ims, axes=axes),
        frames=data.shape[1],
        interval=100,
        blit=True,
    )
    out_name = os.path.join(out_dir, f"{name}_output.gif")
    ani.save(out_name, writer="ffmpeg", dpi=150)
    plt.close(fig)


def plot_grid(
    df,
    component_idx=0,
    frame=-1,
    sigdigits=3,
    var1="A",
    var2="B",
    filename="",
):
    if len(df) == 0:
        return None

    if var1 == "":
        A_count = 1
        B_count = len(df)
    elif var2 == "":
        A_count = len(df[var1].unique())
        df = df.sort_values(by=[var1])
        df[var2] = 0
        B_count = A_count
        A_count = 1
    else:
        A_count = len(df[var1].unique())
        df = df.sort_values(by=[var1, var2])
        B_count = int(len(df) / A_count)

    fig = plt.figure(figsize=(15, 12))
    grid = ImageGrid(fig, 111, nrows_ncols=(A_count, B_count), axes_pad=(0.1, 0.3))
    ims = []

    for i, row in df.iterrows():
        data = get_data(row)
        f_min = data.min()
        f_max = data.max()
        ims.append((row, data[0, frame, :, component_idx::2], f_min, f_max))

    for ax, (row, im, f_min, f_max) in zip(grid, ims):
        label = f"{var1}={row[var1]:.{sigdigits}f}"
        if var1 == "":
            label = ""
        else:
            if isinstance(row[var1], float):
                label = f"{var1}={row[var1]:.{sigdigits}f}"
            else:
                label = f"{var1}={row[var1]}"
            if var2 != "":
                label += f"\n{var2} = {row[var2]:.{sigdigits}f}"
            ax.set_title(
                label,
                fontsize=6,
            )
        ax.imshow(im, cmap="viridis", vmin=f_min, vmax=f_max)
        ax.set_aspect("equal")
        ax.axis("off")

    row = df.iloc[0]
    if frame == -1:
        time = row["dt"] * row["Nt"]
    else:
        time = row["dt"] * frame * row["Nt"] / row["n_snapshots"]
    fig.suptitle(
        f"{row['model'].capitalize()}, Nx={row['Nx']}, dx={row['dx']}, dt={row['dt']}, T={time:.2f}",
        fontsize=16,
    )

    if filename == "":
        plt.show()
    else:
        plt.savefig(filename, dpi=100)
        plt.close()
    return grid


def compute_metrics(row, start_frame, end_frame=-1):
    """
    end_frame: works like Python slicing
    Returns deviations, time_derivatives, spatial_derivatives, relative std
    as 2D arrays of shape (num_frames, 2)
    """
    if end_frame < 0:
        end_frame = row["n_snapshots"] + end_frame
    
    num_frames = end_frame - start_frame + 1

    deviations = np.zeros((num_frames, 2))
    time_derivatives = np.zeros((num_frames, 2))
    spatial_derivatives = np.zeros((num_frames, 2))
    relative_stds = np.zeros((num_frames, 2))

    data = get_data(row)
    steady_state = np.zeros_like(data[0, 0, :, :])

    steady_state[:, 0::2] = row["A"]
    steady_state[:, 1::2] = row["B"] / row["A"]

    u = data[0, :, :, 0::2]
    v = data[0, :, :, 1::2]
    du_dt = np.gradient(u, row["dt"], axis=0)
    dv_dt = np.gradient(v, row["dt"], axis=0)

    for j in range(0, num_frames):
        snapshot = start_frame + j
        u_t = u[snapshot, :, :]
        v_t = v[snapshot, :, :]
        du_dx = np.gradient(u_t, row["dx"], axis=0)
        dv_dx = np.gradient(v_t, row["dx"], axis=0)
        deviations[j, 0] = np.linalg.norm(u_t - steady_state[:, 0::2])
        deviations[j, 1] = np.linalg.norm(v_t - steady_state[:, 1::2])
        time_derivatives[j, 0] = np.linalg.norm(du_dt[snapshot])
        time_derivatives[j, 1] = np.linalg.norm(dv_dt[snapshot])
        spatial_derivatives[j, 0] = np.linalg.norm(du_dx)
        spatial_derivatives[j, 1] = np.linalg.norm(dv_dx)
        relative_stds[j, 0] = np.std(u_t) / np.mean(u_t)
        relative_stds[j, 1] = np.std(v_t) / np.mean(v_t)

    return deviations, time_derivatives, spatial_derivatives, relative_stds


def metrics_grid(
    df,
    start_frame,
    sigdigits=3,
    joint=False,
    var1="A",
    var2="B",
    metric="dev",
    filename="",
    show_title=True,
    scale=1
):
    if metric == "dev":
        text = "Deviation ||u(t) - u*||"
    elif metric == "dx":
        text = "Spatial Derivative ||∇u(t)||"
    elif metric == "dt":
        text = "Time Derivative ||du/dt||"
    elif metric == "std":
        text = "Relative Standard Deviation"
    else:
        raise ValueError("metric must be 'dev', 'dx', or 'dt'")

    if len(df) == 0:
        return None

    if var1 == "":
        A_count = 1
        B_count = len(df)
    elif var2 == "":
        A_count = len(df[var1].unique())
        df = df.sort_values(by=[var1])
        df[var2] = 0
        B_count = A_count
        A_count = 1
    else:
        A_count = len(df[var1].unique())
        df = df.sort_values(by=[var1, var2])
        B_count = int(len(df) / A_count)

    df = df.reset_index(drop=True)
    fig, axes = plt.subplots(A_count, B_count, figsize=(scale * 3 * B_count + 1, scale * 5 * A_count))

    axes = np.atleast_2d(axes)

    for i, row in df.iterrows():
        data = get_data(row)
        steady_state = np.zeros_like(data[0, 0, :, :])

        steady_state[:, 0::2] = row["A"]
        steady_state[:, 1::2] = row["B"] / row["A"]

        metrics = compute_metrics(row, start_frame)
        if metric == "dev":
            values = metrics[0]
        elif metric == "dt":
            values = metrics[1]
        elif metric == "dx":
            values = metrics[2]
        elif metric == "std":
            values = metrics[3]
            
        row_idx = i // B_count if B_count > 1 else i
        col_idx = i % B_count if B_count > 1 else 0

        if not joint:
            axes[row_idx, col_idx].plot(
                np.arange(start_frame, row["n_snapshots"])
                * row["dt"]
                * row["Nt"]
                / row["n_snapshots"],
                values[:, 0],
                label="u",
            )
            axes[row_idx, col_idx].plot(
                np.arange(start_frame, row["n_snapshots"])
                * row["dt"]
                * row["Nt"]
                / row["n_snapshots"],
                values[:, 1],
                label="v",
            )
            if scale >= 1:
                axes[row_idx, col_idx].legend()
        else:
            values = np.linalg.norm(values, axis=1)
            axes[row_idx, col_idx].plot(
                np.arange(start_frame, row["n_snapshots"])
                * row["dt"]
                * row["Nt"]
                / row["n_snapshots"],
                values[:],
            )
            if var1 == "":
                label = ""
            else:
                if isinstance(row[var1], float):
                    label = f"{var1}={row[var1]:.{sigdigits}f}"
                else:
                    label = f"{var1}={row[var1]}"
                if var2 != "":
                    label += f"\n{var2} = {row[var2]:.{sigdigits}f}"
                axes[row_idx, col_idx].set_title(
                    label,
                    fontsize=6,
                )
        # axes[row_idx, col_idx].axis("off")

    row = df.iloc[0]
    time = row["dt"] * row["Nt"]
    if show_title:
        fig.suptitle(
            f"{row['model'].capitalize()}, Nx={row['Nx']}, dx={row['dx']}, dt={row['dt']}, T={time:.2f}, {text}",
            fontsize=4 * scale * B_count,
        )

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    if filename == "":
        plt.show()
    else:
        plt.savefig(filename, dpi=100)
        plt.close()

    return axes


def delete_run(df, run_id) -> pd.DataFrame:
    for i, row in df.iterrows():
        if row["run_id"] == run_id:
            filename = row["filename"]
            df.drop(i, inplace=True)
            os.remove(filename)
            os.remove(filename.replace("_output.nc", ".json"))
            os.remove(filename.replace("_output.nc", ".nc"))

    return df


def get_metrics_array(df, start_frame=0, metric="dev"):
    """
    Returns:
    all_metrics: num_trajectories x n_snapshots array of metric values
    title: for plotting
    """
    title = ""
    if metric not in ["dev", "dt", "dx", "std"]:
        raise ValueError("Not a valid metric!")

    all_metrics = []
    for _, row in df.iterrows():
        d = get_data(row)
        metrics = compute_metrics(row, start_frame=start_frame)
        if metric == "dev":
            title = "Deviation"
            values = metrics[0]
        elif metric == "dt":
            title = "Time Derivative"
            values = metrics[1]
        elif metric == "dx":
            title = "Spatial Derivative"
            values = metrics[2]
        elif metric == "std":
            title = "Relative std"
            values = metrics[3]
        all_metrics.append(values)
    all_metrics = np.array(all_metrics)
    return all_metrics, title


def plot_ball_behavior(df, start_frame=0, metric="dev", joint=False, fig=None, label=None):
    """
    Plot the mean and mean + std of the given metric,
    as well as the trajectory with the minimum final value.
    joint: whether to average u and v to get a single time series
    fig: optional, if several plots are to be combined
    label: if there are several plots, identify which is which using this
    Returns a Plotly figure.
    """
    
    all_metrics, title = get_metrics_array(df, start_frame=start_frame, metric=metric)
    all_metrics = np.array(all_metrics)
    row = df.iloc[0]
    dt = row["dt"] * row["Nt"] / row["n_snapshots"]
    t = np.linspace(start_frame * dt, row["n_snapshots"] * dt, row["n_snapshots"] - start_frame)
    # Compute mean and std
    avg_metric = np.mean(all_metrics, axis=0)
    std_metric = np.std(all_metrics, axis=0)

    ids = ["u", "v"]
    traj_count = 2
    if joint:
        avg_metric_uv = avg_metric
        avg_metric = np.mean(avg_metric_uv, axis=1)
        std_metric = np.linalg.norm(avg_metric_uv, axis=1)
        ids = ["u+v"]
        traj_count = 1

    for j in range(traj_count):
        id = ids[j]
        min_idx = np.argmin(all_metrics[:, -1, j])
        min_row = all_metrics[min_idx, :, j]

        avg_metric_loc = avg_metric[:, j]
        std_metric_loc = std_metric[:, j]
        # Create figure
        if fig is None:
            fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([t, t[::-1]]),
                y=np.concatenate([avg_metric_loc + std_metric_loc, (avg_metric_loc)[::-1]]),
                fill="toself",
                fillcolor="rgba(0,100,80,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
            )
        )

        text_avg = title
        text_std = f"Min({title})"
        if label is not None:
            text_avg += f"({label})"
            text_std += f"({label})"
        text_avg += f", {id}"
        text_std += f", {id}"

        # Add mean line
        fig.add_trace(
            go.Scatter(
                x=t,
                y=avg_metric_loc,
                mode="lines",
                name=text_avg,
                hovertemplate="Index: %{x}<br>Deviation: %{y:.2f}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=t,
                y=min_row,
                mode="lines",
                name=text_std,
                hovertemplate="Index: %{x}<br>Min: %{y:.2f}<extra></extra>",
            )
        )

    # Update layout
    fig.update_layout(
        title="Deviation Metrics",
        xaxis_title="Time Step/Index",
        yaxis_title="Deviation Value",
        hovermode="x unified",
        showlegend=True,
        template="plotly_white",
    )

    return fig



def plot_all_trajectories(df, start_frame=0, metric="dev"):
    t = np.linspace(0, 100, 100)
    title = ""

    # Create figure
    fig = go.Figure()

    all_metrics, title = get_metrics_array(df, start_frame, metric)

    for i, values in enumerate(all_metrics):
        # Add a trace for each row's metric values
        fig.add_trace(
            go.Scatter(
                x=t,
                y=values,
                mode="lines",
                name=f"Row {i}",  # Use row index or a unique identifier
                hovertemplate="Index: %{x}<br>Value: %{y:.2f}<extra></extra>",
            )
        )

    # Update layout
    fig.update_layout(
        title=f"{title} Metrics for All Rows",
        xaxis_title="Time Step/Index",
        yaxis_title=f"{title} Value",
        hovermode="x unified",
        showlegend=True,
        template="plotly_white",
    )

    # Add range slider for better interactivity
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="linear"))

    fig.show()


def db_init(model, run_id, use_class_df=True, class_df="classification_metrics.json"):
    """
    Returns a df and the output directory
    """
    load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    output_dir = os.getenv("OUT_DIR")
    output_dir = os.path.join(output_dir, model, run_id)
    
    if not use_class_df:
        os.makedirs(output_dir, exist_ok=True)
        df = get_db(os.path.join(data_dir, model, run_id))
    else:
        db_file = os.path.join(output_dir, class_df)
        df = pd.read_json(db_file, orient='records', lines=True)
    
    for _, row in df.iterrows():
        if not os.path.isfile(row["filename"]):
            raise FileNotFoundError(f"File {row['filename']} does not exist.")
    
    return df, output_dir

def add_op_params_to_df(df):
    assert "original_point" in df.columns
    df["op_A"] = df["original_point"].apply(lambda x: x.get("A"))
    df["op_B"] = df["original_point"].apply(lambda x: x.get("B"))
    df["op_Du"] = df["original_point"].apply(lambda x: x.get("Du"))
    df["op_Dv"] = df["original_point"].apply(lambda x: x.get("Dv"))
    
    return df