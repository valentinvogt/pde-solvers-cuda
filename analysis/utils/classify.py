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

from utils.db_tools import get_db, classify_trajectories

model = "bruss"
run_id = "abd_big"
load_dotenv()
data_dir = os.getenv("DATA_DIR")
output_dir = os.getenv("OUT_DIR")
output_dir = os.path.join(output_dir, model, run_id)
os.makedirs(output_dir, exist_ok=True)
df0 = get_db(os.path.join(data_dir, model, run_id))

df = df0.copy()
df = df[df['run_id'] == run_id]

def compute_classification_metrics(df, start_frame=0, end_frame=-1) -> pd.DataFrame:
    """
    Compute classification metrics for a given range of frames.
    """
    # Filter the dataframe to only include the relevant frames
    if len(df) == 0:
        return None

    for i, row in df.iterrows():
        num_snapshots = row["n_snapshots"]
        assert (
            start_frame < num_snapshots
        ), "start_frame must be less than num_snapshots"
        if end_frame == -1:
            end_frame = num_snapshots

        ds = nc.Dataset(row["filename"])
        data = ds.variables["data"][:]  # Assume shape [time, spatial, ...]
        steady_state = np.zeros_like(data[0, 0, :, :])
        steady_state[:, 0::2] = row["A"]  # u = A
        steady_state[:, 1::2] = row["B"] / row["A"]  # v = B / A

        deviations = []
        time_derivatives = []

        du_dt = np.gradient(
            data[0, :, :, :], row["dt"], axis=0
        )  # Time derivative of (u, v)

        for j in range(start_frame, min(end_frame, num_snapshots)):
            deviations.append(np.linalg.norm(data[0, j, :, :] - steady_state))
            time_derivatives.append(np.linalg.norm(du_dt[j]))

        final_dev = deviations[-1]
        mean_dev = np.mean(deviations)
        std_dev = np.std(time_derivatives)
        max_derivative = np.max(time_derivatives)
        df.loc[i, "final_deviation"] = final_dev
        df.loc[i, "mean_deviation"] = mean_dev
        df.loc[i, "std_deviation"] = std_dev
        df.loc[i, "max_derivative"] = max_derivative
        ds.close()
    return df

df_class = compute_classification_metrics(df, 80)
df_class.to_csv(os.path.join(output_dir, "classif_metrics.csv"))
