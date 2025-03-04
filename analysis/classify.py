import netCDF4 as nc
import numpy as np
from numpy.linalg import norm

import os
import pandas as pd
import argparse
from dotenv import load_dotenv
from scipy.fft import fft

from utils.db_tools import get_db, get_data


def compute_classification_metrics(
    df, time_ratio=0.1
) -> pd.DataFrame:
    """
    Compute classification metrics for a given range of frames
    """
    if len(df) == 0:
        return None

    for i, row in df.iterrows():
        num_snapshots = row["n_snapshots"]
        data = get_data(row)

        if np.any(data.mask):
            df.drop(i, inplace=True)
            continue

        A, B = row["A"], row["B"]
        u_ss, v_ss = A, B / A
        steady_state = np.zeros_like(data[0, 0, :, :])
        steady_state[:, 0::2] = u_ss  # u steady state
        steady_state[:, 1::2] = v_ss  # v steady state

        starting_idx = int(num_snapshots * time_ratio)
        u = data[0, :, :, 0::2]
        v = data[0, :, :, 1::2]

        # Compute max_u and max_v
        max_u = np.max(u)
        max_v = np.max(v)

        mean_dev_u = norm(u - u_ss, axis=(1, 2))
        mean_dev_v = norm(v - v_ss, axis=(1, 2))
        total_dev = mean_dev_u + mean_dev_v
        deviation = total_dev[-starting_idx:]
        du = np.diff(u, axis=1)
        dv = np.diff(v, axis=2)
        dx_norm = norm(du, axis=(1, 2)) + norm(dv, axis=(1, 2))
        last_dx = dx_norm[-starting_idx:]

        du_dt = np.gradient(u, row["dt"], axis=0)
        dv_dt = np.gradient(v, row["dt"], axis=0)
        dt_norm = norm(du_dt, axis=(1, 2)) + norm(dv_dt, axis=(1, 2))
        last_dt = dt_norm[-starting_idx:]

        u_avg = np.mean(u, axis=(1, 2))
        fft_u = np.abs(fft(u_avg - u_ss)) / len(u_avg)
        fft_u[0] = 0  # Ignore DC component

        rel_std_u = np.std(u, axis=(1, 2)) / np.mean(u, axis=(1, 2))
        rel_std_v = np.std(v, axis=(1, 2)) / np.mean(v, axis=(1, 2))
        rel_std_u_mean = np.mean(rel_std_u[-starting_idx:])
        rel_std_v_mean = np.mean(rel_std_v[-starting_idx:])

        # Store computed metrics in the DataFrame
        df.at[i, "mean_deviation"] = np.mean(deviation)
        df.at[i, "std_deviation"] = np.std(deviation)

        df.at[i, "max_dx"] = np.max(last_dx)
        df.at[i, "mean_dx"] = np.mean(last_dx)
        df.at[i, "max_dt"] = np.max(last_dt)
        df.at[i, "mean_dt"] = np.mean(last_dt)
        df.at[i, "dominant_power"] = np.max(fft_u)
        df.at[i, "total_power"] = np.sum(fft_u)
        df.at[i, "max_u"] = max_u
        df.at[i, "max_v"] = max_v
        df.at[i, "rel_std_u"] = rel_std_u_mean
        df.at[i, "rel_std_v"] = rel_std_v_mean

    return df


def classify_trajectories(
    df,
    deviation_threshold=1e-2,
    dt_threshold=50,
    osc_power_threshold=5e-2,
) -> pd.DataFrame:
    """
    Classify runs based on precomputed metrics.

    Args:
        df: DataFrame containing run metadata and precomputed metrics.
        detailed: If True, use the detailed classification scheme.
        deviation_threshold: Threshold for mean deviation from the steady state.
        dt_threshold: Threshold for near-zero time derivatives.
        osc_power_threshold: Threshold of the dominant frequency for oscillatory behavior.

    Returns:
        Updated DataFrame with classification labels.
    """

    if len(df) == 0:
        return None

    classifications = []
    for i, row in df.iterrows():
        mean_dev = row["mean_deviation"]
        mean_dt = row["mean_dt"]
        dom_power = row["dominant_power"]

        if mean_dev < deviation_threshold:
            category = "SS"  # Steady state
        elif mean_dt < dt_threshold:
            category = "DSS"  # Different steady state
        elif dom_power > osc_power_threshold:
            category = "OSC"  # Oscillatory
        else:
            category = "INT"  # Other, "interesting" behavior

        classifications.append(category)

    df["category"] = classifications
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bruss")
    parser.add_argument("--run_id", default="")
    parser.add_argument("--outfile", default="")
    parser.add_argument("--time_ratio", default=0.1, type=float)

    args = parser.parse_args()
    model = args.model
    run_id = args.run_id
    outfile = args.outfile
    time_ratio = args.time_ratio

    load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    output_dir = os.getenv("OUT_DIR")

    output_dir = os.path.join(output_dir, model, run_id)
    os.makedirs(output_dir, exist_ok=True)
    output_location = os.path.join(output_dir, outfile)
    df0 = get_db(os.path.join(data_dir, model, run_id))

    df = df0.copy()
    df = df[df["run_id"] == run_id]

    df_class = compute_classification_metrics(df, time_ratio=time_ratio)
    df_class.to_csv(output_location)
