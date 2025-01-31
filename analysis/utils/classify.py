import netCDF4 as nc
import numpy as np

import os
import pandas as pd
import argparse
from dotenv import load_dotenv
from scipy.fft import fft, fftfreq

from db_tools import get_db


def compute_classification_metrics(
    df, time_ratio=0.1, start_frame=0, end_frame=-1
) -> pd.DataFrame:
    """
    Compute classification metrics for a given range of frames, including
    all metrics used in the detailed classifier.
    """
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
        A, B = row["A"], row["B"]
        u_ss, v_ss = A, B / A
        steady_state = np.zeros_like(data[0, 0, :, :])
        steady_state[:, 0::2] = u_ss  # u steady state
        steady_state[:, 1::2] = v_ss  # v steady state

        time_steps = data.shape[1]
        last_stretch_start = int(time_steps * time_ratio)

        u = data[0, :, :, 0::2]
        v = data[0, :, :, 1::2]

        mean_dev_u = np.mean(np.abs(u - u_ss), axis=(1, 2))
        mean_dev_v = np.mean(np.abs(v - v_ss), axis=(1, 2))
        total_dev = mean_dev_u + mean_dev_v
        last_dev = total_dev[-last_stretch_start:]

        du = np.diff(u, axis=0)
        dv = np.diff(v, axis=0)
        deriv_norm = np.linalg.norm(du, axis=(1, 2)) + np.linalg.norm(dv, axis=(1, 2))
        last_deriv = deriv_norm[-last_stretch_start:]

        u_avg = np.mean(u, axis=(1, 2))
        fft_u = np.abs(fft(u_avg - u_ss)) / len(u_avg)
        fft_u[0] = 0  # Ignore DC component

        df.loc[i, "final_deviation"] = total_dev[-1]
        df.loc[i, "mean_deviation"] = np.mean(last_dev)
        df.loc[i, "std_deviation"] = np.std(last_dev)
        df.loc[i, "max_derivative"] = np.max(deriv_norm)
        df.loc[i, "mean_derivative"] = np.mean(last_deriv)
        df.loc[i, "dominant_power"] = np.max(fft_u)
        df.loc[i, "total_power"] = np.sum(fft_u)

        ds.close()

    return df


def classify_trajectories(
    df,
    detailed=False,
    dev_threshold=1e-2,
    steady_threshold=1e-3,
    osc_threshold=1e-2,
) -> pd.DataFrame:
    """
    Classify runs based on precomputed metrics.

    Args:
        df: DataFrame containing run metadata and precomputed metrics.
        detailed: If True, use the detailed classification scheme.
        dev_threshold: Threshold for deviation from the steady state.
        steady_threshold: Threshold for near-zero time derivatives.
        osc_threshold: Threshold for oscillatory behavior.

    Returns:
        Updated DataFrame with classification labels.
    """

    if len(df) == 0:
        return None

    classifications = []
    for i, row in df.iterrows():
        final_dev = row["final_deviation"]
        mean_dev = row["mean_deviation"]
        std_dev = row["std_deviation"]
        max_derivative = row["max_derivative"]

        if detailed:
            if max(row["max_u"], row["max_v"]) > 1e6:
                category = "BU"  # Blowup
            elif mean_dev < 1e-3 and np.all(np.diff(row["last_dev"]) <= 1e-3):
                category = "SS"  # Steady state
            elif row["mean_derivative"] < 1e-3 and mean_dev >= 1e-3 and std_dev < 1e-3:
                category = "DS"  # Different steady state
            elif row["dominant_power"] > osc_threshold * row["total_power"]:
                category = "OSC"  # Oscillatory
            else:
                category = "ND"  # No clear classification
        else:
            if final_dev < dev_threshold or (
                final_dev < 5 * dev_threshold and max_derivative < steady_threshold
            ):
                category = "SS"  # Steady state
            elif std_dev > osc_threshold or mean_dev > dev_threshold:
                category = "ND"  # No clear classification / interesting behavior
            else:
                category = "BU"  # Blowup / divergent

        classifications.append(category)

    df["category"] = classifications
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bruss")
    parser.add_argument("--run_id", default="")
    parser.add_argument("--outfile", default="")
    parser.add_argument("--time_ratio", default=0.1)

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
