import netCDF4 as nc
import numpy as np
import pandas as pd
import json
import os
import glob

def get_db(data_dir, model: str = "all") -> pd.DataFrame:
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


def convergence_plot(row):
    """
    Plots the convergence of the model.
    """
    ds = nc.Dataset(row["filename"])
