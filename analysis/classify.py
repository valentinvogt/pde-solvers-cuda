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

from db_tools import get_db, classify_trajectories

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

df_class = classify_trajectories(df, 50, steady_threshold=1.25, osc_threshold=1, dev_threshold=1)
df_class.to_csv(os.path.join(output_dir, "classified_2.csv"))
