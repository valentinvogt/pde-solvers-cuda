import numpy as np
import os
import sys
from dotenv import load_dotenv
from functools import partial
from uuid import uuid4
import pandas as pd

from create_netcdf_input import create_input_file
from helpers import f_scalings, zero_func, const_sigma, create_json


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

"""
Usage:
python rd_script.py model A B Nx dx Nt dt Du Dv n_snapshots

args:
model: str = "bruss", "gray_scott", "fhn"
A: float = 5
B: float = 9
Nx: int = 100
dx: float = 1.0
Nt: int = 1000
dt: float = 0.01
Du: float = 2.0
Dv: float = 22.0
sigma_ic: float = 0.1
random_seed: int = 1
n_snapshots: int = 100
filename: str = "data/bruss.nc"
run_id: str = ""
"""


def initial_sparse_sources(member, coupled_idx, x_position, y_position, sparsity):

    if coupled_idx == 0:
        u = np.ones(x_position.shape)
    elif coupled_idx == 1:
        u = np.zeros(x_position.shape)
        for i in range(0, sparsity * x_position.shape[0]):
            i = np.random.randint(0, x_position.shape[0])
            j = np.random.randint(0, x_position.shape[1])
            u[i, j] = 1.0
    else:
        print("initial_noisy_function is only meant for n_coupled == 2!")
        u = 0.0 * x_position

    return u


def steady_state_plus_noise(member, coupled_idx, x_position, y_position, params, ic_type, ic_param):
    A = params[0]
    B = params[1]
    if coupled_idx == 0:
        u = A * np.ones(shape=x_position.shape) + np.random.normal(
            0.0, ic_param[0], size=x_position.shape
        )
    elif coupled_idx == 1:
        u = (B / A) * np.ones(shape=x_position.shape) + np.random.normal(
            0.0, ic_param[1], size=x_position.shape
        )
    else:
        print("initial_noisy_function is only meant for n_coupled == 2!")
        u = 0.0 * x_position
    return u


def run_wrapper(
    model,
    A, B,
    Nx, dx,
    Nt, dt,
    Du, Dv,
    ic_type, ic_param,
    random_seed,
    n_snapshots,
    filename,
    run_id,
):
    np.random.seed(random_seed)

    fn_order = 4 if model == "fhn" else 3
    fn_scalings = f_scalings(model, A, B)
    input_filename = filename

    output_filename = filename.replace(".nc", "_output.nc")

    if model == "bruss":
        initial_condition = partial(steady_state_plus_noise, params=(A, B), ic_type=ic_type, ic_param=ic_param)
    elif model == "gray_scott":
        initial_condition = partial(initial_sparse_sources, sparsity=0.2)
    else:
        initial_condition = partial(
            steady_state_plus_noise, params=(A, B), ic_type=ic_type, ic_param=ic_param
        )

    create_input_file(
        input_filename,
        output_filename,
        type_of_equation=2,
        x_size=Nx,
        x_length=Nx * dx,
        y_size=Nx,
        y_length=Nx * dx,
        boundary_value_type=2,
        scalar_type=0,
        n_coupled=2,
        coupled_function_order=fn_order,
        number_timesteps=Nt,
        final_time=Nt * dt,
        number_snapshots=n_snapshots,
        n_members=1,
        initial_value_function=initial_condition,
        sigma_function=const_sigma,
        bc_neumann_function=zero_func,
        f_value_function=fn_scalings,
        Du=Du,
        Dv=Dv,
    )

    print(input_filename)

    create_json(
        {
            "model": model,
            "A": A,
            "B": B,
            "Nx": Nx,
            "dx": dx,
            "Nt": Nt,
            "dt": dt,
            "Du": Du,
            "Dv": Dv,
            "ic_type": ic_type,
            "ic_param": ic_param,
            "random_seed": random_seed,
            "n_snapshots": n_snapshots,
            "filename": output_filename,
            "run_id": run_id,
        },
        filename.replace(".nc", ".json"),
    )


def sample_ball(A, B, Du, Dv, sigma, num_samples, sim_params):
    Nx=sim_params["Nx"]
    dx=sim_params["dx"]
    Nt=sim_params["Nt"]
    dt=sim_params["dt"]
    path = sim_params["path"]

    for i in range(num_samples):
        A_new = A + np.random.normal(0, sigma[0])
        B_new = B + np.random.normal(0, sigma[1])
        Du_new = Du + np.random.normal(0, sigma[2])
        Dv_new = Dv + np.random.normal(0, sigma[3])

        run_wrapper(
            "bruss",
            A_new, B_new,
            Nx, dx,
            Nt, dt,
            Du_new, Dv_new,
            "normal", [0.1, 0.1],
            random_seed=i,
            n_snapshots=100,
            filename=os.path.join(path, f"{uuid4()}.nc"),
            run_id="ball_sampling_local",
        )

if __name__ == "__main__":
    model = "bruss"
    run_id = "ball_sampling"
    load_dotenv()

    data_dir = os.getenv("DATA_DIR")
    path = os.path.join(data_dir, model, run_id)
    os.makedirs(path, exist_ok=True)

    center_df = pd.read_csv("data/sampling_centers.csv")
    sigma = center_df[["A", "B", "Du", "Dv"]].std() * 0.1

    sim_params = {
        "Nx": 128,
        "dx": 1.0,
        "Nt": 60_000,
        "dt": 0.0025,
        "path": path
    }

    for i, row in center_df.iterrows():
        A = row["A"]
        B = row["B"]
        Du = row["Du"]
        Dv = row["Dv"]

        sample_ball(A, B, Du, Dv, sigma, 50, sim_params)