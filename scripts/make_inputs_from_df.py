import numpy as np
import os
import sys
from dotenv import load_dotenv
from functools import partial
from uuid import uuid4
import pandas as pd

from create_netcdf_input import create_input_file
from helpers import f_scalings, zero_func, const_sigma, create_json



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


def steady_state_plus_noise(
    member, coupled_idx, x_position, y_position, params, ic_type, ic_param, ic_seed=None
):
    rng = np.random.default_rng(ic_seed)
    A = params[0]
    B = params[1]
    steady_state = A if coupled_idx == 0 else B / A
    u = steady_state * np.ones(shape=x_position.shape)
    v = steady_state * np.ones(shape=x_position.shape)
    if ic_type == "normal":
        u += rng.normal(
                0.0, ic_param["sigma_u"]
            )
        v += rng.normal(
                0.0, ic_param["sigma_v"]
            )
    elif ic_type == "uniform":
        u += rng.uniform(
            ic_param["u_min"], ic_param["u_max"], size=x_position.shape
        )
        v += rng.uniform(
            ic_param["v_min"], ic_param["v_max"], size=x_position.shape
        )
    elif ic_type == "hex_pattern":
        u += rng.normal(1.0, 0.5) * ic_param["amplitude"] * (
            np.cos(2 * np.pi * x_position / ic_param["wavelength"]) +
            np.sin(2 * np.pi * y_position / ic_param["wavelength"]) +
            np.cos(2 * np.pi * (x_position + y_position) / ic_param["wavelength"])
        )
        v += rng.normal(1.0, 0.5) * ic_param["amplitude"] * (
            np.cos(2 * np.pi * x_position / ic_param["wavelength"]) +
            np.sin(2 * np.pi * y_position / ic_param["wavelength"]) +
            np.cos(2 * np.pi * (x_position + y_position) / ic_param["wavelength"])
        )
    else:
        print("initial_noisy_function is only meant for n_coupled == 2!")
        u = 0.0 * x_position
    return u if coupled_idx == 0 else v


def run_wrapper(
    model,
    A, B,
    Nx, dx,
    Nt, dt,
    Du, Dv,
    ic_type,
    ic_param,
    random_seed,
    n_snapshots,
    filename,
    run_id,
    original_point,
):
    fn_order = 4 if model == "fhn" else 3
    fn_scalings = f_scalings(model, A, B)
    input_filename = filename

    output_filename = filename.replace(".nc", "_output.nc")

    if model == "bruss":
        initial_condition = partial(
            steady_state_plus_noise, params=(A, B), ic_type=ic_type, ic_param=ic_param
        )
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
            "original_point": original_point,
        },
        filename.replace(".nc", ".json"),
    )


def sample_ball(A, B, Du, Dv, sigma_sampling, samples_per_ic, sim_params, initial_conditions, run_id):
    Nx = sim_params["Nx"]
    dx = sim_params["dx"]
    Nt = sim_params["Nt"]
    dt = sim_params["dt"]
    path = sim_params["path"]
    sigma_A = sigma_sampling["A"] * A
    sigma_B = sigma_sampling["B"] * B
    sigma_Du = sigma_sampling["Du"] * Du
    sigma_Dv = sigma_sampling["Dv"] * Dv

    for _ in range(samples_per_ic):
        A_new = A + np.random.uniform(-sigma_A, sigma_A)
        B_new = B + np.random.uniform(-sigma_B, sigma_B)
        Du_new = Du + np.random.uniform(-sigma_Du, sigma_Du)
        Dv_new = Dv + np.random.uniform(-sigma_Dv, sigma_Dv)

        for ic in initial_conditions:
            run_wrapper(
                "bruss",
                A_new,
                B_new,
                Nx,
                dx,
                Nt,
                dt,
                Du_new,
                Dv_new,
                ic["type"],
                ic["params"],
                random_seed=np.random.randint(0, 1000000),
                n_snapshots=100,
                filename=os.path.join(path, f"{uuid4()}.nc"),
                run_id=run_id,
                original_point={"A": A, "B": B, "Du": Du, "Dv": Dv},
            )


if __name__ == "__main__":
    model = "bruss"
    run_id = "ball_test"
    initial_conditions = [
        {"type": "normal", "sigma_u": 0.1, "sigma_v": 0.1},
        {"type": "normal", "sigma_u": 0.25, "sigma_v": 0.25},
        {"type": "uniform", "u_min": 0.0, "u_max": 0.25, "v_min": 0.0, "v_max": 0.25},
        {"type": "hex_pattern", "amplitude": 0.25, "wavelength": 0.1},
    ]
    load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    path = os.path.join(data_dir, model, run_id)
    os.makedirs(path, exist_ok=True)

    center_df = pd.read_csv("data/sampling_centers.csv")
    sigma = {key: 0.02 for key in center_df.columns}
    sim_params = {"Nx": 32, "dx": 1.0, "Nt": 1_000, "dt": 0.0025, "path": path}

    for i, row in center_df.iterrows():
        A = row["A"]
        B = row["B"]
        Du = row["Du"]
        Dv = row["Dv"]

        sample_ball(A, B, Du, Dv, sigma, 25, sim_params, initial_conditions, run_id=run_id)
