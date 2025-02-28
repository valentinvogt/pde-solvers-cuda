import numpy as np
from numpy.random import normal, uniform, randint
from functools import partial
import os
from dotenv import load_dotenv
from uuid import uuid4
import pandas as pd
import argparse
from typing import List
from itertools import product
import sys
from create_netcdf_input import create_input_file
from helpers import f_scalings, zero_func, const_sigma, create_json
from initial_conditions import (
    get_ic_function,
    InitialCondition,
    ModelParams,
    SimParams,
    RunInfo,
    ic_from_dict,
)

from config import CONFIG


def run_wrapper(
    model_params: ModelParams,
    sim_params: SimParams,
    initial_condition: InitialCondition,
    run_info: RunInfo,
    filename: str,
    random_seed: int = None,
    original_point: ModelParams = None,
):
    model = run_info.model
    run_id = run_info.run_id
    
    params = model_params.model_dump()
    A, B, Du, Dv = params.values()
    
    sim_values = sim_params.model_dump()
    Nx, dx, Nt, dt, n_snapshots = sim_values.values()
    
    ic_data = initial_condition.model_dump(mode='json')

    fn_order = 4 if model == "fhn" else 3
    fn_scalings = f_scalings(model, A, B)

    assert filename.endswith(".nc")
    input_filename = filename
    output_filename = filename.replace(".nc", "_output.nc")

    ic_function = get_ic_function(model, A, B, initial_condition)

    if random_seed is None:
        random_seed = randint(0, 2**32 - 1)

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
        initial_value_function=ic_function,
        sigma_function=const_sigma,
        bc_neumann_function=zero_func,
        f_value_function=fn_scalings,
        Du=Du,
        Dv=Dv,
    )

    log_dict = {
        "model": model,
        "A": A,
        "B": B,
        "Nx": Nx,
        "dx": dx,
        "Nt": Nt,
        "dt": dt,
        "Du": Du,
        "Dv": Dv,
        "initial_condition": ic_data,
        "random_seed": random_seed,
        "n_snapshots": n_snapshots,
        "filename": output_filename,
        "run_id": run_id,
    }
    if original_point is not None:
        log_dict["original_point"] = original_point.model_dump()

    create_json(
        log_dict,
        filename.replace(".nc", ".json"),
    )

def sample_ball(
    model_params: ModelParams,
    sim_params: SimParams,
    run_info: RunInfo,
    path: str,
    initial_conditions: List[InitialCondition],
    sampling_std: ModelParams,
    num_samples_per_point: int,
    num_samples_per_ic: int,
):
    params = model_params.model_dump()
    A, B, Du, Dv = params.values()
    std_params = sampling_std.model_dump()
    sigma_A = std_params["A"] * A
    sigma_B = std_params["B"] * B
    sigma_Du = std_params["Du"] * Du
    sigma_Dv = std_params["Dv"] * Dv

    for _ in range(num_samples_per_point):
        A_new = A + uniform(-sigma_A, sigma_A)
        B_new = B + uniform(-sigma_B, sigma_B)
        Du_new = Du + uniform(-sigma_Du, sigma_Du)
        Dv_new = Dv + uniform(-sigma_Dv, sigma_Dv)

        for ic in initial_conditions:
            for _ in range(num_samples_per_ic):
                run_wrapper(
                    ModelParams(A=A_new, B=B_new, Du=Du_new, Dv=Dv_new),
                    sim_params,
                    ic,
                    run_info,
                    filename=os.path.join(path, f"{uuid4()}.nc"),
                    original_point=model_params,
                )


def ball_sampling(
    centers: List[ModelParams],
    sim_params: SimParams,
    run_info: RunInfo,
    initial_conditions: List[InitialCondition],
    sampling_std: ModelParams,
    num_samples_per_point: int,
    num_samples_per_ic: int,
    output_dir: str,
):
    data_dir = os.getenv("DATA_DIR")
    path = os.path.join(data_dir, output_dir)
    os.makedirs(path, exist_ok=True)

    for center in centers:
        sample_ball(
            center,
            sim_params,
            run_info,
            path,
            initial_conditions,
            sampling_std,
            num_samples_per_point,
            num_samples_per_ic,
        )


def parameters_on_grid():
    config = CONFIG
    model = config["model"]
    run_id = config["run_id"]
    grid_mode = config["grid_mode"]
    grid_config = config["grid_config"]

    if grid_mode == "absolute":
        param_ranges = [
            grid_config["A"],
            grid_config["B"],
            grid_config["Du"],
            grid_config["Dv"],
        ]
        return [
            {"A": A, "B": B, "Du": Du, "Dv": Dv} 
            for A, B, Du, Dv in product(*param_ranges)
        ]
    elif grid_mode == "relative":
        params = []
        for A in grid_config["A"]:
            for B_over_A in grid_config["B_over_A"]:
                for Du in grid_config["Du"]:
                    for Dv_over_Du in grid_config["Dv_over_Du"]:
                        B = A * B_over_A
                        Dv = Du * Dv_over_Du
                        params.append({"A": A, "B": B, "Du": Du, "Dv": Dv})
        return params
    else:
        raise ValueError(f"Invalid range type: {grid_mode}")


def main():
    load_dotenv()
    config = CONFIG
    model = config["model"]
    run_id = config["run_id"]
    run_type = config["run_type"]
    center_definition = config["center_definition"]
    data_dir = os.getenv("DATA_DIR")
    path = os.path.join(data_dir, model, run_id)
    os.makedirs(path, exist_ok=True)

    sim_params = SimParams(**config["sim_params"])
    run_info = RunInfo(model=model, run_id=run_id)
    initial_conditions = [ic_from_dict(ic) for ic in config["initial_conditions"]]
    sampling_std = ModelParams(**config["sampling_std"])

    if center_definition == "from_grid":
        param_grid = parameters_on_grid()
    elif center_definition == "from_df":
        param_grid = pd.read_csv(config["df_path"])
    else:
        raise ValueError(f"Invalid center definition: {center_definition}")

    if run_type == "ball":
        centers = [ModelParams(**center) for center in param_grid]
        num_samples_per_point = config["num_samples_per_point"]
        num_samples_per_ic = config["num_samples_per_ic"]
        ball_sampling(
            centers,
            sim_params,
            run_info,
            initial_conditions,
            sampling_std,
            num_samples_per_point,
            num_samples_per_ic,
            path,
        )
    elif run_type == "one_trajectory":
        for center in param_grid:
            for ic in initial_conditions:
                run_wrapper(
                    ModelParams(**center),
                    sim_params,
                    ic,
                    run_info,
                    filename=os.path.join(path, f"{uuid4()}.nc"),
                )
    else:
        raise ValueError(f"Invalid run type: {run_type}")


if __name__ == "__main__":
    main()
