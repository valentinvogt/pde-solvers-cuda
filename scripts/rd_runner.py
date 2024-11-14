import numpy as np
import os
import argparse, sys
from create_netcdf_input import create_input_file
from helpers import f_scalings, zero_func, const_sigma

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
n_snapshots: int = 100
"""

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="bruss")
parser.add_argument("--A", type=float, default=5)
parser.add_argument("--B", type=float, default=9)
parser.add_argument("--Nx", type=int, default=100)
parser.add_argument("--dx", type=float, default=1.0)
parser.add_argument("--Nt", type=int, default=1000)
parser.add_argument("--dt", type=float, default=0.01)
parser.add_argument("--Du", type=float, default=2.0)
parser.add_argument("--Dv", type=float, default=22.0)
parser.add_argument("--n_snapshots", type=int, default=100)

args = parser.parse_args()
model = args.model
A = args.A
B = args.B
Nx = args.Nx
dx = args.dx
Nt = args.Nt
dt = args.dt
Du = args.Du
Dv = args.Dv
n_snapshots = args.n_snapshots

def initial_noisy_function(member, coupled_idx, x_position, y_position):
    np.random.seed(1)
    sigma = 0.5
    if coupled_idx == 0:
        u = A * np.ones(shape=x_position.shape) + np.random.normal(
            0.0, sigma, size=x_position.shape
        )
    elif coupled_idx == 1:
        u = (B / A) * np.ones(shape=x_position.shape) + np.random.normal(
            0.0, sigma, size=x_position.shape
        )
    else:
        print("initial_noisy_function is only meant for n_coupled == 2!")
        u = 0.0 * x_position
    # u += (
    #     0.5
    #     * np.sin(2 * np.pi * x_position / 100)
    #     * np.sin(2 * np.pi * y_position / 100)
    # )
    return u



fn_order = 4 if model == "fhn" else 3
fn_scalings = f_scalings(model, A, B)
input_filename = f"data/{model}.nc"
create_input_file(
    f"data/{model}.nc",
    f"data/{model}_output.nc",
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
    initial_value_function=initial_noisy_function,
    sigma_function=const_sigma,
    bc_neumann_function=zero_func,
    f_value_function=fn_scalings,
    Du=Du,
    Dv=Dv,
)

print(input_filename)
