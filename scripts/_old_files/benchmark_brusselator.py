
from create_netcdf_input import *
import sys

def initial_noisy_function(member, coupled_idx, x_position, y_position):
    A = 2.5
    B = 5.5
    np.random.seed(1)
    if coupled_idx == 0:
        u = A * np.ones(shape=x_position.shape) + np.random.normal(0., 0.1, size=x_position.shape)
    elif coupled_idx == 1:
        u = (B / A) * np.ones(shape=x_position.shape) - np.random.normal(0., 0.1, size=x_position.shape)
    else:
        print("initial_noisy_function is only meant for n_coupled == 2!")
        u = 0. * x_position
    return u  


def zero_func(member, coupled_idx, x_position, y_position):
    return np.zeros(shape=x_position.shape)

def const_sigma(member, x_position, y_position):
    return np.ones(x_position.shape) * 2.

def f_scalings_brusselator(member, size):
    A = 2.5
    B = 5.5
    assert(size == 18)
    f = np.zeros(size)
    f[0] = A         # constant in first function
    f[2] = - B - 1.  # u-term in first function
    f[10] = 1.       # u^2v in first function
    f[7] = B         # u term in second function
    f[11] = -1.      # u^2v in second function
    return f




if len(sys.argv) != 2:
    print("input size of grid as command line argument!")
    print("example: python scripts/benchmark_brusselator.py 100")

size = int(sys.argv[1])
print(size)

create_input_file('data/benchmark.nc', 'data/bench_out.nc', type_of_equation=0, 
                      x_size=size, x_length=float(size), y_size=size, y_length=float(size), boundary_value_type=2,
                      scalar_type=0, n_coupled=2, 
                      coupled_function_order=3, number_timesteps=1000,
                      final_time=0.1, number_snapshots=2, n_members=1, initial_value_function=initial_noisy_function,
                      sigma_function=const_sigma, bc_neumann_function=zero_func, f_value_function=f_scalings_brusselator)
