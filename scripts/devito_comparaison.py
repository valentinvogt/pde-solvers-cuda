
from create_netcdf_input import *

def initial_ring_function(member, coupled_idx, x_position, y_position):
    r = (x_position - .5)**2. + (y_position - .5)**2.
    res = np.zeros(shape=x_position.shape)
    print(res.shape)
    res[np.logical_and(.05 <= r, r <= .1)] = 1.
    return res


def sigma(member, x_position, y_position):
    res = np.ones(shape=x_position.shape) * 0.01
    return res


def f_scalings_ring(member, size):
    f = np.zeros(size)
    f[1] = 2.
    return f


if len(sys.argv) != 2:
    print("input size of grid as command line argument!")
    print("example: python scripts/benchmark_brusselator.py 100")

size = int(sys.argv[1])
print(size)

create_input_file('data/example_devito.nc', 'data/example_devito_out.nc', type_of_equation=0, 
                      x_size=size, x_length=1., y_size=size, y_length=1., boundary_value_type=0,
                      scalar_type=0, n_coupled=1, 
                      coupled_function_order=2, number_timesteps=10000,
                      final_time=0.25, number_snapshots=5, n_members=1, initial_value_function=initial_ring_function,
                      sigma_function=sigma, bc_neumann_function=initial_ring_function, f_value_function=f_scalings_ring)
