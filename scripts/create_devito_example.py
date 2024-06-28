from create_netcdf_input import *

def initial_ring_function(member, coupled_idx, x_position, y_position):
    r = (x_position - .5)**2. + (y_position - .5)**2.
    res = np.zeros(shape=x_position.shape)
    res[np.logical_and(.05 <= r, r <= .1)] = 1.
    print(res)
    return res

def zero_func(member, coupled_idx, x_position, y_position):
    return np.zeros(shape=x_position.shape)

def const_sigma(member, x_position, y_position):
    return np.ones(shape=x_position.shape) * 0.5

def f_scalings_ring(member, size):
    f = np.zeros(size)
    f[1] = -50.
    return f


create_input_file('data/example_devito.nc', 'data/example_devito_out.nc', type_of_equation=0, 
                      x_size=100, x_length=1., y_size=100, y_length=1., boundary_value_type=0,
                      scalar_type=0, n_coupled=1, 
                      coupled_function_order=2, number_timesteps=500,
                      final_time=.0025, number_snapshots=5, n_members=1, initial_value_function=initial_ring_function,
                      sigma_function=const_sigma, bc_neumann_function=zero_func, f_value_function=f_scalings_ring)
