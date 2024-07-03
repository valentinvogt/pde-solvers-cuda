
from create_netcdf_input import *

# test cases for heat equation with periodic boundary conditions.
# this should primarily test if the run_from_netcdf function works
# properly with several inputs.
def initial_function(member, coupled_idx, x_position, y_position):
    if member == 0:
        return np.zeros(shape=x_position.shape)
    if member == 1:
        if coupled_idx == 0:
            return np.ones(shape=x_position.shape) * .5
        if coupled_idx == 1:
            # random pde to check if periodicity is correct
            # return np.zeros(shape=x_position.shape)
            return x_position**6 - x_position + y_position**4 - y_position
        if coupled_idx == 2:
            return np.zeros(shape=x_position.shape)

def is_ignored(member, coupled_idx, x_position, y_position):
    print("this should not be printed because \
          the periodic heat equation does not need \
          the derivative, therefore this function \
          does not get called")
    return np.ones(shape=x_position.shape) * 1000. 

def sigma(member, x_position, y_position):
    if member == 0:
        return np.zeros(shape=x_position.shape)
    if member == 1:
        return np.ones(shape=x_position.shape) * 0.1

def f(member, size):
    if member == 0:
        # f_1(x, y, z) = 1 => x(t) = t^2/2, x'(0) = 0
        # f_2(x, y, z) = 2x + y = t^2 + y => y(t) = exp(t) + exp(-t) - t^2 - 2, y'(0) = 0
        # f_3(x, y, z) = 6x = 3t^2 => z(t) = t^4/4, z'(0) = 0
        f = np.zeros(size)
        f[0] = 1.  # f_1 += 1
        f[4] = 1.  # f_2 += x^2
        f[10] = 1. # f_2 += y
        f[5] = 6.  # f_3 += 6x
        return f
    else:
        return np.zeros(size)


create_input_file('data/test_wave.nc', 'data/test_wave.nc', type_of_equation=1, 
                      x_size=64, x_length=1., y_size=64, y_length=1., boundary_value_type=2,
                      scalar_type=0, n_coupled=3, 
                      coupled_function_order=3, number_timesteps=5000,
                      final_time=2., number_snapshots=3, n_members=1, initial_value_function=initial_function,
                      sigma_function=sigma, bc_neumann_function=is_ignored, f_value_function=f)
