from create_netcdf_input import *

def initial_noisy_function(member, coupled_idx, x_position, y_position):
    A = 2.
    B = 5.5
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
    return np.ones(x_position.shape) * 0.00001

def f_scalings_brusselator(member, size):
    A = 2.
    B = 5.5
    assert(size == 18)
    f = np.zeros(size)
    f[0] = A        # constant in first function
    f[2] = - B - 1.  # u-term in first function
    f[10] = 1. # u^2v in first function
    f[7] = B #u term in second function
    f[11] = -1.     #u^2v in second function
    return f

create_input_file('data/brusselator.nc', 'data/brusselator_out.nc', type_of_equation=0, 
                      x_size=160, x_length=0.5, y_size=160, y_length=0.5, boundary_value_type=2,
                      scalar_type=0, n_coupled=2, 
                      coupled_function_order=3, number_timesteps=160000,
                      final_time=40., number_snapshots=16, n_members=1, initial_value_function=initial_noisy_function,
                      sigma_function=const_sigma, bc_neumann_function=zero_func, f_value_function=f_scalings_brusselator)
