from create_netcdf_input import *

def initial_noisy_function(member, coupled_idx, x_position, y_position):
    if coupled_idx == 0:
        u = np.exp(x_position + y_position)
    elif coupled_idx == 1:
        # TODO
        u = np.zeros(shape=x_position.shape)
        # u = (B / A) * np.ones(shape=x_position.shape) - np.random.normal(0., 0.1, size=x_position.shape)
    else:
        print("initial_noisy_function is only meant for n_coupled == 2!")
        u = np.zeros(shape=x_position.shape)
    return u  


def one_func(member, coupled_idx, x_position, y_position):
    return np.ones(shape=x_position.shape) * 1000. 

def const_sigma(member, x_position, y_position):
    return np.ones(x_position.shape) * 2.

def sigma(member, x_position, y_position):
    return np.exp(-x_position - y_position) +  0.25

def f_lin(member, size):
    f = np.zeros(size)
    f[0] = 1000.
    f[2] = -0.5
    return f

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

create_input_file('data/test.nc', 'data/test_out.nc', type_of_equation=0, 
                      x_size=256, x_length=1., y_size=256, y_length=1., boundary_value_type=1,
                      scalar_type=0, n_coupled=2, 
                      coupled_function_order=3, number_timesteps=5000,
                      final_time=0.01, number_snapshots=4, n_members=1, initial_value_function=initial_noisy_function,
                      sigma_function=sigma, bc_neumann_function=one_func, f_value_function=f_lin)
