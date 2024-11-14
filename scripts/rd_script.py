from create_netcdf_input import *

A = 5
B = 9


def initial_noisy_function(member, coupled_idx, x_position, y_position):
    np.random.seed(1)
    sigma = 0.1
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


def zero_func(member, coupled_idx, x_position, y_position):
    return np.zeros(shape=x_position.shape)


def const_sigma(member, x_position, y_position):
    return np.ones(x_position.shape)


def f_scalings_brusselator(member, size):
    assert size == 18
    f = np.zeros(size)
    f[0] = A  # constant in first function
    f[2] = -B - 1.0  # u-term in first function
    f[10] = 1.0  # u^2v in first function
    f[3] = B  # u term in second function
    f[11] = -1.0  # u^2v in second function

    return f


create_input_file(
    "data/rd.nc",
    "data/rd_out.nc",
    type_of_equation=2,
    x_size=100,
    x_length=100.0,
    y_size=100,
    y_length=100.0,
    boundary_value_type=2,
    scalar_type=0,
    n_coupled=2,
    coupled_function_order=3,
    number_timesteps=20000,
    final_time=100.0,
    number_snapshots=200,
    n_members=1,
    initial_value_function=initial_noisy_function,
    sigma_function=const_sigma,
    bc_neumann_function=zero_func,
    f_value_function=f_scalings_brusselator,
    Du=2.0,
    Dv=22.0,
)
