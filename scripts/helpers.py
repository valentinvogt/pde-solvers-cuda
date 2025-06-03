import numpy as np
import json

def zero_func(member, coupled_idx, x_position, y_position):
    return np.zeros(shape=x_position.shape)


def const_sigma(member, x_position, y_position):
    return np.ones(x_position.shape)


def f_scalings(model, A, B):
    if model == "bruss":
        return lambda member, size: f_scalings_brusselator(member, size, A, B)
    elif model == "gray_scott":
        return lambda member, size: f_scalings_gray_scott(member, size, A, B)
    elif model == "fhn":
        return lambda member, size: f_scalings_fhn(member, size, A, B)
    else:
        raise ValueError("Not implemented for model: " + model)


def f_scalings_brusselator(member, size, A, B):
    assert size == 18
    f = np.zeros(size)
    f[0] = A  # constant in first function
    f[2] = -B - 1.0  # u-term in first function
    f[10] = 1.0  # u^2v in first function
    f[3] = B  # u term in second function
    f[11] = -1.0  # u^2v in second function

    return f


def f_scalings_gray_scott(member, size, A, B):
    assert size == 18
    f = np.zeros(size)
    f[0] = A
    f[2] = -A
    f[7] = -A - B
    f[14] = -1
    f[15] = 1

    return f


def f_scalings_fhn(member, size, A, B):
    assert size == 32
    f = np.zeros(size)
    f[0] = A
    f[2] = 1
    f[3] = -B
    f[6] = -1
    f[8] = -1
    f[9] = -B

    return f


def create_json(dict, filename):
    with open(filename, "w") as json_file:
        json.dump(dict, json_file)
