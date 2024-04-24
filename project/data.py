import numpy as np


# data prep
def gaussian_noise(n=1):
    return np.random.normal(size=(n))


def ar_data_generator(init_seq, weights, noise_func, steps=500, normalize=False):
    """Generate AR(n) simulation data.

    Args:
        init_seq (np.array): Initial values to start with. An AR(n) process should have n initial values.
        weights (np.array): Weights on each prev time step and the noise term.
        noise_func (func): Calling this noise function gives a random term.
        steps (int): Length of the generated sequence.
    """
    init_steps = len(init_seq)
    sim_data = np.zeros(steps + init_steps)
    sim_data[:init_steps] = init_seq
    for i in range(steps):
        cur_val = 0
        for j in range(init_steps):
            cur_val += weights[j] * sim_data[i+j]
        cur_val += weights[-1] * noise_func()
        sim_data[i+init_steps] = cur_val
    sim_data = sim_data[init_steps:]
    if normalize:
        sim_data = (sim_data - np.mean(sim_data)) / np.std(sim_data)
    return sim_data


def mytan(x):
    return np.tan(x)


def ar_nonlinear_generator(init_seq, weights, noise_func, nonlinear_func=mytan, steps=500, normalize=False):
    """Generate nonlinear AR(n) simulation data.

    Args:
        init_seq (np.array): Initial values to start with. An AR(n) process should have n initial values.
        weights (np.array): Weights on each prev time step and the noise term.
        noise_func (func): Calling this noise function gives a random term.
        steps (int): Length of the generated sequence.
    """
    init_steps = len(init_seq)
    sim_data = np.zeros(steps + init_steps)
    sim_data[:init_steps] = init_seq
    for i in range(steps):
        cur_val = 0
        for j in range(init_steps):
            cur_val += nonlinear_func(weights[j] * sim_data[i+j])
        cur_val += weights[-1] * noise_func()
        sim_data[i+init_steps] = cur_val
    sim_data = sim_data[init_steps:]
    if normalize:
        sim_data = (sim_data - np.mean(sim_data)) / np.std(sim_data)
    return sim_data


def multi_variate_data_generator(init_data, theta, u, noise_func, steps, dist_shift_factor):
    seq_data_list = []
    cur_data = np.array(init_data)
    for _ in range(steps):
        u = u + dist_shift_factor
        u = u / np.linalg.norm(u)
        u = u[None, :]
        trans_mat = u * u.transpose()
        u = np.squeeze(u)
        new_data = trans_mat @ cur_data + theta * np.array(noise_func())
        cur_data = new_data
        seq_data_list.append(new_data)
    return np.array(seq_data_list)


def get_data(data_type, init_seq=None, weights=None, noise_func=None, normalize=None, theta=None, dist_shift_factor=None):
    steps = 10000
    if data_type == 'ar':
        sim_data = ar_data_generator(init_seq, weights, noise_func, steps, normalize=normalize)
        data = sim_data[100:, None]
    elif data_type == 'multivar':
        sim_data = multi_variate_data_generator(init_data=init_seq, theta=theta, u=weights, noise_func=noise_func, steps=steps, dist_shift_factor=dist_shift_factor)
        data = sim_data[100:, :]
    elif data_type == 'nonlinear_ar':
        sim_data = ar_nonlinear_generator(init_seq, weights, noise_func, steps, normalize=normalize)
        data = sim_data[100:, None]
    return data