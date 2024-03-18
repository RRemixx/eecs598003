import numpy as np
import matplotlib.pyplot as plt
import torch
import random

from ts_pred_helper import *


def func1():
    iterations = 3
    model_names = ['FF', 'RNN', 'Transformer']
    models = [FF(1, 3, 64), RNN(1, 64, 4, 1), Transformer(1, 64, 2, 4, 3)]
    init_seq = np.array([1, 2, 3])
    weights = np.array([0.1, 0.2, 0.7, 0.01])
    noise_func = gaussian_noise

    x_ticks = np.linspace(-5, 5, 20)
    noise_factors = [10**x_tick for x_tick in x_ticks]


    def train_and_save(idx):
        results = {}
        for noise_factor in tqdm(noise_factors):
            weights[-1] = noise_factor
            results[noise_factor] = []
            for i in range(iterations):
                set_seed(i+1)
                tys, pys = train_eval(models[idx], init_seq, weights, gaussian_noise, epochs=200)
                results[noise_factor].append((tys, pys))
        pickle_save(f'results/{model_names[idx]}_res.pkl', results)

    for idx in range(3):
        train_and_save(idx)

def func2():
    iterations = 3
    model_names = ['FF', 'RNN', 'Transformer']
    input_dim = 3
    seq_len = 3
    models = [FF(input_dim, seq_len, 64, output_dim=1), RNN(input_dim, 64, 4, output_dim=1), Transformer(input_dim, 64, 2, 4, seq_len, output_dim=1)]
    init_data = np.array([1, 2, 3])
    u = np.array([1, 1.2, 0.8])
    dist_shift_factor = np.array([0, 0, 0])

    def noise_function():
        return gaussian_noise(input_dim)

    x_ticks = np.linspace(-3, 3, 12)
    noise_factors = [10**x_tick for x_tick in x_ticks]


    def train_and_save(idx):
        results = {}
        for noise_factor in tqdm(noise_factors):
            results[noise_factor] = []
            for i in range(iterations):
                set_seed(i+1)
                tys, pys = train_eval(
                    model=models[idx], 
                    init_seq=init_data, 
                    weights=u, 
                    noise_func=noise_function, 
                    data_type='other',
                    theta=noise_factor,
                    dist_shift_factor=dist_shift_factor,
                    window_size=seq_len,
                    epochs=200,
                )
                results[noise_factor].append((tys, pys))
        pickle_save(f'results/{model_names[idx]}_res_1.pkl', results)


    for idx in range(3):
        train_and_save(idx)


func1()
func2()