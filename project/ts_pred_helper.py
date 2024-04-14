import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import random
from tqdm import tqdm
import pickle
import pickle5
from sklearn.preprocessing import MinMaxScaler

from data import *



def rmse(predictions, targets):
    """
    Root Mean Squared Error
    Args:
        predictions (np.ndarray): Point Predictions of the model
        targets (np.ndarray): Point Targets of the model
    Returns:
        float: RMSE
    """
    return np.sqrt(((predictions - targets) ** 2).mean())


def norm_rmse(predictions, targets):
    """
    Root Mean Squared Error
    Args:
        predictions (np.ndarray): Point Predictions of the model
        targets (np.ndarray): Point Targets of the model
    Returns:
        float: RMSE
    """
    try:
        scale = MinMaxScaler()
        targets = scale.fit_transform(targets[:,None])
        predictions = scale.transform(predictions[:,None])
    finally:
       return np.sqrt(((predictions - targets) ** 2).mean())


def mape(predictions, targets):
    """
    Mean Absolute Percentage Error
    Args:
        predictions (np.ndarray): Predictions of the model
        targets (np.ndarray): Targets of the model
    Returns:
        float: MAPE
    """
    targets[targets==0] = np.nan
    return np.nanmean(np.abs((predictions - targets) / targets)) * 100


class SeqData(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.X[idx, :], self.y[idx])


def prepare_ts_ds(data, window_size, target_idx, train_size, val_size, test_size, batch_size):
    """Generate a time series dataset from simulation data.

    Args:
        data (np.array): Possible with F time series data of length N, in the shape of (N, F)
        target_idx (int): The index of the feature to be predicted.
        window_size (int)
        train_size (int)
        val_size (int)
        test_size (int)
    """
    data_len = len(data)
    train_xs, train_ys, test_xs, test_ys = [], [], [], []
    cur_test_num = 0
    for i in range(data_len):
        if i >= window_size:
            if i < (train_size + val_size) * 1.5:
                train_xs.append(data[i-window_size:i, :])
                train_ys.append(data[i, target_idx])
            elif cur_test_num < test_size:
                cur_test_num += 1
                test_xs.append(data[i-window_size:i, :])
                test_ys.append(data[i, target_idx])
            else:
                break
    train_xs = np.array(train_xs, dtype=np.float32)
    train_ys = np.array(train_ys, dtype=np.float32)
    test_xs = np.array(test_xs, dtype=np.float32)
    test_ys = np.array(test_ys, dtype=np.float32)
    dataset = SeqData(X=train_xs, y=train_ys)
    test_dataset = SeqData(X=test_xs, y=test_ys)
    train_dataset, val_dataset, _ = torch.utils.data.random_split(dataset, [train_size, val_size, len(dataset)-train_size-val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size ,shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader


def perfect_model(x, weights):
    """
    x is in the shape of (batch size x seq length x input dim)
    """
    x = x[:, :, 0]
    res = []
    for b in range(x.shape[0]):
        cur_res = 0
        for i in range(len(weights)-1):
            cur_res += x[b, i] * weights[i]
        res.append(cur_res)
    return torch.tensor(res)


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_eval1(model, data, window_size, epochs):
    learning_rate = 2e-5
    device = torch.device('cuda:0')
    
    train_dl, val_dl, test_dl = prepare_ts_ds(data, window_size, 0, 2000, 500, 1000, 32)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.L1Loss()

    for _ in range(epochs):
        model.train()
        losses = []
        for batch in train_dl:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_fn(y, y_pred)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach().cpu().numpy())
        # print(f'loss is {np.mean(losses)}')
        model.eval()
        val_losses = []
        for batch in val_dl:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            # print(y.shape)
            y_pred = model(x)
            # print(y_pred.shape)
            loss = loss_fn(y, y_pred)
            val_losses.append(loss.detach().cpu().numpy())
        # print(f'val loss is {np.mean(val_losses)}, perfect loss: {np.mean(plosses)}')
    # test
    model.eval()
    tys = []
    pys = []
    for batch in test_dl:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        # save
        y_pred = y_pred.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        for i in range(len(y)):
            tys.append(y[i])
            pys.append(y_pred[i])
    tys = np.array(tys)
    pys = np.array(pys)    
    return tys, pys


def train_eval(model, init_seq, weights, noise_func, normalize=False, data_type='ar', theta=1, dist_shift_factor=None, window_size=3, epochs=200):
    learning_rate = 2e-5
    device = torch.device('cuda:0')

    steps = 10000
    if data_type == 'ar':
        sim_data = ar_data_generator(init_seq, weights, noise_func, steps, normalize=normalize)
        data = sim_data[100:, None]
    else:
        sim_data = multi_variate_data_generator(init_data=init_seq, theta=theta, u=weights, noise_func=noise_func, steps=steps, dist_shift_factor=dist_shift_factor)
        data = sim_data[100:, :]
    
    train_dl, val_dl, test_dl = prepare_ts_ds(data, window_size, 0, 2000, 500, 1000, 32)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.L1Loss()

    for _ in range(epochs):
        model.train()
        losses = []
        for batch in train_dl:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_fn(y, y_pred)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach().cpu().numpy())
        # print(f'loss is {np.mean(losses)}')
        model.eval()
        val_losses = []
        plosses = []
        for batch in val_dl:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            # print(y.shape)
            y_pred = model(x)
            # print(y_pred.shape)
            loss = loss_fn(y, y_pred)
            val_losses.append(loss.detach().cpu().numpy())
            py_pred = perfect_model(x, weights).to(device)
            ploss = loss_fn(y, py_pred)
            plosses.append(ploss.detach().cpu().numpy())
        # print(f'val loss is {np.mean(val_losses)}, perfect loss: {np.mean(plosses)}')
    # test
    model.eval()
    tys = []
    pys = []
    perfect_pys = []
    for batch in test_dl:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        py_pred = perfect_model(x, weights).to(device)
        # save
        y_pred = y_pred.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        py_pred = py_pred.detach().cpu().numpy()
        for i in range(len(y)):
            tys.append(y[i])
            pys.append(y_pred[i])
            perfect_pys.append(py_pred[i])
    tys = np.array(tys)
    pys = np.array(pys)    
    return tys, pys


def pickle_save(fname, data):
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(fname, version5=False):
    if version5:
        with open(fname, 'rb') as handle:
            return pickle5.load(handle)
    with open(fname, 'rb') as handle:
        return pickle.load(handle)