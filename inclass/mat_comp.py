import numpy as np


def get_spectrum(n, theta, outliers):
    G = np.random.normal(size=(n, n))
    X = (G + np.transpose(G)) / (2*n)**0.5
    u = np.random.rand(n)
    u = u / np.linalg.norm(u)
    u = u[None, :]
    X_tilda = X + theta * u * u.transpose()
    X_bar = X_tilda + outliers
    eigvals, eigvecs = np.linalg.eig(X_bar)
    idx = np.argmax(eigvals)
    max_eigvec = eigvecs[:, idx]
    inner_prod = np.inner(max_eigvec, u.reshape(-1))
    return sorted(eigvals), inner_prod