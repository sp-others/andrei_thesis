# import modules

import numpy as np
from pysindy.differentiation import *
import pysindy as ps
from sklearn.metrics import mean_squared_error
from pysindy.optimizers import STLSQ
from pyGPGO.covfunc import squaredExponential
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO
import matplotlib.pyplot as plt
from pysindy.differentiation import FiniteDifference


def read_data(file_path):
    """

    :param file_path: file to extract Electroencephalograms from
    :return: 13 channels from the file in matrix form
    """
    file = open(file_path, 'r')
    matrix = []
    for line in file:
        tokens = line.split(",")
        matrix.append(list(map(lambda x: float(x), tokens)))
    return np.matrix(matrix)


# Data set preparing

N = 2  # Number of files used for training
X = []  # Training data_set
for it in range(1, N + 1):
    X.append(read_data(f"training_{it}.csv"))
X_data_set = np.array(X)

variav = 1

rows = 10
columns = 16384

sfd1 = SmoothedFiniteDifference()

matrix = []
time = range(0, 16384)
time = list(time)
for it in range(10):
    matrix.append(time)
matrix2 = []
matrix2.append(matrix)
matrix2.append(matrix)
t = np.linspace(0, 10, 1)
x = np.sin(t)
sfd = SmoothedFiniteDifference(smoother_kws={'window_length': 5})
print(x)
print(t)
"""
for it in range(10):
    x_derivative.append(sfd1._differentiate(X_data_set[0][it],time))
for it in range(10):
    x_derivative.append(sfd1._differentiate(X_data_set[1][it],time))
"""


# model initialization and objective function
def setup_model(threshold, alpha, polynomial_degree, trig_degree):
    differentiation_method = SmoothedFiniteDifference()
    optimizer = STLSQ(threshold=threshold, alpha=alpha)
    poly_lib = ps.PolynomialLibrary(degree=int(polynomial_degree), include_bias=False)
    trig_lib = ps.FourierLibrary(
        n_frequencies=int(trig_degree))  # TODO try with include bias true and include_sin_cos_true/false
    custom_lib = poly_lib + trig_lib

    model_3 = ps.SINDy(

        optimizer=optimizer,
        differentiation_method=differentiation_method,
        feature_library=custom_lib,
        feature_names=["CHF5", "CHFC1", "CHP5", "CHCP1", "CHP4", "CHPO8", "CHFP2", "CHFC6", "CHFZ", "CHPZ"],
    )
    model_3.fit(X_data_set, t=matrix2, multiple_trajectories=True)
    error = model_3.score(X_data_set, t=matrix2, multiple_trajectories=True,
                          metric=mean_squared_error) + variav * model_3.complexity()
    return error


# Bayesian optimizer

cov = squaredExponential()
surogate = GaussianProcess(cov)
acq = Acquisition(mode='ExpectedImprovement')
params = {'threshold': ('int', [0, 1]),
          'alpha': ('int', [0, 1]),
          'polynomial_degree': ('int', [2, 100]),
          'trig_degree': ('int', [2, 100])
          }
np.random.seed(23)
gpgo = GPGO(surogate, acq, setup_model, params, n_jobs=4)
gpgo.run(max_iter=200, init_evals=30)
