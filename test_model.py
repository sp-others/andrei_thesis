import numpy as np
import matplotlib.pyplot as plst
from mpl_toolkits import mplot3d
import pysindy as ps
import random
import pyGPGO
from pysindy import PolynomialLibrary, FourierLibrary, SINDy
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.covfunc import squaredExponential
from pyGPGO.acquisition import Acquisition
from pyGPGO.GPGO import GPGO
from sklearn.metrics import mean_squared_error


def generate_data():
    data_sine = np.linspace(-np.pi / 2, np.pi / 2, num=1000)
    return np.array(data_sine)


x_data = generate_data()
differentiation_method = ps.FiniteDifference(order=2)
alpha = 1
TIME = np.linspace(0, 1, 1000)


def get_model(param1, param2):
    poly_lib = ps.PolynomialLibrary(degree=int(param1))
    trig_lib = ps.FourierLibrary(n_frequencies=int(param2))
    custom_lib = poly_lib + trig_lib
    optimizer = ps.STLSQ(threshold=0.2)
    model = ps.SINDy(
        differentiation_method=differentiation_method,
        feature_library=custom_lib,
        optimizer=optimizer,
        feature_names=["x"])
    model.fit(x_data, t=TIME)
    # print(f'parametrii folositi in model sunt acestea{param1, param2}\n')
    return model


def get_score(model: SINDy):
    return model.score(x_data, metric=mean_squared_error) + alpha * model.complexity


for param1 in range(2, 101, 10):
    for param2 in range(2, 101, 10):
        model = get_model(param1, param2)
        score = get_score(model)
        print(f'param1: {param1}, param2: {param2} score: {score}')
