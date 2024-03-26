import numpy as np
import pysindy as ps
from sklearn.metrics import mean_squared_error

np.random.seed(42)
TIME = np.linspace(0, 1, 1000)
data_sine = np.linspace(-np.pi / 2, np.pi / 2, num=1000)
x_data = np.array(data_sine)

param_range = range(2, 101, 10)
for param1 in param_range:
    for param2 in param_range:
        feature_lib = ps.PolynomialLibrary(degree=int(param1)) + ps.FourierLibrary(n_frequencies=int(param2))
        model = ps.SINDy(
            differentiation_method=(ps.FiniteDifference(order=2)),
            feature_library=feature_lib,
            optimizer=(ps.STLSQ(threshold=0.2)),
            feature_names=["x"])
        model.fit(x_data, t=TIME)
        # print(f'parametrii folositi in model sunt acestea{param1, param2}\n')
        print(f'param1: {param1}, param2: {param2} score: {model.score(x_data, metric=mean_squared_error)}')
