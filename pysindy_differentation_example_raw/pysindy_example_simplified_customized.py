import numpy as np
import pysindy as ps
from pysindy.utils import linear_damped_SHO
from scipy.integrate import solve_ivp

name = 'PySINDy Finite Difference'
method = ps.FiniteDifference()

equations_clean = {}
equations_noisy = {}
coefficients_clean = {}
coefficients_noisy = {}
input_features = ["x", "y"]
threshold = 0.5

noise_level = 0.1

dt = 0.01
t_train = np.arange(0, 10, dt)
t_train_span = (t_train[0], t_train[-1])
x0_train = [2, 0]
x_train = solve_ivp(
    linear_damped_SHO, t_train_span, x0_train, t_eval=t_train
).y.T
x_train_noisy = x_train + noise_level * np.random.randn(*x_train.shape)

model = ps.SINDy(
    differentiation_method=method,
    optimizer=ps.STLSQ(threshold=threshold),
    t_default=dt,
    feature_names=input_features,
)

model.fit(x_train, quiet=True)
equations_clean[name] = model.equations()
coefficients_clean[name] = model.coefficients()

model.fit(x_train_noisy, quiet=True)
equations_noisy[name] = model.equations()
coefficients_noisy[name] = model.coefficients()
