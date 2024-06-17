import datetime
import timeit

import numpy as np
import pandas as pd
from pyGPGO.GPGO import GPGO
from pyGPGO.covfunc import squaredExponential
from sklearn.model_selection import train_test_split
from pysindy import SINDy, PolynomialLibrary, FourierLibrary, STLSQ
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.acquisition import Acquisition
from scipy.integrate import odeint
from osqp import OSQP


# Function to read the data
def load_data(file1, file2):
    header = None
    used_columns = list(range(1, 10 + 1))
    data1 = pd.read_csv(file1, header=header, usecols=used_columns).values
    data2 = pd.read_csv(file2, header=header, usecols=used_columns).values
    return data1, data2


# Define the objective function
def objective(degree, n_frequencies, lambda_val, threshold):
    poly_library = PolynomialLibrary(degree=int(degree), include_bias=True)
    fourier_library = FourierLibrary(n_frequencies=int(n_frequencies))
    feature_library = poly_library + fourier_library

    model = SINDy(feature_library=feature_library, optimizer=STLSQ(threshold=threshold, alpha=lambda_val))

    x_dot_predicted = model.fit(x, t=t).predict(x)
    error = np.mean((x_dot - x_dot_predicted) ** 2)

    return -error


# Load data
file1 = 'training_1.csv'
file2 = 'training_2.csv'
data1, data2 = load_data(file1, file2)

# Assuming time vector t and derivative x_dot are known
# For the sake of this example, let's create synthetic ones
t = np.linspace(0, 10, len(data1))
x = data1
x_dot = np.gradient(data1, axis=0)  # Replace this with actual derivative if available

# Define the parameter space
int_bounds = [2, 10]
cont_bounds = [1e-128, 1e-64]
# TODO: find the type for lambda_val & threshold
param_bounds = {'degree': ('int', int_bounds), 'n_frequencies': ('int', int_bounds), 'lambda_val': ('cont', cont_bounds),
                'threshold': ('cont', cont_bounds)}

# Set up Bayesian Optimization
cov = squaredExponential()
surogate = GaussianProcess(cov)
acq = Acquisition(mode='ExpectedImprovement')

gpgo = GPGO(surogate, acq, objective, param_bounds, n_jobs=13)
# gpgo = GPGO(f=objective,
#                             domain=param_bounds,
#                             acquisition='UCB',
#                             y_max=None)

# Run Bayesian Optimization
start_time = datetime.datetime.now().isoformat()
print(start_time)
GPGO_ITERATIONS = 20
gpgo.run(max_iter=GPGO_ITERATIONS)
end_time = datetime.datetime.now().isoformat()

# Get the best parameters
best_params = gpgo.getResult()

print("Best Parameters:")
print(best_params)

# Refine the model with the best parameters
degree_best = int(best_params[0]['degree'])
n_frequencies_best = int(best_params[0]['n_frequencies'])
lambda_best = best_params[0]['lambda_val']
threshold_best = best_params[0]['threshold']

poly_library_best = PolynomialLibrary(degree=degree_best, include_bias=True)
fourier_library_best = FourierLibrary(n_frequencies=n_frequencies_best)
feature_library_best = poly_library_best + fourier_library_best

best_model = SINDy(feature_library=poly_library_best, optimizer=STLSQ(threshold=threshold_best, alpha=lambda_best))
best_model.fit(x, t=t)
x_dot_predicted_best = best_model.predict(x)

print("Best Model Predictions:")
print(x_dot_predicted_best)
print(f'GPGO with {GPGO_ITERATIONS} iterations ran from')
print(start_time)
print('to')
print(end_time)
