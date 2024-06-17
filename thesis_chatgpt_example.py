import datetime
import os

import numpy as np
import pandas as pd
from pyGPGO.GPGO import GPGO
from pyGPGO.covfunc import squaredExponential
from pysindy import SINDy, PolynomialLibrary, FourierLibrary, STLSQ
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.acquisition import Acquisition


file1 = 'training_1.csv'
file2 = 'training_2.csv'
DATA_WIDTH = 10  # number of columns used from the csv file

ALPHA = 0

int_bounds = [2, 10]
cont_bounds = [1e-128, 1e-64]

GPGO_ITERATIONS = 10
CPU_CORES_FOR_GPGO = int(os.getenv('CPU_CORES_FOR_GPGO', 4))


# Function to read the data
def read_data(filename, last_column_number=None):
    """
    Read data from a CSV file
    :param filename: the csv file that contains the data
    :param last_column_number: if None, all columns are used, else only the first last_column_number columns are used
    :return:
    """
    if last_column_number is None:
        used_columns = None
    else:
        used_columns = list(range(1, last_column_number + 1))
    return pd.read_csv(filename, header=None, usecols=used_columns).values


# Define the objective function
def objective(degree, n_frequencies, lambda_val, threshold):
    poly_library = PolynomialLibrary(degree=int(degree), include_bias=True)
    fourier_library = FourierLibrary(n_frequencies=int(n_frequencies))
    feature_library = poly_library + fourier_library

    model = SINDy(feature_library=feature_library, optimizer=STLSQ(threshold=threshold, alpha=lambda_val))

    x_dot_predicted = model.fit(x, t=t).predict(x)
    error = np.mean((x_dot - x_dot_predicted) ** 2) + ALPHA * model.complexity

    # Store hyperparameters and error for plotting
    global hyperparameter_history, error_history
    hyperparameter_history.append((degree, n_frequencies, lambda_val, threshold))
    error_history.append(error)

    return -error


# Load data
data1 = read_data(file1, DATA_WIDTH)

# Assuming time vector t and derivative x_dot are known
# For the sake of this example, let's create synthetic ones
t = np.linspace(0, 10, len(data1))
x = data1
x_dot = np.gradient(data1, axis=0)  # Replace this with actual derivative if available

# Define the parameter space
# TODO: find the type for lambda_val & threshold
param_bounds = {
    'degree': ('int', int_bounds),
    'n_frequencies': ('int', int_bounds),
    'lambda_val': ('cont', cont_bounds),
    'threshold': ('cont', cont_bounds)
}

# Initialize history storage
hyperparameter_history = []
error_history = []

# Set up Bayesian Optimization
cov = squaredExponential()
surogate = GaussianProcess(cov)
acq = Acquisition(mode='ExpectedImprovement')

print(f'Using {CPU_CORES_FOR_GPGO} CPU cores for GPGO')
gpgo = GPGO(surogate, acq, objective, param_bounds, n_jobs=CPU_CORES_FOR_GPGO)

# Run Bayesian Optimization
start_time = datetime.datetime.now().isoformat()
print(start_time)
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

best_model = SINDy(feature_library=feature_library_best, optimizer=STLSQ(threshold=threshold_best, alpha=lambda_best))
best_model.fit(x, t=t)
x_dot_predicted_best = best_model.predict(x)

print("Best Model Predictions:")
print(x_dot_predicted_best)
print(f'GPGO with {GPGO_ITERATIONS} iterations ran from')
print(start_time)
print('to')
print(end_time)
