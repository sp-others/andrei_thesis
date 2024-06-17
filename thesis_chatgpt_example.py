import datetime
import os
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyGPGO.GPGO import GPGO
from pyGPGO.covfunc import squaredExponential
from pysindy import SINDy, PolynomialLibrary, FourierLibrary, STLSQ
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.acquisition import Acquisition


"""
channels:
F5|FC1|P5|CP1|P4|PO8|FP2|FC6|FZ|PZ
3p;7p;10p;18p;22p;36p;43p;46p;48p;57p
"""

eeg_channels = ["CH_F5", "CH_FC1", "CH_P5", "CH_CP1", "CH_P4", "CH_PO8", "CH_FP2", "CH_FC6", "CH_FZ", "CH_PZ"]

file1 = 'training_1.csv'
file2 = 'training_2.csv'
DATA_WIDTH = 9  # number of columns used from the csv file

ALPHA = 0

int_bounds = ('int', [2, 10])
threshold_bounds = ('cont', [0, 0.1])
alpha_bounds = ('cont', [0, 1e-12])

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
t = np.linspace(1, len(data1), len(data1), dtype=int)
x = data1
x_dot = np.gradient(data1, axis=0)  # Replace this with actual derivative if available

# Define the parameter space
# TODO: find the type for lambda_val & threshold
param_bounds = {
    'degree': int_bounds,
    'n_frequencies': int_bounds,
    'lambda_val': threshold_bounds,
    'threshold': alpha_bounds
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
start = time.time()
print(start_time)
gpgo.run(max_iter=GPGO_ITERATIONS)
end_time = datetime.datetime.now().isoformat()
end = time.time()

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
print(f'Total time: {end - start} seconds')

# Plot actual vs expected derivatives
plt.figure()
plt.plot(t, x_dot, label='Actual Derivative')
plt.plot(t, x_dot_predicted_best, label='Predicted Derivative')
plt.xlabel('Time')
plt.ylabel('Derivative')
plt.legend()
plt.title('Actual vs. Predicted Derivative')
plt.show()


hyperparameter_history = np.array(hyperparameter_history)
# Plot evolution of hyperparameters (degree & n_frequencies)
plt.figure()
plt.plot(hyperparameter_history[:, 0], label='degree')
plt.plot(hyperparameter_history[:, 1], label='n_frequencies')
plt.xlabel('Iteration')
plt.ylabel('Hyperparameter Value')
plt.legend()
plt.title('Evolution of Hyperparameters (degree & n_frequencies)')
plt.show()

# Plot evolution of hyperparameters (lambda_val)
plt.figure()
plt.plot(hyperparameter_history[:, 2], label='lambda_val')
plt.xlabel('Iteration')
plt.ylabel('Hyperparameter Value')
plt.legend()
plt.title('Evolution of Hyperparameters (lambda_val)')
plt.show()

# Plot evolution of hyperparameters (threshold)
plt.figure()
plt.plot(hyperparameter_history[:, 3], label='threshold')
plt.xlabel('Iteration')
plt.ylabel('Hyperparameter Value')
plt.legend()
plt.title('Evolution of Hyperparameters (threshold)')
plt.show()

# Plot evolution of error
plt.figure()
plt.plot(error_history, label='Error')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.legend()
plt.title('Evolution of Error')
plt.show()

plt.figure(figsize=(12, len(eeg_channels)))
t_columns = np.linspace(1, DATA_WIDTH, DATA_WIDTH, dtype=int)
for i, eeg_data in enumerate(data1):
    plt.plot(t_columns, data1[i], label=eeg_channels[i])

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('EEG Data from CSV')
plt.legend(loc='upper right')
plt.show()

# Plot the data from the CSV file w/ bigger width
data_big = read_data(file1, 400)

width = len(data_big[0])
t = np.linspace(1, width, width, dtype=int)
plt.figure(figsize=(12, len(eeg_channels)))
for i, eeg_data in enumerate(data_big):
    plt.plot(t, data_big[i], label=eeg_channels[i])

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('EEG Data from CSV')
plt.legend(loc='upper right')
plt.show()
