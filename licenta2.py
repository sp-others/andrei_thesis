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
DATA_WIDTH = 11  # number of columns used from the csv file

ALPHA = 1

int_bounds = ('int', [2, 10])
threshold_bounds = ('cont', [0, 0.1])
alpha_bounds = ('cont', [0, 1e-12])

GPGO_ITERATIONS = 20
CPU_CORES_FOR_GPGO = int(os.getenv('CPU_CORES_FOR_GPGO', 4))


# Initialize history storage
hyperparameter_history = []
error_history = []


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


def plot_derivatives(file_name, actual_derivative, expected_derivative):
    # plot the 2 derivatives, fully
    for derivative_type, derivative in {'actual': actual_derivative, 'expected': expected_derivative}.items():
        plt.figure()
        plt.plot(t, derivative)
        plt.xlabel('Time')
        plt.ylabel('Derivative')
        plt.title(f'{derivative_type} Derivative for {file_name}')
        plt.savefig(f'out/derivative_{file_name}_{derivative_type}.png', bbox_inches='tight')
        plt.show()
    # plot a plot for each set of channels from both derivatives
    for i, channel in enumerate(eeg_channels):
        plt.figure(figsize=(12, len(eeg_channels)))
        plt.plot(t_columns, actual_derivative[i], label=f'{channel} actual derivative')
        plt.plot(t_columns, expected_derivative[i], label=f'{channel} expected derivative')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'actual vs expected for data {file_name} for channel {channel}')
        plt.legend(loc='upper right')
        plt.savefig(f'out/derivative_{file_name}_{channel}.png', bbox_inches='tight')
        plt.show()


# Load data
file_names = ['training_1.csv', 'training_2.csv', 'validation_1.csv']
data_dict = {file_name: read_data(file_name, DATA_WIDTH) for file_name in file_names}
data1 = data_dict['training_1.csv']
data2 = data_dict['training_2.csv']
data3 = data_dict['validation_1.csv']

# Assuming time vector t and derivative x_dot are known
# For the sake of this example, let's create synthetic ones
t = np.linspace(1, len(data1), len(data1), dtype=int)
t_columns = np.linspace(1, DATA_WIDTH, DATA_WIDTH, dtype=int)
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


# Plot derivatives for data1
plot_derivatives('training_1.csv', x_dot, x_dot_predicted_best)

hyperparameter_history = np.array(hyperparameter_history)
# Plot evolution of hyperparameters (degree & n_frequencies)
plt.figure()
plt.plot(hyperparameter_history[:, 0], label='degree')
plt.plot(hyperparameter_history[:, 1], label='n_frequencies')
plt.xlabel('Iteration')
plt.ylabel('Hyperparameter Value')
plt.legend()
plt.title('Evolution of Hyperparameters (degree & n_frequencies)')
plt.savefig('out/hyperparams_degree_and_n_frequencies.png', bbox_inches='tight')
plt.show()

# Plot evolution of hyperparameters (lambda_val)
plt.figure()
plt.plot(hyperparameter_history[:, 2], label='lambda_val')
plt.xlabel('Iteration')
plt.ylabel('Hyperparameter Value')
plt.legend()
plt.title('Evolution of Hyperparameters (lambda_val)')
plt.savefig('out/hyperparams_lambda_val.png', bbox_inches='tight')
plt.show()

# Plot evolution of hyperparameters (threshold)
plt.figure()
plt.plot(hyperparameter_history[:, 3], label='threshold')
plt.xlabel('Iteration')
plt.ylabel('Hyperparameter Value')
plt.legend()
plt.title('Evolution of Hyperparameters (threshold)')
plt.savefig('out/hyperparams_threshold.png', bbox_inches='tight')
plt.show()

# Plot evolution of error
plt.figure()
plt.plot(sorted(error_history, reverse=True), label='Error')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.legend()
plt.title('Evolution of Error')
plt.show()

# Plot EEG data
for name, data in data_dict.items():
    plt.figure(figsize=(12, len(eeg_channels)))
    for j, eeg_data in enumerate(data):
        plt.plot(t_columns, data[j], label=f'{j + 1:00}: {eeg_channels[j]}')

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'EEG Data from {name}')
    plt.legend(loc='upper right')
    plt.savefig(f'out/data_{name}.png', bbox_inches='tight')
    plt.show()
