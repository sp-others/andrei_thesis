import datetime
import os
import random
import time
from enum import Enum
from typing import List, Union, Tuple, OrderedDict

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyGPGO.GPGO import GPGO
from pyGPGO.covfunc import squaredExponential
from pysindy import SINDy, PolynomialLibrary, FourierLibrary, STLSQ
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.acquisition import Acquisition
from sklearn.metrics import mean_squared_error

# region tweak-able constants
EMOTIONS = ['0_neutru', '1_tristete', '2_teama', '3_fericire']
CHANNELS = ["F5", "FC1", "P5", "CP1", "P4", "PO8", "FP2", "FC6", "FZ", "PZ"]
DATA_WIDTH = 5  # number of columns used from the csv file
NR_TRAINING_SAMPLES = 2
NR_VALIDATION_SAMPLES = 1

USE_NEGATIVE_ERROR = True
ALPHA = 1

degree_bounds = ('int', [2, 10])
n_frequencies_bounds = ('int', [2, 9])
threshold_bounds = ('cont', [0, 0.1])
lambda_bounds = ('cont', [0, 1e-12])

GPGO_ITERATIONS = 10
CPU_CORES_FOR_GPGO = int(os.getenv('CPU_CORES_FOR_GPGO', 4))

PLOTS_DIR = 'out'
SHOW_PLOTS = False  # whether to show the plots interactively or not (recommend to use False when displaying many plots)
# endregion

# region computed properties
nr_samples = NR_TRAINING_SAMPLES + NR_VALIDATION_SAMPLES

error_sign = -1 ** USE_NEGATIVE_ERROR
# endregion

# Initialize history storage
hyperparameter_history = []
error_history = []


class Params:
    class Names(str, Enum):
        # declaration order is important for from_tuple_list() and other usages of Params.Names
        DEGREE = 'degree'
        N_FREQUENCIES = 'n_frequencies'
        THRESHOLD = 'threshold'
        LAMBDA = 'lambda_val'

    def __init__(self, degree: int, n_frequencies: int, threshold: float, lambda_val: float):
        self.degree = degree
        self.n_frequencies = n_frequencies
        self.threshold = threshold
        self.lambda_val = lambda_val

    @staticmethod
    def from_tuple_list(params: OrderedDict[str, int | float]):
        return Params(*[params.get(name.value) for name in Params.Names])

    def to_tuple(self):
        return self.degree, self.n_frequencies, self.threshold, self.lambda_val


def read_channel_indices(file_path, channel_list):
    """
    :return: a dictionary that maps the channel name to its index (0-based) in the data file
    """
    channel_indices = {}
    with open(file_path, 'r') as file:
        for index, channel_name in enumerate(file.read().splitlines()):
            if channel_name in channel_list:
                channel_indices[channel_name] = index
    return channel_indices


# Function to read the data
def read_data(filename, last_column_number=None, use_rows: List[int] = None):
    """
    Read data from a CSV file, transposed
    :param filename: the csv file that contains the data
    :param last_column_number: if None, all columns are used, else only the first last_column_number columns are used
    :param use_rows: the rows to use from the csv file (0-indexed)
    :return:
    """
    used_columns = None if last_column_number is None else list(range(last_column_number))
    skip_rows = None if use_rows is None else lambda x: x not in use_rows
    return pd.read_csv(filename, header=None, usecols=used_columns, skiprows=skip_rows).T


# Define the objective function
def get_objective_function(x):
    def objective(degree, n_frequencies, threshold, lambda_val):
        params = Params(degree, n_frequencies, threshold, lambda_val)
        return get_error_model_and_derivatives(x, params)[0]

    return objective


def get_error_model_and_derivatives(x, params: Params, save_metadata=True):
    model = get_fitted_model(x, params)
    return get_error_and_derivatives(model, x, params, save_metadata)


def get_error_and_derivatives(model, x, params: Params, save_metadata=True):
    x_dot_predicted = model.predict(x)
    x_dot = model.differentiate(x, t=1)
    # model.print()
    unsigned_error = model.score(x, metric=mean_squared_error) + ALPHA * model.complexity
    error = error_sign * unsigned_error
    if all(equation == '0.000' for equation in model.equations()):
        error = error_sign * 10 ** 5

    # Store hyperparameters and error for plotting
    if save_metadata:
        global hyperparameter_history, error_history
        hyperparameter_history.append(params.to_tuple())
        error_history.append(error)
    return error, model, x_dot, x_dot_predicted


def get_fitted_model(x, params: Params):
    poly_library = PolynomialLibrary(degree=int(params.degree), include_bias=True)
    fourier_library = FourierLibrary(n_frequencies=int(params.n_frequencies))
    feature_library = poly_library + fourier_library
    model = SINDy(feature_library=feature_library, optimizer=STLSQ(threshold=params.threshold, alpha=params.lambda_val))
    return model.fit(x, t=t)


def run_gpgo_and_get_result(matrix) -> Tuple[OrderedDict[str, Union[int, float]], float]:
    gpgo = GPGO(surogate, acq, get_objective_function(matrix), param_bounds, n_jobs=CPU_CORES_FOR_GPGO)

    # Run Bayesian Optimization
    start_time = datetime.datetime.now().isoformat()
    start = time.time()
    print(start_time)
    gpgo.run(max_iter=GPGO_ITERATIONS)
    end_time = datetime.datetime.now().isoformat()
    end = time.time()

    print(f'GPGO with {GPGO_ITERATIONS} iterations ran from')
    print(start_time)
    print('to')
    print(end_time)
    print(f'Total time: {end - start} seconds')

    # noinspection PyTypeChecker
    return gpgo.getResult()


def plot_data():
    for name, data in data_dict.items():
        plt.figure(figsize=(12, len(CHANNELS)))
        for j, eeg_data in enumerate(data):
            plt.plot(t_columns, data[j], label=f'{j + 1:00}: {CHANNELS[j]}')

        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'EEG Data from {name}')
        plt.legend(loc='upper right')
        plt.savefig(f'out/data_{name}.png', bbox_inches='tight')
        plt.show() if SHOW_PLOTS else plt.close()


def plot_derivatives(file_name, computed_derivative, predicted_derivative):
    global plot_derivatives_runs
    runs = plot_derivatives_runs
    # plot the 2 derivatives, fully
    for derivative_type, derivative in {'predicted': predicted_derivative, 'computed': computed_derivative}.items():
        plt.figure()
        plt.plot(t, derivative)
        plt.xlabel('Time')
        plt.ylabel('Derivative')
        plt.title(f'{derivative_type} Derivative for {file_name}')
        plt.savefig(f'out/derivative{runs}_{file_name}_full_{derivative_type}.png', bbox_inches='tight')
        plt.show() if SHOW_PLOTS else plt.close()
    # plot a plot for each set of channels from both derivatives
    for i, channel in enumerate(CHANNELS):
        plt.figure(figsize=(12, len(CHANNELS)))
        plt.plot(t_columns, computed_derivative[i], 'k', label=f'{channel} computed derivative')
        plt.plot(t_columns, predicted_derivative[i], 'r--', label=f'{channel} predicted derivative')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'predicted vs computed for data {file_name} for channel {channel}')
        plt.legend(loc='upper right')
        plt.savefig(f'out/derivative{runs}_{file_name}_{channel}.png', bbox_inches='tight')
        plt.show() if SHOW_PLOTS else plt.close()


def plot_hyperparams_and_error():
    global hyperparameter_history, plot_hyperparams_and_error_runs
    hyperparameter_history_as_np_array = np.array(hyperparameter_history)
    plot_hyperparams_and_error_runs += 1
    savefig_prefix = f'out/hyperparams_and_error_{plot_hyperparams_and_error_runs}'

    # Plot evolution of hyperparameters (degree & n_frequencies)
    plt.figure()
    plt.plot(hyperparameter_history_as_np_array[:, 0], label='degree')
    plt.plot(hyperparameter_history_as_np_array[:, 1], label='n_frequencies')
    plt.xlabel('Iteration')
    plt.ylabel('Hyperparameter Value')
    plt.legend()
    plt.title('Evolution of Hyperparameters (degree & n_frequencies)')
    plt.savefig(f'{savefig_prefix}_degree_and_n_frequencies.png', bbox_inches='tight')
    plt.show() if SHOW_PLOTS else plt.close()
    # Plot evolution of hyperparameters (lambda_val)
    plt.figure()
    plt.plot(hyperparameter_history_as_np_array[:, 2], label='lambda_val')
    plt.xlabel('Iteration')
    plt.ylabel('Hyperparameter Value')
    plt.legend()
    plt.title('Evolution of Hyperparameters (lambda_val)')
    plt.savefig(f'{savefig_prefix}_lambda_val.png', bbox_inches='tight')
    plt.show() if SHOW_PLOTS else plt.close()
    # Plot evolution of hyperparameters (threshold)
    plt.figure()
    plt.plot(hyperparameter_history_as_np_array[:, 3], label='threshold')
    plt.xlabel('Iteration')
    plt.ylabel('Hyperparameter Value')
    plt.legend()
    plt.title('Evolution of Hyperparameters (threshold)')
    plt.savefig(f'{savefig_prefix}_threshold.png', bbox_inches='tight')
    plt.show() if SHOW_PLOTS else plt.close()
    # Plot evolution of error
    plt.figure()
    plt.plot(sorted(error_history, reverse=True), label='Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Evolution of Error')
    plt.savefig(f'out/error{plot_hyperparams_and_error_runs}_threshold.png', bbox_inches='tight')
    plt.show() if SHOW_PLOTS else plt.close()


# make sure the out dir exists
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

"""
if 429 Too Many Requests is thrown by PyCharm when plotting the graphs, then 
https://youtrack.jetbrains.com/issue/PY-43687/Problems-with-many-plots-in-scientific-view#focus=Comments-27-6266042.0-0
"""

# to avoid Tcl_AsyncDelete: async handler deleted by the wrong thread
# (see more at https://github.com/matplotlib/matplotlib/issues/27713)
if not SHOW_PLOTS:
    matplotlib.use('Agg')

plot_hyperparams_and_error_runs = 0
plot_derivatives_runs = 0

print(f'Using {CPU_CORES_FOR_GPGO} CPU cores for GPGO')
channel_to_index = read_channel_indices('Channel Order.csv', CHANNELS)
print(f'channel_to_index: {channel_to_index}')
print()
channel_index_list = list(channel_to_index.values())

# Define the parameter space
param_bounds = {
    Params.Names.DEGREE.value: degree_bounds,
    Params.Names.N_FREQUENCIES.value: n_frequencies_bounds,
    Params.Names.THRESHOLD.value: threshold_bounds,
    Params.Names.LAMBDA.value: lambda_bounds
}

# Set up Bayesian Optimization
cov = squaredExponential()
surogate = GaussianProcess(cov)
acq = Acquisition(mode='ExpectedImprovement')

t = np.linspace(1, DATA_WIDTH, DATA_WIDTH, dtype=int)

for emotion_i, emotion in enumerate(EMOTIONS):
    print(f'Running for emotion {emotion_i + 1}/{len(EMOTIONS)}: {emotion}')

    global out_subdir
    out_subdir = os.path.join(PLOTS_DIR, emotion)

    data_files = os.listdir(emotion)

    if nr_samples > len(data_files):
        print(f'Not enough data files for emotion {emotion}. Requested {nr_samples}, available {len(data_files)}. '
              f'Skipping...')
        continue

    chosen_samples: List[str] = random.sample(data_files, nr_samples)
    training_samples, validation_samples = chosen_samples[:NR_TRAINING_SAMPLES], chosen_samples[NR_TRAINING_SAMPLES:]
    print(f'Training samples: {training_samples}')
    print(f'Validation samples: {validation_samples}')

    list_of_name_and_result = []
    for sample_name_i, sample_name in enumerate(training_samples):
        print(f'Running for training sample {sample_name_i + 1}/{len(training_samples)}: {sample_name}')
        transposed_matrix = read_data(f'{emotion}/{sample_name}', DATA_WIDTH, channel_index_list)
        result = run_gpgo_and_get_result(transposed_matrix)
        list_of_name_and_result.append((sample_name, result))

        # TODO: plot_matrix(sample_name, transposed_matrix)
        # TODO: plot_derivative_and_channel_comparison(sample_name, transposed_matrix, result)
        # TODO: plot_hyperparams_and_error()  # TODO: print at every matrix in the emotion, but reset at every emotion

    print('Best results:')
    print(*list_of_name_and_result, sep='\n')
    list_of_name_and_result_sorted = sorted(list_of_name_and_result, key=lambda x: x[1][1], reverse=USE_NEGATIVE_ERROR)
    print('Best results sorted:')
    print(*list_of_name_and_result_sorted, sep='\n')

    best_result = list_of_name_and_result_sorted[0][1]
    print('Best result:', best_result)

    best_params = Params.from_tuple_list(best_result[0])

    print()

file1 = 'training_1.csv'
file2 = 'training_2.csv'
file3 = 'validation_1.csv'
file_names = [file1, file2, file3]
data_dict = {file_name: read_data(file_name, DATA_WIDTH) for file_name in file_names}
data1 = data_dict[file1]
data2 = data_dict[file2]
data3 = data_dict[file3]
z = read_data('3_fericire/cz_eeg3.txt', DATA_WIDTH, channel_index_list)

# Assuming time vector t and derivative x_dot are known
# For the sake of this example, let's create synthetic ones
t_columns = np.linspace(1, DATA_WIDTH, DATA_WIDTH, dtype=int)
# x = data1
# x_dot = np.gradient(data1, axis=0)  # Replace this with computed derivative if available


# Run gpgo and get the best parameters
best_result1 = run_gpgo_and_get_result(data1)

print("Best Parameters:")
print(best_result1)

# Refine the model with the best parameters
best_params1 = Params.from_tuple_list(best_result1[0])

data1_error, model1, data1_x_dot, data1_x_dot_predicted = get_error_model_and_derivatives(data1, best_params1,
                                                                                          save_metadata=False)

print("Best Model Predictions:")
print(data1_x_dot_predicted)

plot_data()
plot_hyperparams_and_error()
plot_derivatives(file1, data1_x_dot, data1_x_dot_predicted)
data2_error, model2, data2_x_dot, data2_x_dot_predicted = get_error_and_derivatives(model1, data2, best_params1)
plot_hyperparams_and_error()
plot_derivatives(file2, data2_x_dot, data2_x_dot_predicted)
print("plotted graphs after 1st GPGO run")

best_result2 = run_gpgo_and_get_result(data2)
best_params2 = Params.from_tuple_list(best_result2[0])

best_error1 = best_result1[1]
best_error2 = best_result2[1]
best_error = None

if best_error1 < best_error2:
    best_result = best_result1
    print("Best parameters are from the 1st run")
else:
    best_result = best_result2
    print("Best parameters are from the 2nd run")

best_params = Params.from_tuple_list(best_result[0])

data1_error, model1, data1_x_dot, data1_x_dot_predicted = get_error_model_and_derivatives(data1, best_params,
                                                                                          save_metadata=False)
plot_derivatives_runs += 1
plot_hyperparams_and_error()
plot_derivatives(file1, data1_x_dot, data1_x_dot_predicted)

data2_error, model2, data2_x_dot, data2_x_dot_predicted = get_error_model_and_derivatives(data2, best_params,
                                                                                          save_metadata=False)

plot_derivatives(file2, data2_x_dot, data2_x_dot_predicted)
print("plotted graphs after 2nd GPGO run")

data3_error, model3, data3_x_dot, data3_x_dot_predicted = get_error_model_and_derivatives(data3, best_params)

plot_hyperparams_and_error()
plot_derivatives(file3, data3_x_dot, data3_x_dot_predicted)
print()
