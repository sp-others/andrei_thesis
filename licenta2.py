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
from pyGPGO.acquisition import Acquisition
from pyGPGO.covfunc import squaredExponential
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pysindy import SINDy, PolynomialLibrary, FourierLibrary, STLSQ
from sklearn.metrics import mean_squared_error

# region tweak-able constants
EMOTIONS = ['3_fericire']  # ['0_neutru', '1_tristete', '2_teama', '3_fericire']
CHANNELS = ["F5", "FC1", "P5", "CP1", "P4", "PO8", "FP2", "FC6", "FZ", "PZ"]
DATA_WIDTH = 5  # number of columns used from the csv file
NR_TRAINING_SAMPLES = 3
NR_VALIDATION_SAMPLES = 2

STLSQ_NORMALIZE_COLUMNS = True
USE_NEGATIVE_ERROR = True
ALPHA = 1

degree_bounds = ('int', [2, 10])
n_frequencies_bounds = ('int', [2, 10])
threshold_bounds = ('cont', [0, 10 ** -5])
lambda_bounds = ('cont', [0, 1e-20])

GPGO_ITERATIONS = 10
GPGO_INIT_EVALS = 3
CPU_CORES_FOR_GPGO = int(os.getenv('CPU_CORES_FOR_GPGO', 4))

PLOTS_DIR = 'out'
PLT_FIG_SIZE = (12, 10)
SHOW_PLOTS = False  # whether to show the plots interactively or not (recommend to use False when displaying many plots)
# endregion


# region computed properties
nr_samples = NR_TRAINING_SAMPLES + NR_VALIDATION_SAMPLES

error_sign = -1 ** USE_NEGATIVE_ERROR
# endregion


out_subdir = None
graph_name_prefix = None

# Initialize history storage
emotion_params_history = []
emotion_error_history = []
matrix_params_history = []
matrix_error_history = []


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
        return fit(x, params)[0]

    return objective


def fit(x, params: Params, save_history=True):
    poly_library = PolynomialLibrary(degree=int(params.degree), include_bias=True)
    fourier_library = FourierLibrary(n_frequencies=int(params.n_frequencies))
    feature_library = poly_library + fourier_library
    optimizer = STLSQ(threshold=params.threshold, alpha=params.lambda_val, normalize_columns=STLSQ_NORMALIZE_COLUMNS)
    model = SINDy(feature_library=feature_library, optimizer=optimizer)
    model.fit(x, t=t)
    return compute_error_and_derivatives(model, x, params, save_history)


def compute_error_and_derivatives(model, x, params: Params, save_history=True):
    x_dot_predicted = model.predict(x)
    x_dot = model.differentiate(x, t=1)
    # model.print()
    unsigned_error = model.score(x, metric=mean_squared_error) + ALPHA * model.complexity
    error = error_sign * unsigned_error
    if all(equation == '0.000' for equation in model.equations()):
        error = error_sign * 10 ** 4

    # Store hyperparameters and error for plotting
    if save_history:
        global emotion_params_history, emotion_error_history, matrix_params_history, matrix_error_history
        emotion_params_history.append(params.to_tuple())
        emotion_error_history.append(error)
        matrix_params_history.append(params.to_tuple())
        matrix_error_history.append(error)
    return error, x_dot, x_dot_predicted


def run_gpgo(matrix) -> Tuple[OrderedDict[str, Union[int, float]], float]:
    gpgo = GPGO(surogate, acq, get_objective_function(matrix), param_bounds, n_jobs=CPU_CORES_FOR_GPGO)

    # Run Bayesian Optimization
    start_time = datetime.datetime.now().isoformat()
    start = time.time()
    print(start_time)
    gpgo.run(max_iter=GPGO_ITERATIONS, init_evals=GPGO_INIT_EVALS)
    end_time = datetime.datetime.now().isoformat()
    end = time.time()

    print(f'GPGO with {GPGO_ITERATIONS} iterations ran from')
    print(start_time)
    print('to')
    print(end_time)
    print(f'Total time: {end - start} seconds')

    # noinspection PyTypeChecker
    return gpgo.getResult()


def plot_data(matrix_name: str, matrix_to_plot):
    plt.figure(figsize=PLT_FIG_SIZE)
    for i, eeg_data in enumerate(matrix_to_plot):
        plt.plot(t, matrix_to_plot[i], label=f'{i + 1:00}: {CHANNELS[i]}')

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'EEG Data from {matrix_name}')
    plt.legend(loc='upper right')
    save_plot('1_data')


def plot_derivative_and_channel_comparison(file_name, computed_derivative_, predicted_derivative_):
    # plot the 2 derivatives, fully
    for derivative_type, derivative in {'computed': computed_derivative_, 'predicted': predicted_derivative_}.items():
        plt.figure(figsize=PLT_FIG_SIZE)
        plt.plot(t, derivative.T)
        plt.xlabel('Time')
        plt.ylabel('Derivative')
        plt.title(f'{derivative_type} Derivative for {file_name}')
        save_plot(f'2_derivative_{derivative_type}')
    # plot a plot for each set of channels from both derivatives
    for i, channel in enumerate(CHANNELS):
        plt.figure(figsize=PLT_FIG_SIZE)
        plt.plot(t, computed_derivative_[i], 'k', label=f'{channel} computed derivative')
        plt.plot(t, predicted_derivative_[i], 'r--', label=f'{channel} predicted derivative')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'predicted vs computed for data {file_name} for channel {channel}')
        plt.legend(loc='upper right')
        save_plot(f'3_derivative_comparison_{channel}')


def plot_hyperparams_and_error():
    level_param_history_error_history_tuples = [
        ('matrix', matrix_params_history, matrix_error_history),
        ('emotion', emotion_params_history, emotion_error_history),
    ]
    for i, (level, params_history, error_history) in enumerate(level_param_history_error_history_tuples):
        params_history_as_np = np.array(params_history)
        plot_sets = [
            ('params_degree_and_n-frequencies',
             [('degree', params_history_as_np[:, 0]), ('n_frequencies', params_history_as_np[:, 1])]),
            ('params_threshold', [('threshold', params_history_as_np[:, 2])]),
            ('params_lambda_val', [('lambda_val', params_history_as_np[:, 3])]),
            ('error', [('error', error_history)])
        ]
        for y_label, plot_set in plot_sets:
            plt.figure(figsize=PLT_FIG_SIZE)
            for label, data in plot_set:
                plt.plot(data, label=label)
            plt.xlabel('Iteration')
            plt.ylabel(y_label)
            plt.legend()
            plt.title(f'Evolution of {y_label} at {level} level')
            save_plot(f'4.{i + 1}_{level}_{y_label}')


def save_plot(name: str):
    plt.savefig(os.path.join(out_subdir, f'{graph_name_prefix}_{name}.png'), bbox_inches='tight')
    plt.show() if SHOW_PLOTS else plt.close()


# make sure the out dir exists
for emotion in EMOTIONS:
    os.makedirs(os.path.join(PLOTS_DIR, emotion), exist_ok=True)

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

t = np.linspace(0, DATA_WIDTH - 1, DATA_WIDTH, dtype=int)
t_channels = np.linspace(0, len(CHANNELS) - 1, len(CHANNELS), dtype=int)

for emotion_i, emotion in enumerate(EMOTIONS):
    print(f'Running for emotion {emotion_i + 1}/{len(EMOTIONS)}: {emotion}')

    out_subdir = os.path.join(PLOTS_DIR, emotion)
    emotion_params_history = []
    emotion_error_history = []

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

        matrix_params_history = []
        matrix_error_history = []
        result = run_gpgo(transposed_matrix)
        list_of_name_and_result.append((sample_name, result))

        graph_name_prefix = f'1_training_{sample_name_i + 1:00}_{sample_name}'
        params_ = Params.from_tuple_list(result[0])
        _, computed_derivative, predicted_derivative = fit(transposed_matrix, params_, False)

        plot_data(sample_name, transposed_matrix)
        plot_derivative_and_channel_comparison(sample_name, computed_derivative.T, predicted_derivative.T)
        plot_hyperparams_and_error()

    print('Best results:')
    print(*list_of_name_and_result, sep='\n')
    list_of_name_and_result_sorted = sorted(list_of_name_and_result, key=lambda x: x[1][1], reverse=USE_NEGATIVE_ERROR)
    print('Best results sorted:')
    print(*list_of_name_and_result_sorted, sep='\n')

    best_result = list_of_name_and_result_sorted[0][1]
    print('Best result:', best_result)

    best_params = Params.from_tuple_list(best_result[0])

    for sample_name_i, sample_name in enumerate(validation_samples):
        print(f'Running for validation sample {sample_name_i + 1}/{len(validation_samples)}: {sample_name}')
        transposed_matrix = read_data(f'{emotion}/{sample_name}', DATA_WIDTH, channel_index_list)

        graph_name_prefix = f'2_validation_{sample_name_i + 1:00}_{sample_name}'
        _, computed_derivative, predicted_derivative = fit(transposed_matrix, best_params, False)

        plot_data(sample_name, transposed_matrix)
        plot_derivative_and_channel_comparison(sample_name, computed_derivative.T, predicted_derivative.T)

    print()

print('Done')
