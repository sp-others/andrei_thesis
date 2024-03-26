import numpy as np
import matplotlib.pyplot as plst
from mpl_toolkits import mplot3d
import pysindy as ps
import random
import pyGPGO
from pysindy import PolynomialLibrary, FourierLibrary
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.covfunc import squaredExponential
from pyGPGO.acquisition import Acquisition
from pyGPGO.GPGO import GPGO
from sklearn.metrics import mean_squared_error

#importarea modulelor necesare prelucrarii datelor, optimizarii bayesiene si gasirii ecuatiilor sistemului dinamic



data_sine = np.linspace(-np.pi/2, np.pi/2, num=1000)



def err(param1, param2):
    _model, x_data = get_model_and_data(param1, param2)
    alpha = 1
    score = _model.score(x_data, metric=mean_squared_error) + alpha * _model.complexity
    print(f'parametrii folositi in functie sunt acestea {param1,param2} \n')
    return score

# variabila score modeleaza performantele aproximarii, unde _model.complexity este un numar ce reprezinta numarul
# de parametri non-zero


TIME = np.linspace(0, 1, 1000)

def get_model_and_data(param1, param2):

    x = np.array(data_sine)
    x_data = x
    differentiation_method = ps.FiniteDifference(order=2)
    poly_lib = ps.PolynomialLibrary(degree=int(param1))
    trig_lib = ps.FourierLibrary(n_frequencies=int(param2))
    custom_lib = poly_lib + trig_lib
    feature_library = custom_lib
    optimizer = ps.STLSQ(threshold=0.2)
    model = ps.SINDy(
        differentiation_method=differentiation_method,
        feature_library=custom_lib,
        optimizer=optimizer,
        feature_names=["x"])
    model.fit(x_data, t=TIME)
    print(f'parametrii folositi in model sunt acestea{param1,param2}\n')
    return model, x_data




cov = squaredExponential()
surogate = GaussianProcess(cov)
acq = Acquisition(mode = 'ExpectedImprovement')
params = {'param1' : ('int',[2,100]),
          'param2' : ('int',[2,100])}
np.random.seed(23)
gpgo = GPGO(surogate, acq, err,params)
gpgo.run(max_iter = 20,init_evals=5)
print(gpgo.GP.y)



from pysindy.utils import concat_sample_axis, drop_nan_samples
from pysindy.pysindy import _adapt_to_multiple_trajectories, _comprehend_and_validate_inputs


def get_x_dot_and_x_dot_predicted(_model, x, t=None, x_dot=None, u=None, multiple_trajectories=False, metric=err, **metric_kws):
    """this is copy-pasted from pysindy.pysindy.PySINDy.score"""
    if t is None:
        t = _model.t_default

    if not multiple_trajectories:
        x, t, x_dot, u = _adapt_to_multiple_trajectories(x, t, x_dot, u)
        multiple_trajectories = True
    x, x_dot, u = _comprehend_and_validate_inputs(
        x, t, x_dot, u, _model.feature_library
    )

    x_dot_predict = _model.predict(x, u, multiple_trajectories=multiple_trajectories)

    if _model.discrete_time and x_dot is None:
        x_dot_predict = [xd[:-1] for xd in x_dot_predict]

    x, x_dot = _model._process_multiple_trajectories(x, t, x_dot)

    x_dot = concat_sample_axis(x_dot)
    x_dot_predict = concat_sample_axis(x_dot_predict)

    x_dot, x_dot_predict = drop_nan_samples(x_dot, x_dot_predict)
    return x_dot, x_dot_predict


params_list = gpgo.GP.X[3:]
current_eval_list = gpgo.GP.y[3:]

t_list = list(range(1, len(params_list) + 1))
param1_list = [point[0] for point in params_list]
param2_list = [point[1] for point in params_list]


print(f't_list={t_list}')
print('\n')
print(f'point_param1_list={param1_list}')
print('\n')
print(f'point_param2_list={param2_list}')
print('\n')
print(f'current_eval={current_eval_list}')
print('\n')


def plot_subplot(x, y, label):
    subplot = plst.figure().add_subplot()
    subplot.plot(x, y, label = label)
    subplot.legend()

plot_subplot(t_list, param1_list, "param1")
plot_subplot(t_list, param2_list, "param2")
plot_subplot(t_list, current_eval_list, "current_eval")

best_eval = min(current_eval_list)
best_params = params_list[np.where(current_eval_list == best_eval)][0]
print(f'\nbest_params, best_eval={best_params, best_eval}')

model, x_data = get_model_and_data(best_params[0], best_params[1])
model.print() # prints (x)'

x_derivative_real = model.differentiate(x_data)
_, x_derivative_estimated = get_x_dot_and_x_dot_predicted(model, x_data) # NOTE that _ = x_derivative_real

derivative_subplot = plst.figure().add_subplot()
derivative_subplot.plot(TIME, x_derivative_real, label ="x_derivative_real")
derivative_subplot.plot(TIME, x_derivative_estimated, label ="x_derivative_estimated")
derivative_subplot.legend()

plst.show()






#sa verific ca param1 si param 2 se fedeaza la fiecare iteratie a optimizarii bayesiene
# optimizarea lui sindy si BO.
# set de date/aplicatie public data sets for ML/kagel