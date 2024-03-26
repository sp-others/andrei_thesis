import numpy as np
import pysindy as ps
from sklearn.metrics import mean_squared_error

np.random.seed(42)
count = 1000
t = np.linspace(0, 1, count)
x = np.linspace(-np.pi / 2, np.pi / 2, num=count) # problem HERE
x = np.linspace(-np.pi / 2, np.pi / 2, num=count) + np.random.normal(scale=0.1, size=count)
# x = np.sin(t) # still gives warning
# x = np.sin(t) + np.random.normal(scale=0.1, size=count)  # Adding noise to sin function
data = x.reshape(-1, 1)

scores = set()

param_range = range(2, 101, 10)
for param1 in param_range:
    for param2 in param_range:
        feature_lib = ps.PolynomialLibrary(degree=int(param1)) + ps.FourierLibrary(n_frequencies=int(param2))
        model = ps.SINDy(
            differentiation_method=(ps.FiniteDifference(order=2)),
            feature_library=feature_lib,
            optimizer=ps.STLSQ(threshold=0.2),
            feature_names=["x"])
        model.fit(data, t=t)
        # print(f'parametrii folositi in model sunt acestea{param1, param2}\n')
        score = model.score(data, metric=mean_squared_error)
        scores.add(score)
        print(f'param1: {param1}, param2: {param2} score: {score}')

print(f'len(scores): {len(scores)}')
