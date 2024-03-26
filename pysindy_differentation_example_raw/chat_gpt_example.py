"""
do a pysindy differntation with the  feature_library combining
PolynomialLibrary(degree=int(param1)) and FourierLibrary(n_frequencies=int(param2)

param1 and param2 takes values from 2 to 100 with step 10
show the model score for each, with the metric=mean_squared_error
"""

import numpy as np
import pysindy as ps
from pysindy import SINDy, FiniteDifference
from pysindy.feature_library import PolynomialLibrary, FourierLibrary
from sklearn.metrics import mean_squared_error

# Generate synthetic 1D data
np.random.seed(42)
count = 100
t = np.linspace(0, 10, count)
x = np.sin(t) + np.random.normal(scale=0.1, size=count)  # Adding noise to sin function
data = x.reshape(-1, 1)  # Reshape to make it 2D for PySINDy

scores = set()

param_range = range(2, 101, 10)  # param1 and param2 values from 2 to 100 with step 10
for param1 in param_range:
    for param2 in param_range:
        feature_library = PolynomialLibrary(degree=param1) + FourierLibrary(n_frequencies=param2)
        model = SINDy(
            differentiation_method=FiniteDifference(order=2),
            feature_library=feature_library,
            optimizer=ps.STLSQ(threshold=0.2),
            feature_names=["x"])
        model.fit(data, t=t)

        score = model.score(data, metric=mean_squared_error)
        scores.add(score)
        print(f"param1={param1}, param2={param2} score: {score}")

print(f'len(scores): {len(scores)}')
