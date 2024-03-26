"""
do a pysindy differntation with the  feature_library combining
PolynomialLibrary(degree=int(param1)) and FourierLibrary(n_frequencies=int(param2)

param1 and param2 takes values from 2 to 100 with step 10
show the model score for each, with the metric=mean_squared_error
"""



import numpy as np
from sklearn.metrics import mean_squared_error
from pysindy import SINDy, FiniteDifference
from pysindy.feature_library import PolynomialLibrary, FourierLibrary

# Generate synthetic 1D data
np.random.seed(42)
t = np.linspace(0, 10, 100)
x = np.sin(t) + np.random.normal(scale=0.1, size=100)  # Adding noise to sin function
data = x.reshape(-1, 1)  # Reshape to make it 2D for PySINDy

# Define parameter ranges
param_range = range(2, 101, 10)  # param1 and param2 values from 2 to 100 with step 10

# Iterate over parameter combinations
for param1 in param_range:
    for param2 in param_range:
        # print(f"Parameters: param1={param1}, param2={param2}")

        # Create feature library combining PolynomialLibrary and FourierLibrary
        feature_library = PolynomialLibrary(degree=param1) + FourierLibrary(n_frequencies=param2)

        # Initialize SINDy estimator
        model = SINDy(differentiation_method=FiniteDifference(order=2), feature_names=["x"], feature_library=feature_library)

        # Fit the model to the data
        model.fit(data, t=t)

        # Predict derivatives
        dxdt_pred = model.predict(data)

        # Compute MSE between true derivatives and predicted derivatives
        # mse = mean_squared_error(data, dxdt_pred)

        print(f"Parameters: param1={param1}, param2={param2} Model Score (MSE): {model.score(data, metric=mean_squared_error)}")
