"""
do a pysindy differntation with the  feature_library combining
PolynomialLibrary(degree=int(param1)) and FourierLibrary(n_frequencies=int(param2)

param1 and param2 takes values from 2 to 100 with step 10
show the model score for each, with the metric=mean_squared_error
"""



import numpy as np
from sklearn.metrics import mean_squared_error
from pysindy import SINDy
from pysindy.feature_library import PolynomialLibrary, FourierLibrary

# Generate synthetic data for demonstration
np.random.seed(42)
t = np.linspace(0, 10, 100)
x = np.sin(t)
y = np.cos(t)
data = np.array([x, y]).T

# Define parameter ranges
param_range = range(2, 101, 20)  # param1 and param2 values from 2 to 100 with step 10

# Iterate over parameter combinations
for param1 in param_range:
    for param2 in param_range:
        # print(f"Parameters: param1={param1}, param2={param2}")

        # Create feature library combining PolynomialLibrary and FourierLibrary
        feature_library = PolynomialLibrary(degree=param1) + FourierLibrary(n_frequencies=param2)

        # Initialize SINDy estimator
        model = SINDy(feature_names=["x", "y"], feature_library=feature_library)

        # Fit the model to the data
        model.fit(data, t=t)

        # Predict derivatives
        dxdt_pred = model.predict(data)

        # Compute MSE between true derivatives and predicted derivatives
        # mse = mean_squared_error(data, dxdt_pred)

        print(f"Parameters: param1={param1}, param2={param2} Model Score (MSE): {model.score(data, metric=mean_squared_error)}")
