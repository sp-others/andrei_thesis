# Licenta
https://github.com/AndreiRaceanu/Bachelor-Thesis


## thesis chatgpt example prompt:
```
use bayesian optimization (using pygpgo) to find the best approximation of a derivative for a data set
the data set consists of 2 files of training data, which are matrices of real numbers, as comma separated values
the feature library used by sindy should be composed of PolynomialLibrary of degree param1 and of FourierLibrary with n_frequencies equal to param2
the optimizer should use OSQP with a given lambda and threshold
for the feature library take the bias into consideration
```


##  curent problem
de ce current eval, dat fiind ca reprezinta eroarea dintre derivata reala si cea aproximata, nu tinde catre 0, 
avand in vedere ca am rezolvat problema derivatelor, iar acum amandoua sunt aproape identice.



https://pysindy.readthedocs.io/en/latest/examples/2_introduction_to_sindy/example.html
https://pygpgo.readthedocs.io/en/latest/

https://github.com/josejimenezluna/pyGPGO/tree/master/pyGPGO

## python environment (linux)
```
python -m venv venv
source venv/bin/activate # Windows: .\venv\Scripts\activate

pip install ipykernel
python -m ipykernel install --user --name=andrei_licenta

deactivate
# open a new terminal
python -m notebook
# switch kernel to andrei_licenta
# install dependencies

# uninstalling an environment (kernel)
jupyter kernelspec uninstall andrei_licenta
```