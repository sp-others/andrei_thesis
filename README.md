# Licenta
https://github.com/AndreiRaceanu/Bachelor-Thesis

##  curent problem
make model.score() return different results for different (param1, param2) values
at this point `chat_gpt_example.py` and `test_model.py` differ only in x

discovery: `UserWarning: Sparsity parameter is too big (0.2) and eliminated all coefficients` is caused by `optimizer=ps.STLSQ(threshold=0.2)`

original data with seed 42: len(scores): 2
x from chatgpt w/  noise: len(scores): 93
x from chatgpt w/o noise: len(scores): 34
original data with seed 42 w/ noise: len(scores): 80


np.linspace(-np.pi / 2, np.pi / 2, num=count) 
TIME = np.linspace(0, 1, count)

f(0) = a * 0 + b = -pi/2
f(1) = a * 1 + b = pi/2

b = -pi/2

a + b = pi/2 
<=> a = pi/2 - b
<=> a = pi/2 - (-pi/2)
<=> a = 2 * pi/2
<=> a = pi
=> f(x) = pi*x - pi/2
https://www.desmos.com/calculator/osnqrqnsor



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