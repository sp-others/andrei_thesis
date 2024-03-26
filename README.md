# Licenta

##  curent problem
make model.score() return different results for different (param1, param2) values
at this point `chat_gpt_example.py` and `test_model.py` differ only in x

discovery: `UserWarning: Sparsity parameter is too big (0.2) and eliminated all coefficients` is caused by `optimizer=ps.STLSQ(threshold=0.2)`

original data with seed 42: len(scores): 2
x from chatgpt w/  noise: len(scores): 93
x from chatgpt w/o noise: len(scores): 34
original data with seed 42 w/ noise: len(scores): 80


https://github.com/AndreiRaceanu/Bachelor-Thesis



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