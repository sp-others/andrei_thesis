# Licenta
https://github.com/AndreiRaceanu/Bachelor-Thesis

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