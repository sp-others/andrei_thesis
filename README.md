# Licenta
https://github.com/AndreiRaceanu/Bachelor-Thesis

you need to add the 'Channel Order.csv' file in the root directory of the project

you can tweak the following in the 'tweak-able constants' region
1. the emotions it runs for
2. the channels it uses (it automatically gets those channel from the whole 62-channel data files)
3. the number of columns used
4. the number of training samples and validation samples (then it chooses the files randomly)
5. whether STLSQ should normalize columns (when True it seems to solve 'pysindy/optimizers/stlsq.py:191: UserWarning: Sparsity parameter is too big'. Found at https://github.com/dynamicslab/pysindy/issues/93)
6. whether to use negative sign when computing error (and this is taken into account when computing the best parameters)
7. ALHPA, as before
8. the bounds for the hyperparameters
9. the number of iterations for GPGO
10. the number of init iterations for GPGO
11. the number of cores GPGO is using

data from https://bcmi.sjtu.edu.cn/~seed/seed-iv.html

```bash
sed -n '3p;7p;10p;18p;22p;36p;43p;46p;48p;57p' 3_fericire/cz_eeg3.txt | cut -d, -f1-5
\grep -n "40.3821468353271,40.2629375457764,35.3753566741943,32.4845314025879,36.2694263458252" 3_fericire/cz_eeg3.txt | cut -d, -f1-5
```

# TODO: 2024 06 23 00:46
1. remove out/ before remaking it

# TODO: 2024 06 22 22:04
1. ~~fix derivative plotting~~

# TODO: 2024 06 18 18:40

Observatie :  
In acest document, M_1 se refera la prima matrice de training din data_set, de 10 * nr de coloane computational fezabile  
In acest document. M_2 se refeta la a doua matrice de trainning din data_set, de 10 * nr de coloane computational fezabile  
prin derivata matricii M_n, ma refer la derivata numerica, reala, a celor 10 channeluri din matrice.  
se cere:

1. ~~grafic M_1~~
2. ~~GRafic M_2~~
3. ~~Grafic derivata Matricii M_1, reala.~~
4. ~~Grafic derivata Matricii M_2, reala.~~
5. ~~Grafic derivata aproximata a Matricii M_1.~~
6. ~~Grafic derivata aproximata a Matricii M_2.~~
7. ~~GRafic derivata aproximata vs derivata reala pentru 2 canale matricea 1.~~
8. ~~grafic derivata aproximata vs derivata reala pentru 2 canale matricea 2.~~
9. ~~Graficul erorii in timp, ordonata descrescator; Se iau erorile, se sorteaza descrescator, se afiseaza eroarea in functie de timp.~~
10. ~~Graficul evolutiei hiperparametrilor in timp pentru cei 2 hipr subunitari.~~
11. ~~Graficul pentru hiperparametrii > 1 pentru cei 2 hiperparametrii >1.~~
12. ~~!!!! pentru antrenarea lui M_2, trebuie sa se foloseasca printre parametrii, solutia optimala de la antrenarea lui M_1, adica tuplul jipeparametrilor pentru care eroarea era minima.----> folosesti SINDY pe setul de param cei mai buni la 1, obtii o eroare.~~
13. ~~!!!!dupa antrenarea cu M_2, se inlocuieste tuplyl parametrilor cei mai buni, cu noua solutie, doar ca de data asta,
in LISTA ERORILOR incluzi si eroarea obtinuta cu hiperparametrii de la 1.,  apoi sortezi iar descrescator erorile, incluzand si eroarea calculata de tine in manual.~~

    ~~iei tuplul de parametrii care au eroarea cea mai mica;~~
14. ~~Se face validation, se aproximeaza aceeasi dimensiunea din M_3, matricea validation, folosind DOAR hiperparametrii gasiti anterior.~~
15. ~~Se face graficul lui M_3.~~
16. ~~GRaficul lui M_3 derivat.~~
17. ~~Graficele de eroare, dintre M_3 derivat aproximat si real.~~
18. ~~Graficele cu 2 channele apxoimate si derivate din M_3.~~



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