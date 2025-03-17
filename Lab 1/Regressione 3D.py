#!/usr/bin/env python
# coding: utf-8

# In[42]:


import matplotlib.pyplot as plt#import delle librerie
import csv
from scipy import stats
from scipy.stats import linregress
import numpy as np
import sklearn as sl
from sklearn import linear_model as lm
from sklearn.linear_model import LinearRegression
filename = "data/km_year_power_price.csv" #dichiaro il file
linee = csv.reader(open(filename, newline=''), delimiter= ',') #apro il file e lo metto in un iteratore
dataset = list(linee) #converto l'iteratore in una lista;
data = list(dataset[1:])#tolgo la prima riga che non serve
X = np.array([[float(row[0]), float(row[1]), float(row[2])] for row in data]) #importo i dati per fare la regressione, importando le simgole colonne
Y = np.array([float(row[3]) for row in data])
model = LinearRegression()#uso LinearRegression perchè mim permette di usare vettori a dimensioni multiple
model.fit(X,Y)
nuovo_esempio = np.array([[50000, 2018, 110]])  # Esempio di dati nuovi
# Fare una previsione
previsione = model.predict(nuovo_esempio)
print(f"Prezzo stimato: {previsione[0]}")
#per stampare il grafico serve un plotter 3d della libreria axis





# In[ ]:




