# -*- coding: utf-8 -*-
#! /bin/python3

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nb_proc = comm.Get_size()

"""
Created on Mon May 16 14:24:13 2022

@author: mbaye
"""

"""
Cette partie contient le code pour le modèle de machine learning permettant d'estimer la qualité d'une 
divergence prédite par l'algo
"""
import numpy as np
import csv

import datetime as dt

import numpy as np
import csv
from GetRUssel3000 import orderByDataType, getEntryML
from Functions_algo import *
from Functions_algo2 import *

import math
from math import *
import xlrd
import requests
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram

from scipy.signal import find_peaks
import datetime as dt
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.dates as mdates
import pandas as pd
import requests
from stockstats import StockDataFrame as sdf
from sympy.combinatorics.permutations import Permutation


import yfinance as yf
import datetime as dt
#%%
# Récupération des données d'apprentissage


exec(open('Imports.py').read())
exec(open('Functions_algo.py').read())
exec(open('Functions_algo2.py').read())
#exec(open('GetRUssel3000.py').read())
"""
"""
#%%

""" 

Cette fonction renvoie les données (X, Y) du modèle de machine learning (Réseau 
de neurones, fôret aléatoire...)
indicesEntrprise est une liste de deux entiers: l'indice de début et l'indice
de fin des entreprises du russel3000 à récupérer

"""
def entryModeleML( indicesEntrprise, startTime='2020-12-14', endTime=dt.datetime.now(), fileAssetName='ticker_R3000.csv', type_data="data_app", longueur_contexte=15):
    
    (indiceDebutEntrprise, indiceFinEntreprise)=indicesEntrprise
    listR3000 = []

    fh = open(fileAssetName) # entreprise du russel3000
    reader = csv.reader(fh)
    for ligne in reader:
        listR3000.append(ligne[0])
    fh.close()


    nbVariables=longueur_contexte+5
    X = np.empty((0, nbVariables))  # inputs of the ML model
    #Y = np.empty((0))            # output ML modele
    nbErreur=0
    listeEntrepriseError=[]
    for i in range(indiceDebutEntrprise, indiceFinEntreprise):
      try:
        result_getEntry=getEntryML(listR3000[i],type_data=type_data, startTime=startTime, endTime=endTime, longueur_contexte=longueur_contexte)
        dataML=orderByDataType(result_getEntry, type_data=type_data, longueur_contexte=longueur_contexte) #  [liste_pente cours, liste_pente rsi, angle, liste_longueur div, liste_reversal (hauteur) ]
        X=np.concatenate( ( X, np.array(np.transpose(dataML))) , axis=0)
        
        """
        if type_data=="data_app":
            Y= np.concatenate( ( Y, np.array(dataML[-1])) )
        print( "Entreprise n°"+str(i-indiceDebutEntrprise) +" en cours ...")
        #dataML.append( orderByDataType(getEntryML(listR3000[i])) ) 
        """
      except ValueError:
        print( " Erreur sur l'entreprise "+str(i))
        nbErreur+=1
        listeEntrepriseError.append(i)


    print( "Nombre d'entreprises comportant des erreurs dans la détection de div "+str(nbErreur) )
    
    """
    if type_data=="data_app":
        return (X, Y)
    elif type_data=="data_test":
        return X
    """
    return X

def main():
    longueur_contexte=15
    nbEntrepriseTot=3000
     
    indicesEntrprise=[i for i in range(rank*nbEntrepriseTot//nb_proc, min(nbEntrepriseTot, (rank+1)*nbEntrepriseTot//nb_proc+1 ) )  ] # Indice de début et de fin des entreprises sélectionnées
    fileAssetName='ticker_R3000.csv'
    startTime='2020-12-14'
    endTime=dt.datetime.now()            #'2021-12-14'
    
    nbVariables=20
    dataML=np.empty((0, nbVariables))  # inputs of the ML model
    
    dataML_partiel=entryModeleML( indicesEntrprise, startTime=startTime, endTime=endTime, fileAssetName=fileAssetName, type_data="data_app",longueur_contexte=longueur_contexte )

    TAG = 20

    if rank != 0:
        comm.send( dataML_partiel, dest =0, tag = TAG)
    else:
        for i in range(1, nb_proc):
            dataML= np.concatenate( (dataML, comm.recv(source = i, tag = TAG) ), axis=0)
    
    if rank==0:
        np.savetxt('test.out', dataML, delimiter=',')   # X is an array
    
    """
    with open("test.out", 'r') as file:
        dataX= file.readlines()
    
    dataX_retrived = np.loadtxt(dataX, delimiter=',')
    """

main()

#%%------------- Apprentissage -----------------------
