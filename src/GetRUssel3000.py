# -*- coding: utf-8 -*-
"""
Ce fichier permet de calculer, pour plusieurs valeurs d'assets prédetermines, 
les paramètres d'entrée et le resultat de sortie pour du machine learning.
Le but du ML est de determiner la qualité d'une divergence.
Dans ce fichier, on va detecter les divergences (bull, bear) de plusieurs
assets (compris dans un fichier csv qu'on lira) pour extraire, pour chaque
divergence :
    - la durée de la divergence [entrée du ML]
    - la pente du cours et du RSI pour cette divergence [entrée du ML]
    - la hauteur du saut ou du décrochage du cours 100 tics après la fin
        de la divergence [sortie du ML]

Projet PRICE 2021 : Trading Algorithm based on Machine Learning
"""
import matplotlib.pyplot as plt
import datetime as dt
import csv
import numpy as np
import yfinance as yf
from Functions_algo2 import Extremas, Creation_rsi, pt_interessant_typeDiv, creation_divergenceBearReg
from Functions_algo2 import Filtration, Effect, pente_qualite, creation_divergenceBullReg, creation_divergenceBearHidden, creation_divergenceBullHidden
import os

#%%
print("Current Working Directory " , os.getcwd())
#Attention : changer la ligne suivante selon le PC utilisé
#mettre le chemin
#os.chdir("D:/Cours EMSE/23. PRICE/algorithmic_trading-main/Trading_project")

#The next two scripts are to execute before launching the code 
exec(open('Imports.py').read())         #fichier pour importer les packages
exec(open('Functions_algo2.py').read())  #fonctions de l'algo

#On va chercher le csv qui contient les assets qu'on veut utiliser
NomDuFichier = "ticker_R3000.csv"

from pandas import read_csv  
data = read_csv(NomDuFichier, sep=";")
data = np.array(data)

#%%

#data : liste des valeurs boursières où on ira chercher
#nos divergences.
listR3000 = []

fh = open('ticker_R3000.csv')
reader = csv.reader(fh)
for ligne in reader:
    listR3000.append(ligne[0])

#print(listR3000)

#%%

def getEntryML(asset_list, type_data="data_app", startTime='2020-12-14', endTime=dt.datetime.now()):
    out_Bear_Reg = []
    out_Bear_Hidden = []
    out_Bull_Reg = []
    out_Bull_Hidden = []
    #startTime = '2020-12-14'
    #endTime=dt.datetime.now()
    for asset in asset_list :
        #Retrieving the asset data
        data = yf.Ticker(asset) 
        dataDF = data.history(interval='1h',start=startTime,end=endTime) 
        l=dataDF['Close']
        l=l.tolist()
        (Mins,indMin,Maxs,indMax)=Extremas(l)
        
        #Retrieving indicator (here : RSI)
        dataDF=Creation_rsi(dataDF)
        l2=dataDF['RSI']
        l2=l2.tolist()
        (Mins2,indMin2,Maxs2,indMax2)=Extremas(l2)
            
        #Detection divergences BEARISH ---------------------------------------
        #---------------------------------------------------------------------
            #regular divergences
        PHHausses_interessantes_cours=pt_interessant_typeDiv(indMax,"cours",l,l2,"Haut","RegularBearish",50, dataDF)
        PHBaisses_interessantes_rsi=pt_interessant_typeDiv(indMax2,"RSI",l,l2,"Haut","RegularBearish",50, dataDF)
            #hidden divergences
        PHBaisses_interessantes_cours=pt_interessant_typeDiv(indMax,"cours",l,l2,"Haut","HiddenBearish",50,dataDF)
        PHHausses_interessantes_rsi=pt_interessant_typeDiv(indMax2,"RSI",l,l2,"Haut","HiddenBearish",50, dataDF)
        
        #Reception des données :
                            #BEAR REGULAR ------------------------------------
            #Filtering

        BearRegDivergence = creation_divergenceBearReg(PHHausses_interessantes_cours,PHBaisses_interessantes_rsi, dataDF)
        BearRegDivergence_filtree = Filtration(BearRegDivergence)
            #Calculating input data for ML
        for div in BearRegDivergence_filtree :
            longeur = div[1]-div[0]
            (p1, p2) = pente_qualite(div,l,l2)
            if type_data=="data_app" and Effect(div,l,'bear')!=None :
                # angle 1, angle 2, longeur, hauteur de fin
                out_Bear_Reg.append([p1, p2, longeur, Effect(div,l,'bear'), div[1]])
            if type_data=="data_test":
                # angle 1, angle 2, longeur
                out_Bear_Reg.append([p1, p2, longeur, div[1]])
                
            
        #print([p1, p2, longeur, Effect(div,l,'bear')])
                            #BEAR HIDDEN -------------------------------------
            #Filtering
        BearHiddenDivergence = creation_divergenceBearHidden(PHBaisses_interessantes_cours,PHHausses_interessantes_rsi, dataDF)
        BearHiddenDivergence_filtree = Filtration(BearHiddenDivergence)
        
            #Calculating input data for ML
        for div in BearHiddenDivergence_filtree :
            longeur = div[1]-div[0]
            (p1, p2) = pente_qualite(div,l,l2)
            if type_data=="data_app" and Effect(div,l,'bear')!=None :
                out_Bear_Hidden.append([p1, p2, longeur, Effect(div,l,'bear'), div[1]])  
            if type_data=="data_test":
                # angle 1, angle 2, longeur
                out_Bear_Hidden.append([p1, p2, longeur, div[1]])


        #Detection divergences BULLISH ---------------------------------------
        #---------------------------------------------------------------------
            #regular bullish
        PBBaisses_interessantes_cours = pt_interessant_typeDiv(indMin,"cours",l,l2,"Bas","RegularBullish",50, dataDF)
        PBHausses_interessantes_rsi=pt_interessant_typeDiv(indMin2,"RSI",l,l2,"Bas","RegularBullish",50, dataDF)
            #hidden bullish
        PBHausses_interessantes_cours=pt_interessant_typeDiv(indMin,"cours",l,l2,"Bas","HiddenBullish",50, dataDF)
        PBBaisses_interessantes_rsi=pt_interessant_typeDiv(indMin2,"RSI",l,l2,"Bas","HiddenBullish",50, dataDF)
        
        #Reception des données :
                            #BULL REGULAR ------------------------------------
            #Filtering
        BullRegDivergence=creation_divergenceBullReg(PBBaisses_interessantes_cours,PBHausses_interessantes_rsi, dataDF)
        BullRegDivergence_filtree=Filtration(BullRegDivergence)
            #Calculating input data for ML
        for div in BullRegDivergence_filtree :
            longeur = div[1]-div[0]
            (p1, p2) = pente_qualite(div,l,l2)
            if type_data=="data_app" and Effect(div,l,'bull')!=None :
                out_Bull_Reg.append([p1, p2, longeur, Effect(div,l,'bull'), div[1]])
            if type_data=="data_test":
                out_Bull_Reg.append([p1, p2, longeur, div[1]])
                
                            #BULL HIDDEN -------------------------------------
            #Filtering
        BullHiddenDivergence=creation_divergenceBullHidden(PBHausses_interessantes_cours,PBBaisses_interessantes_rsi, dataDF)
        BullHiddenDivergence_filtree=Filtration(BullHiddenDivergence)
        
            #Calculating input data for ML
        for div in BullHiddenDivergence_filtree :
            longeur = div[1]-div[0]
            (p1, p2) = pente_qualite(div,l,l2)
            if type_data=="data_app" and Effect(div,l,'bull')!=None :
                out_Bull_Hidden.append([p1, p2, longeur, Effect(div,l,'bull'), div[1]]) 
            if type_data=="data_test":
                out_Bull_Hidden.append([p1, p2, longeur], div[1]) 
    """"
    print("Nb d'asset étudiés : ", len(asset_list))
    print("TOTAL DIVERGENCES : ", len(out_Bull_Hidden)+len(out_Bull_Reg)+len(out_Bear_Hidden)+len(out_Bear_Reg))
    print("ordre : Bear Reg, Bear Hid, Bull Reg, Bull Hid")

    
    print("Bear Regular : ", len(out_Bear_Reg))
    print("Bear Hidden : ", len(out_Bear_Hidden))
    print("Bull Regular : ", len(out_Bull_Reg))
    print("Bull Hidden : ", len(out_Bull_Hidden))       
    """
    return([out_Bear_Reg, out_Bear_Hidden, out_Bull_Reg, out_Bull_Hidden])
    
TEST  = getEntryML(listR3000[0])




#%%
"""
Permet de classer le resultat de la fonction ci dessus selon le type
de la donnée [angle cours, angle RSI, longueur ...]

L'ordre de sortie est le suivant :
    pente cours, pente RSI, angle, longueur, hauteur, tps_fin_div.
    
"""
def orderByDataType(outGetEntryML,cours,rsi, type_data="data_app", include_contexte=True, longueur_contexte=15) :
    list_p1=[]
    list_p2=[]
    list_alpha=[]
    list_l=[]
    list_h=[]
    list_contexte_cours=[]
    list_contexte_rsi=[]
           
    for cat in range(4) :
        for div in outGetEntryML[cat] :
            list_p1.append(div[0])
            list_p2.append(div[1])
            list_alpha.append(  div[1]-div[0] )
            list_l.append(div[2])
            tps_fin_div=div[4]
            if include_contexte:
                list_contexte_cours.append( cours[(tps_fin_div-longueur_contexte):tps_fin_div])
                list_contexte_rsi.append( div[(tps_fin_div-longueur_contexte):tps_fin_div])
                
            if type_data=="data_app":
                list_h.append(div[3])
                
    #print("ordre : pente cours, pente rsi, angle, longueur, reversal")
    
    result=[list_p1, list_p2,list_alpha,list_l]
    
    if include_contexte:
        for k in range( longueur_contexte):
            result.append( list_contexte_cours[k])
            result.append( list_contexte_rsi[k])
        
    if type_data =="data_app":
        result.append( list_h)
        
    return result
    

TEST2 = orderByDataType(TEST, l1, l2)


#%%
""" Visualisation angle vs hauteur"""

plt.scatter(TEST2[2], TEST2[4])
plt.title("angle cours-RSI  vs  hauteur décrochage")
plt.show()
    
    