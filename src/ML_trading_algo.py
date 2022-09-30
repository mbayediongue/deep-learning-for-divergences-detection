# -*- coding: utf-8 -*-
"""
Cette partie contient le code pour le modèle de machine learning permettant d'estimer la qualité d'une 
divergence prédite par l'algo
"""
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from sklearn import svm,preprocessing
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestRegressor as rf
import os
import yfinance as yf 
from pandas import read_csv
import csv
from GetRUssel3000 import orderByDataType, getEntryML
import datetime as dt

#%%
# Récupération des données d'apprentissage


exec(open('Imports.py').read())
exec(open('Functions_algo.py').read())
exec(open('Functions_algo2.py').read())
#exec(open('GetRUssel3000.py').read())

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


    nbVariables=longueur_contexte+4
    X = np.empty((0, nbVariables))  # inputs of the ML model
    Y = np.empty((0))            # output ML modele
    nbErreur=0
    listeEntrepriseError=[]
    for i in range(indiceDebutEntrprise, indiceFinEntreprise):
      try:
        result_getEntry=getEntryML(listR3000[i],type_data=type_data, startTime=startTime, endTime=endTime )
        dataML=orderByDataType(result_getEntry, type_data=type_data, longueur_contexte=longueur_contexte) #  [liste_pente cours, liste_pente rsi, angle, liste_longueur div, liste_reversal (hauteur) ]
        X=np.concatenate( ( X, np.array(np.transpose(dataML[0:-1]))) , axis=0)
        
        if type_data=="data_app":
            Y= np.concatenate( ( Y, np.array(dataML[-1])) )
        print( "Entreprise n°"+str(i-indiceDebutEntrprise) +" en cours ...")
        #dataML.append( orderByDataType(getEntryML(listR3000[i])) ) 
        
      except ValueError:
        print( " Erreur sur l'entreprise "+str(i))
        nbErreur+=1
        listeEntrepriseError.append(i)


    print( "Nombre d'entreprises comportant des erreurs dans la détection de div "+str(nbErreur) )
    
    if type_data=="data_app":
        return (X, Y)
    elif type_data=="data_test":
        return X

""" 

Cette fonction renvoie le modèle de machine learning entrainé (Réseau 
de neurones, fôret aléatoire...)

 * indicesEntrprise : une liste de deux entiers: l'indice de début et l'indice
de fin des entreprises du russel3000 (ou de fileAssetName) à récupérer pour la phase d'apprentissage
et de test
 * typeModel : type d'algo utilié ( par défaut un réseau de neurones NN)
     Choix disponible: "NN", "SVM", "RF" (random forest), "REG_LIN" (regression linéaire)
""" 
def construct_modeleML(X, Y, fileAssetName='ticker_R3000.csv',startTime='2020-12-14', endTime=dt.datetime.now(), typeModele="NN", longueur_contexte=15):
    
    #(X,Y)=entryModeleML( indicesEntrprise, startTime=startTime, endTime=endTime, fileAssetName=fileAssetName, type_data="data_app",longueur_contexte=longueur_contexte )
    
    #---------SPLITING TRAINING AND VALIDATION SET----------------------#
    val_size=0.3 # size validation set 
    print(len(Y))
    (X_train, X_val, Y_train, Y_val) = train_test_split(X, Y, test_size=val_size, random_state=None)
    
    # STANDARDIZE DATA

    scalerX = preprocessing.StandardScaler().fit(X_train)
    X_train = scalerX.transform(X_train)
    X_val = scalerX.transform(X_val)
    
    
    scalerY= preprocessing.StandardScaler().fit(Y_train)
    #mean_Ytrain=np.mean(Y_train)
    #std_Ytrain=np.std(Y_train)
    #Y_train = (Y_train-mean_Ytrain)/std_Ytrain
    Y_val = scalerY.transform(Y_val)
    
    # -------------
    #  TRAINING
    # -------------
    if  typeModele=="NN":
        #----------------------------- Neural Network -------------------------
        #tf.random.set_seed(1234)
        # build a model
        modele = Sequential()
        modele.add(Dense(5, input_shape=(X_train.shape[1],), activation='relu')) # Add an input shape! (features,)
        modele.add(Dense(5, activation='relu'))
        modele.add(Dense(1, activation='relu'))
        modele.summary() 
        
        # compile the model
        modele.compile(optimizer='Adam', 
                      loss='MeanSquaredError',
                      metrics=['MSE'])
        
        # now we just update our model fit call
        history = modele.fit(X_train, Y_train,
                            epochs=200, 
                            batch_size=10,
                            validation_data = (X_val, Y_val),
                            verbose=0)
        
        Y_val_hat=modele.predict(X_val) # probabilities
        Y_val_hat=np.array( [ x[0] for x in Y_val_hat])
        classifier=" Neural Network "
    elif  typeModele=="REG_LIN":
        #------------------ Régression linéaire avec sklearn ----------------
        modele=LinearRegression()
        modele.fit(X_train, Y_train, ) # entrainement
        Y_val_hat=modele.predict(X_val)
        #Y_test_hat= Y_test_hat[:,1]
        classifier=" Linear regression "

    
    elif  typeModele=="SVM":
        # -------------------SVM------------------------------------ -----------
        modele = svm.SVR(kernel="rbf", gamma=0.09, C=1)
        modele.fit(X_train, Y_train)
        Y_val_hat=modele.predict(X_val)
        #Y_test_hat= Y_test_hat[:,1]
        classifier=" SVM "
    
    elif  typeModele=="RF":
        #------------------ Random Forest --------------------------------
        modele=rf( max_depth=10, random_state=0)
        modele.fit(X_train, Y_train)
        Y_val_hat=modele.predict(X_val)
        #Y_test_hat=np.array( [ x[1] for x in Y_test_hat])
    
        classifier=" Random Forest "
    else :
        print("Modèle choisi non disponible")
    
    Y_val_hat=scalerY.inverse_transform(Y_val_hat)
    #Y_val_hat =std_Ytrain*Y_val_hat +mean_Ytrain
    rmse=np.sqrt( np.mean( (Y_val_hat-Y_val)**2) )

    print("***************")
    print("The standard deviation of the test set is {} ( rmse de reference)".format( round(np.std(Y_val),3) ))
    print("The RMSE of the"+classifier+"regressor is {}".format( round(rmse,3)))
    print("***************")
    #print( "Classifier:"+ classifier)
    
    plt.plot( Y_val, Y_val_hat, '*')
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title("Performance of the model in the Validation set "+typeModele)
    
    return modele, scalerX 


"""
Renvoie la hauteur h (ou reversal ) prédite par le modele
"""
def compute_reversal_ML( model, scalerX, indicesEntrprise,  startTime='2021-12-14', endTime=dt.datetime.now(), fileAssetName='ticker_R3000.csv', model_name="NN", longueur_contexte=15):
    
    X_test=entryModeleML( indicesEntrprise, startTime=startTime, endTime=endTime, fileAssetName=fileAssetName, type_data="data_test", longueur_contexte=longueur_contexte)

    X_test = scalerX.transform(X_test)
    Y_pred_hat=model.predict(X_test) # probabilities
    
    if model_name=="NN":
        Y_pred_hat=np.array( [ x[0] for x in Y_pred_hat])
    return Y_pred_hat
    


#%%
"""
data = yf.Ticker("AMZN") #AAPL can be replaced by the desired asset
dataDF = data.history(interval='1h',start='2021-2-1',end=dt.datetime.now()) 

l=dataDF['Close']
l1=l.tolist()

dataDF=Creation_rsi(dataDF)
l2=dataDF['RSI']
l2=l2.tolist()

"""
#entryModeleML( indicesEntrprise, startTime=startTime, endTime=endTime, fileAssetName=fileAssetName, type_data="data_app" )
#%%------------- Apprentissage -----------------------
longueur_contexte=15
indicesEntrprise=[0, 2] # Indice de début et de fin des entreprises sélectionnées
fileAssetName='ticker_R3000.csv'
startTime='2020-12-14'
endTime=dt.datetime.now()            #'2021-12-14'

with open("testtot.out", 'r') as file:
    data= file.readlines()
dataX_retrieved = np.loadtxt(data, delimiter=',')
data=np.unique(dataX_retrieved, axis=0)
data.shape
typeModele="REG_LIN" #neural network: NN, REG_LIN

#X,Y=entryModeleML( indicesEntrprise, startTime=startTime, endTime=endTime, fileAssetName=fileAssetName, type_data="data_app",longueur_contexte=longueur_contexte )



(X,Y)=data[:,2].reshape(-1,1), data[:,-1].reshape(-1,1)

model, scalerX=construct_modeleML(X,Y, fileAssetName=fileAssetName,startTime=startTime, endTime=endTime, typeModele=typeModele, longueur_contexte=longueur_contexte)

plt.plot(X, Y, '*')



#%% ---------- TEST -------------------------

startTime='2021-12-14'
endTime=dt.datetime.now()
#Y_pred=compute_reversal_ML( model, scalerX, indicesEntrprise,  startTime=startTime, endTime=endTime, fileAssetName=fileAssetName, model_name=typeModele)
import os
import glob
nbVariables=20
Xtot = np.empty((0, nbVariables))  # inputs of the ML model
image_labels = os.listdir("./algo_parallele" )
for i, lab in enumerate(image_labels):
    #print(lab[-3:-1])
    if lab[-3:]=="out":
        with open("./algo_parallele/"+lab, 'r') as file:
            dataX= file.readlines()
    
        dataX_retrived = np.loadtxt(dataX, delimiter=',')
        Xtot=np.concatenate( ( Xtot, np.array(dataX_retrived )) , axis=0)

np.savetxt('testtot.out', Xtot, delimiter=',')   # X is an array

