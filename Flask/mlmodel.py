
'''
This is a simple linear regression model to predit the CO2 emmission from cars
Dataset:
FuelConsumption.csv, which contains model-specific fuel consumption ratings and estimated carbon dioxide emissions
for new light-duty vehicles for retail sale in Canada
'''

import pickle
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from xgboost import plot_importance, to_graphviz
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score   

dataset=pd.read_excel('c_fraud_transaction.xlsx')

# Dans notre nouveau dataframe on ne garde que les transactions de type 'TRANSFER' ou 'CASH_OUT'
dat=dataset.loc[(dataset.type=='TRANSFER')|(dataset.type=='CASH_OUT')].copy()
print('Le dataset, en sélectionant les colonnes \'TRANSFER\' et \'CASH_OUT\', a {} lignes et {} colonnes.'.format(dat.shape[0], dat.shape[1]))
dat=dataset.loc[(dataset.type=='TRANSFER')|(dataset.type=='CASH_OUT')].copy()
print('Le dataset, en sélectionant les colonnes \'TRANSFER\' et \'CASH_OUT\', a {} lignes et {} colonnes.'.format(dat.shape[0], dat.shape[1]))

# Transformation des valeurs de la variable type en 1 et 0, pas besoin d'utiliser le module sklearn car on peut le 
# faire directement
dat['type'] = np.where(dat['type'] == 'TRANSFER', 1, 0)
dat = dat.reset_index(drop=True)

# Supprimons les variables peu représentatives
dat.drop(['nameOrig','isFlaggedFraud','nameDest'],1,inplace=True)

"""
Les valeurs manquantes crédits on le remplace par -1 car il y avait plus des transactions frauduleuses 
qui avaient des valeurs manquantes pour les comptes crédits, cela permet à l'algorithme de differencier
"""
dat.loc[(dat.oldbalanceDest == 0) & (dat.newbalanceDest == 0) & (dat.amount != 0), \
      ['oldbalanceDest', 'newbalanceDest']] = - 1

# les valeurs manquantes debits on le remplace par 'NaN'
dat.loc[(dat.oldbalanceOrg == 0) & (dat.newbalanceOrig == 0) & (dat.amount != 0), \
      ['oldbalanceOrg', 'newbalanceOrig']] = np.nan
# les 'NaN' sont à leur tour remplacées par les valeurs medianes
dat[['oldbalanceOrg','newbalanceOrig']]=dat[['oldbalanceOrg','newbalanceOrig']].fillna(dat.median())

# Nous avons créé deux variables qui seront chacune une combinaison linéaire des deux variables corréelées
# et du montant de la transaction
d_model = pd.DataFrame(dat)
d_model['errorBalanceSender'] = d_model.oldbalanceOrg  - d_model.amount - d_model.newbalanceOrig
d_model['errorBalanceReceiver'] = d_model.oldbalanceDest + d_model.amount - d_model.newbalanceDest
d_model.drop(['oldbalanceOrg', 'oldbalanceDest'], 1, inplace=True)
d_model = d_model.rename(columns={'newbalanceOrig':'balanceSender', 'newbalanceDest':'balanceReceiver'})

# Standartiser
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# XGboost
X = d_model.iloc[:, d_model.columns !='isFraud']
y = d_model.iloc[:, 5]
clf = XGBClassifier(max_depth=7, learning_rate=0.1, n_estimators=180, objective='binary:logistic', nthread=4,
    seed=42)
clf.fit(X_train, y_train)

# Saving model to disk
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(clf, open('model.pkl','wb'))

#Loading model to compare the results
#model = pickle.load(open('model.pkl','rb'))
#Faire une prédiction de la probabilité p(Y =1) à l'aide de la fonction .predict_proba()
#print(model.predict_proba([[2.6, 8, 10.1, 5, 8, 9, 1000]]))










