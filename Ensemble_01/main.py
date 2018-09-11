#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler


columnNames=['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion',
         'Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
data = pd.read_csv('C:\MyWorkspace\python_pj\Ensemble_01\datasets\\breast-cancer-wisconsin.data',
                    header=0,names=columnNames)
data.head()

### Preprocessing 
# Drop Sample Code Number
data.drop(['Sample code number'],axis = 1, inplace = True)

data.info()

# fix missing value
(data['Bare Nuclei'] == '?').sum()

# replace by mean
# convert ? to 0 first
data.replace('?',0, inplace=True)

# Convert the DataFrame object into NumPy array otherwise you will not be able to impute
values = data.values

# Now impute it
imputer = Imputer()
imputedData = imputer.fit_transform(values)

# normalize the ranges of the features to a uniform range, in this case, 0 - 1.
scaler = MinMaxScaler(feature_range=(0, 1))
normalizedData = scaler.fit_transform(imputedData)

#%%
### Build Model
# Segregate the features from the labels
X = normalizedData[:,0:9]
Y = normalizedData[:,9]
seed = 7

#%%
### Bagging based Ensembling
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# First, you initialized a 10-fold cross-validation fold. 
kfold = model_selection.KFold(n_splits=10, random_state=7)

# After that, you instantiated a Decision Tree Classifier with 100 trees and wrapped it in a Bagging-based Ensemble.
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)

# Then you evaluated your model.
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#%%
### Boosting-based Ensemble
# In this case, you did an AdaBoost classification (with 70 trees) which is based on Boosting type of Ensembling
from sklearn.ensemble import AdaBoostClassifier
num_trees = 70
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results2 = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results2.mean())


#%%
### Voting-based Ensemble 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

kfold = model_selection.KFold(n_splits=10, random_state=seed)
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results3 = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
print(results3.mean())