# -*- coding: utf-8 -*-
"""final_project_2024-12-02_model comparison.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1n35XJfSsVgWHUdZjYjb_W75BE6P9eIAT
"""

!pip install fastparquet
!pip install rdkit
!pip install xgboost
import xgboost as xgb
import random
import rdkit
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from google.colab import drive
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay

#Data input
drive.mount('/content/drive')

train_ori_df = pd.read_parquet('/content/drive/My Drive/Bootcamp_final project/train.parquet', engine='fastparquet')

train_df = train_ori_df.copy()

"""#Data exploration"""

train_ori_df.head()

print(f"There are {train_df['binds'].sum()} of protein-molecule pairs that are bound to each other")
print(f"in total, {round(1589906/295246830 *100,2)}% of protein-molecule pairs were reported.")

m = Chem.MolFromSmiles('Cc1ccccc1')

m

"""#Create test sets"""

#Create dataframe for train set (n sample = 1000 + 1000 = 2000)
train_bind_df = train_df.loc[train_df['binds'] == 0, :].sample(n=1000)
train_no_bind_df = train_df.loc[train_df['binds'] == 1, :].sample(n=1000)
train_2000_df = pd.concat([train_bind_df, train_no_bind_df], axis = 0)
train_2000_df

##Convert SMILES data to Morgan Fingerprints (Extended-Connectivity Fingerprints; ECFP)
#### need to solve the issue with Deprecation warning!!!

def generate_ecfp(molecule, radius=2, bits=2048):
   if molecule is None:
       return None
   return list(AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=bits))

train_2000_df['molecule'] = train_2000_df['molecule_smiles'].apply(Chem.MolFromSmiles)
train_2000_df['ecfp'] = train_2000_df['molecule'].apply(generate_ecfp)
train_2000_df

#One-hot encoding for protein type
#initialize
my_onehot_encoder = OneHotEncoder(sparse_output=False)   # consider drop // set_output -->e.g. my_onehot = OneHotEncoder(drop="first",sparse_output=False).set_output(transform='pandas')
#fit
my_onehot_encoder.fit(train_2000_df['protein_name'].values.reshape(-1, 1))
#transform
protein_onehot = my_onehot_encoder.transform(train_2000_df['protein_name'].values.reshape(-1, 1))

# combine fingerprint and one-hot encoded protein features
X = [ecfp + list(protein) for ecfp, protein in zip(train_2000_df['ecfp'].tolist(), protein_onehot.tolist())]
# target binding values to we would like to predict
y = train_2000_df['binds'].tolist()
# split model input and outputs to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""#option 1) KNN"""

#randomized search for KNN
params = {
   'n_neighbors': range(2, 50, 2),
   'weights': ["uniform", "distance"],
   'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev'],
   'p': range(1, 5),
   'leaf_size': range(10, 50, 2),
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']

}
knn_search = RandomizedSearchCV(
    estimator=KNeighborsClassifier(),
    param_distributions=params,
    n_iter=100,
    cv=4,
    n_jobs=-1,
    verbose=2,
    scoring='average_precision',
    random_state=42
)

# fitting each candidates
knn_search.fit(X_train, y_train)

# retrieve the best model
best_knn_model = knn_search.best_estimator_

knn_search.best_score_

knn_search.best_params_

#Evaluate the model - precision and F1
y_pred = best_knn_model.predict(X_test)
precision = average_precision_score(y_test, y_pred)
F1score = f1_score(y_test, y_pred)
print(f"Precision: {precision}; F1 score: {F1score}")

"""#option 2) Random forest"""

#randomized search for random forest
params = {
   'n_estimators': [100, 200, 300],
   'max_depth': [None, 10, 20, 30],
   'min_samples_split': [2, 5, 10],
   'min_samples_leaf': [1, 2, 4, 6],
   'max_features': ['sqrt'],
   'bootstrap': [True, False]
}
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=params,
    n_iter=100,
    cv=4,
    n_jobs=-1,
    verbose=2,
    scoring='average_precision',
    random_state=42
)

# fitting each candidates
random_search.fit(X_train, y_train)

# retrieve the best model
best_RF_model = random_search.best_estimator_

random_search.best_score_

random_search.best_params_

#Evaluate the model - precision and F1
y_pred = best_RF_model.predict(X_test)
precision = average_precision_score(y_test, y_pred)
F1score = f1_score(y_test, y_pred)
print(f"Precision: {precision}; F1 score: {F1score}")

"""##Random forest - f1 scoring"""

#randomized search for random forest
params = {
   'n_estimators': [100, 200, 300],
   'max_depth': [None, 10, 20, 30],
   'min_samples_split': [2, 5, 10],
   'min_samples_leaf': [1, 2, 4, 6],
   'max_features': ['sqrt'],
   'bootstrap': [True, False]
}
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=params,
    n_iter=100,
    cv=4,
    n_jobs=-1,
    verbose=2,
    scoring='f1',
    random_state=42
)

# fitting each candidates
random_search.fit(X_train, y_train)

random_search.best_score_

"""#option 3) SVM"""

#randomized search for SVM

params = {
    'C': [10**e for e in range(-2,2)],                          #4 options
    'kernel': ['linear', 'poly', 'rbf'],                        #3 options
    'degree': range(2, 5),                                      #3 options
    'gamma': ['scale', 'auto'] + [10**e for e in range(-3, -1)] #4 options
}
svm_search = RandomizedSearchCV(
    estimator=SVC(random_state=42),
    param_distributions=params,
    n_iter=100,
    cv=4,
    n_jobs=-1,
    verbose=2,
    scoring='average_precision',
    random_state=42
)

# fitting each candidates
svm_search.fit(X_train, y_train)

# retrieve the best model
best_svm_model = random_search.best_estimator_

svm_search.best_score_

svm_search.best_params_

#Evaluate the model - precision and F1
y_pred = best_svm_model.predict(X_test)
precision = average_precision_score(y_test, y_pred)
F1score = f1_score(y_test, y_pred)
print(f"Precision: {precision}; F1 score: {F1score}")

"""#Error analysis - confusion matrix"""

# Create confusion matrix with our tuned decision tree
ConfusionMatrixDisplay.from_estimator(svm_search,
                                      X_test,
                                      y_test,
                                      display_labels=['No bind', 'Bind']);

"""#option 4) XGBoost"""

#Randomized search for XGBoost

params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}
xgb_search = RandomizedSearchCV(
    estimator=xgb.XGBClassifier(random_state=42),
    param_distributions=params,
    n_iter=100,
    cv=4,
    n_jobs=-1,
    verbose=2,
    scoring='average_precision',
    random_state=42
)

# fitting each of candidates
xgb_search.fit(X_train, y_train)

# retrieve the best model
best_xgb_model = random_search.best_estimator_

xgb_search.best_score_

xgb_search.best_params_

#Evaluate the model - precision and F1
y_pred2 = best_xgb_model.predict(X_test)
precision2 = average_precision_score(y_test, y_pred2)
F1score2 = f1_score(y_test, y_pred2)
print(f"Precision: {precision2}; F1 score: {F1score2}")

# Create confusion matrix with our tuned decision tree
ConfusionMatrixDisplay.from_estimator(xgb_search,
                                      X_test,
                                      y_test,
                                      display_labels=['No bind', 'Bind']);