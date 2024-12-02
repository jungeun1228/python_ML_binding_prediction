# -*- coding: utf-8 -*-
"""final_project_2024-12-02_10,000_50 vs 50_xgb_streamlit_v2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/122SWsPS1VK2elr0Ts-L0pZJKUOYVUWlp
"""

!pip install fastparquet
!pip install rdkit
!pip install xgboost
import pandas as pd
import numpy as np
import pickle
import rdkit
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem
from google.colab import drive
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

#Data input
drive.mount('/content/drive')

train_ori_df = pd.read_parquet('/content/drive/My Drive/Bootcamp_final project/train.parquet', engine='fastparquet')

train_df = train_ori_df.copy()

"""#0) Data exploration"""

#Data exploration
print(f"There are {train_df['binds'].sum()} of protein-molecule pairs that are bound to each other")
print(f"in total, {round(1589906/295246830 *100,2)}% of protein-molecule pairs were reported.")

#Show chemical structure
m = Chem.MolFromSmiles(train_df['molecule_smiles'][0])

m

"""#1) Preprocessing - Convert into ECFP; Split into Train/Test sets"""

#Create dataframe for train set (n sample = binding 5000 + no binding 5000 = 10,000)
train_no_bind_df = train_df.loc[train_df['binds'] == 0, :].sample(n=5000)
train_bind_df = train_df.loc[train_df['binds'] == 1, :].sample(n=5000)
train_df = pd.concat([train_bind_df, train_no_bind_df], axis = 0)

##Convert SMILES data to Morgan Fingerprints (Extended-Connectivity Fingerprints; ECFP)
#### need to solve the issue with Deprecation warning

def generate_ecfp(molecule, radius=2, bits=2048):
   if molecule is None:
       return None
   return list(AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=bits))

train_df['molecule'] = train_df['molecule_smiles'].apply(Chem.MolFromSmiles)
train_df['ecfp'] = train_df['molecule'].apply(generate_ecfp)
train_df

##One-hot encoding for protein type
#initialize
my_onehot_encoder = OneHotEncoder(sparse_output=False)   # consider drop // set_output -->e.g. my_onehot = OneHotEncoder(drop="first",sparse_output=False).set_output(transform='pandas')
#fit
my_onehot_encoder.fit(train_df['protein_name'].values.reshape(-1, 1))
#transform
protein_onehot = my_onehot_encoder.transform(train_df['protein_name'].values.reshape(-1, 1))

## combine fingerprint and one-hot encoded protein features
X = [ecfp + list(protein) for ecfp, protein in zip(train_df['ecfp'].tolist(), protein_onehot.tolist())]
X = pd.DataFrame(X, columns = range(0,2051))
# target binding values to we would like to predict
y = train_df['binds'].tolist()
y = pd.DataFrame(y, columns = ['binding'])
## split model input and outputs to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""#1-2) Create Pickle for X conversion function (not applied at the final version in the stream app)"""

def X_conversion(input_df):
  #[1] SMILES data to Morgan Fingerprints (Extended-Connectivity Fingerprints; ECFP)
  from rdkit.Chem import AllChem
  def generate_ecfp(molecule, radius=2, bits=2048):
    if molecule is None:
        return None
    return list(AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=bits))
  input_df['molecule'] = input_df['molecule_smiles'].apply(Chem.MolFromSmiles)
  input_df['ecfp'] = input_df['molecule'].apply(generate_ecfp)

  #[2]One-hot encoding for protein type
  import pandas as pd
  from sklearn.preprocessing import OneHotEncoder
  my_onehot_encoder = OneHotEncoder(sparse_output=False)  #initialize
  my_onehot_encoder.fit(input_df['protein_name'].values.reshape(-1, 1))  #fit
  protein_onehot = my_onehot_encoder.transform(input_df['protein_name'].values.reshape(-1, 1))  #transform

  #[3] combine fingerprint and one-hot encoded protein features
  X = [ecfp + list(protein) for ecfp, protein in zip(input_df['ecfp'].tolist(), protein_onehot.tolist())]
  X = pd.DataFrame(X, columns = range(0,2051))

with open('X_conversion_function.sav', 'wb') as file:
    pickle.dump(X_conversion, file)

"""#2) Feature selection - XGboost: baseline vs. select_from_model

"""

#XGBoost
#Baseline Model
baseline_xgb = xgb.XGBClassifier()
baseline_xgb.fit(X_train,y_train)
y_pred = baseline_xgb.predict(X_test)
#print performance
precision = average_precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
F1score = f1_score(y_test, y_pred)
print(f"Precision: {precision}; Recall: {recall}; F1 score: {F1score}")

select_model_xgb = SelectFromModel(xgb.XGBClassifier(),
                                    threshold=0.02 * np.max(baseline_xgb.feature_importances_))

# Transform the train set.
X_train_selected_model_xgb = select_model_xgb.fit_transform(X_train, y_train)
# Transform the test set.
X_test_selected_model_xgb = select_model_xgb.transform(X_test)

"""#2-2) Create pickle for feature selection model"""

import pickle
with open('X_feature_selection.sav', 'wb') as file:
    pickle.dump(select_model_xgb, file)

"""#2-3) Performance test with selected model"""

# Show shape of the train and test dataset to check for the number of features kept.
print(X_train_selected_model_xgb.shape, X_test_selected_model_xgb.shape)

#Use data after selectfrommodel
model_with_select_xgb = xgb.XGBClassifier()
model_with_select_xgb.fit(X_train_selected_model_xgb, y_train)
y_pred_selected_model_xgb = model_with_select_xgb.predict(X_test_selected_model_xgb)
#print performance
precision = average_precision_score(y_test, y_pred_selected_model_xgb)
F1score = f1_score(y_test, y_pred_selected_model_xgb)
print(f"Precision: {precision}; Recall: {recall}; F1 score: {F1score}")

"""#3) Hyperparameter"""

from sklearn.model_selection import RandomizedSearchCV

params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}
xgb_search = RandomizedSearchCV(
    estimator=model_with_select_xgb,
    param_distributions=params,
    n_iter=100,
    cv=4,
    n_jobs=-1,
    verbose=2,
    scoring='average_precision',
    random_state=42
)

# fitting with each candidates
xgb_search.fit(X_train_selected_model_xgb, y_train)

# retrieve the best model
best_xgb_model = xgb_search.best_estimator_

xgb_search.best_score_

xgb_search.best_params_

#Evaluate for test set
y_pred = best_xgb_model.predict(X_test_selected_model_xgb)
precision = average_precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
F1score = f1_score(y_test, y_pred)
print(f"Precision: {precision}; F1 score: {F1score}")

"""#3-2) Create pickle for best model after parameter optimization"""

with open('model.sav', 'wb') as file:
    pickle.dump(best_xgb_model, file)