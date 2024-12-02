# -*- coding: utf-8 -*-
"""final_project_2024-09-02_10,000_50 vs 50_xgb+PCA.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Hl4CoYlwTGJgQBk1zeUjB9SMYuUp_prn
"""

!pip install fastparquet
!pip install rdkit
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

train_ori_df = pd.read_parquet('/content/drive/My Drive/Bootcamp_final project/train.parquet', engine='fastparquet')

train_df = train_ori_df.copy()

"""#Data exploration"""

#Data exploration
print(f"There are {train_df['binds'].sum()} of protein-molecule pairs that are bound to each other")
print(f"in total, {round(1589906/295246830 *100,2)}% of protein-molecule pairs were reported.")

m = Chem.MolFromSmiles(train_df['molecule_smiles'][0])

m

"""#1) Preprocessing - Convert into ECFP; Split into Train/Test sets"""

#Create dataframe for train set (n sample = 5000 + 5000 = 10,000)
train_no_bind_df = train_df.loc[train_df['binds'] == 0, :].sample(n=5000)
train_bind_df = train_df.loc[train_df['binds'] == 1, :].sample(n=5000)
train_df = pd.concat([train_bind_df, train_no_bind_df], axis = 0)
train_df

# convert SMILES data to Morgan Fingerprints (Extended-Connectivity Fingerprints; ECFP)
#### need to solve the issue with Deprecation warning!!!

def generate_ecfp(molecule, radius=2, bits=2048):
   if molecule is None:
       return None
   return list(AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=bits))


train_df['molecule'] = train_df['molecule_smiles'].apply(Chem.MolFromSmiles)
train_df['ecfp'] = train_df['molecule'].apply(generate_ecfp)

train_df

#One-hot encoding for protein type

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

#initialize
my_onehot_encoder = OneHotEncoder(sparse_output=False)   # consider drop // set_output -->e.g. my_onehot = OneHotEncoder(drop="first",sparse_output=False).set_output(transform='pandas')
#fit
my_onehot_encoder.fit(train_df['protein_name'].values.reshape(-1, 1))
#transform
protein_onehot = my_onehot_encoder.transform(train_df['protein_name'].values.reshape(-1, 1))

from sklearn.model_selection import train_test_split

# combine fingerprint and one-hot encoded protein features
X = [ecfp + list(protein) for ecfp, protein in zip(train_df['ecfp'].tolist(), protein_onehot.tolist())]
X = pd.DataFrame(X, columns = range(0,2051))
# target binding values to we would like to predict
y = train_df['binds'].tolist()
y = pd.DataFrame(y, columns = ['binding'])
# split model input and outputs to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""#2) Feature selection - xgb: baseline vs. select_from_model"""

!pip install xgboost
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.feature_selection import SelectFromModel

#1.Random forest
from sklearn.ensemble import RandomForestClassifier
#Baseline Model
baseline_rf = RandomForestClassifier()
baseline_rf.fit(X_train,y_train)
y_pred = baseline_rf.predict(X_test)
#print performance
precision = average_precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
F1score = f1_score(y_test, y_pred)
print(f"Precision: {precision}; Recall: {recall}; F1 score: {F1score}")

#2. XGBoost
#Baseline Model
baseline_xgb = xgb.XGBClassifier()
baseline_xgb.fit(X_train,y_train)
y_pred = baseline_xgb.predict(X_test)
#print performance
precision = average_precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
F1score = f1_score(y_test, y_pred)
print(f"Precision: {precision}; Recall: {recall}; F1 score: {F1score}")

"""#2-2) Select from model"""

select_model_xgb = SelectFromModel(xgb.XGBClassifier(),
                                    threshold=0.02 * np.max(baseline_xgb.feature_importances_))

# Transform the train set.
X_train_selected_model_xgb = select_model_xgb.fit_transform(X_train, y_train)

# Transform the test set.
X_test_selected_model_xgb = select_model_xgb.transform(X_test)

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

"""#2-3) PCA"""

from sklearn.decomposition import PCA

from sklearn import set_config
set_config(transform_output="pandas")

# Initialise the PCA object
pca = PCA()

# Fit the PCA object to the data
pca_fit = pca.fit(X_train_selected_model_xgb)

# Transform scaled_features_df based on the fit calculations
pca_basic_df = pca.transform(X_train_selected_model_xgb)

pca_basic_df

# Get the variance explained by each principal component
explained_variance_array = pca_fit.explained_variance_ratio_

pd.DataFrame(explained_variance_array, columns=["Variance explained"])

import matplotlib.pyplot as plt
import seaborn as sns

# Create a Pandas DataFrame from the variance explained array
explained_variance_array_df = pd.DataFrame(explained_variance_array, columns=["Variance explained"])

(
  # Create a line chart with sns.relplot
  sns.relplot(
      kind = 'line',
      data = explained_variance_array_df,
      x = explained_variance_array_df.index,
      y = "Variance explained",
      marker = 'o',
      aspect = 1.3)
  # Set the title of the plot
  .set(title = "Proportion of variance explained by each principal component")
  # Set the axis labels
  .set_axis_labels("Principal component number", "Proportion of variance")
);

plt.axvline(x=31)

# Set the variable elbow to where you believe the elbow is
elbow = 31

# Create a PCA object with {elbow} principal components
# We add 1 as the principal components start at 0 and not 1
pca_elbow = PCA(n_components = elbow + 1)

# Fit the PCA object to the scaled features dataframe and transform it
X_train_pca_elbow_df = pca_elbow.fit_transform(X_train_selected_model_xgb)

# The dataframe now contains the principal components of the scaled features dataframe
X_train_pca_elbow_df

X_test_pca_elbow_df = pca_elbow.fit_transform(X_test_selected_model_xgb)

# The dataframe now contains the principal components of the scaled features dataframe
X_test_pca_elbow_df

#Use data after selectfrommodel
model_with_PCA_xgb = xgb.XGBClassifier()
model_with_PCA_xgb.fit(X_train_pca_elbow_df, y_train)
y_pred_pca_elbow_xgb = model_with_PCA_xgb.predict(X_test_pca_elbow_df)
#print performance
precision = average_precision_score(y_test, y_pred_pca_elbow_xgb)
recall = recall_score(y_test, y_pred_pca_elbow_xgb)
F1score = f1_score(y_test, y_pred_pca_elbow_xgb)
print(f"Precision: {precision}; Recall: {recall}; F1 score: {F1score}")