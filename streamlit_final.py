import streamlit as st
import pandas as pd
import numpy as np
# pip install rdkit
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
# pip install xgboost
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel

st.title("New medicine prediction")
# Custom CSS to change background color
st.markdown(
    """
    <style>
    body {
        background-color: #f5ede6;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Load and display the banner image
banner_image = "banner_image.jpg"  # Replace with your image path
st.image(banner_image, use_column_width=True)
st.write("""
### Project description
This prediction model for new medicine was developed based on "Big Encoded Library for Chemical Assessment (BELKA)". Dataset for SMILES of individual molecule and protein types were provided in Kaggle competition (https://www.kaggle.com/c/leash-BELKA/overview) and used to train this model for predicting molecule-protein interactions. \n\n
Please upload csv file below as input data and you need two types of data: (1) SMILES for molecules (molecule_smiles), (2) name of target protein (protein_name). Please make sure that you have the same column names as shown below in the example. \n\n
Here is an example of input data:
""")
# Load and display the example image
input_example_image = "input_example_image.jpg"  # Replace with your image path
st.image(input_example_image, use_column_width=True)

#
def X_conversion(input_df):
  import pandas as pd
  input_df = pd.DataFrame(input_df)
  # pip install rdkit
  import rdkit
  from rdkit import Chem
  from rdkit.Chem import AllChem
  def generate_ecfp(molecule, radius=2, bits=2048):
    if molecule is None:
        return None
    return list(AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=bits))

  input_df['molecule'] = input_df['molecule_smiles'].apply(Chem.MolFromSmiles)
  input_df['ecfp'] = input_df['molecule'].apply(generate_ecfp)

  #[2]One-hot encoding for protein type
  from sklearn.preprocessing import OneHotEncoder
  my_onehot_encoder = OneHotEncoder(sparse_output=False)  #initialize
  my_onehot_encoder.fit(input_df['protein_name'].values.reshape(-1, 1))  #fit
  protein_onehot = my_onehot_encoder.transform(input_df['protein_name'].values.reshape(-1, 1))  #transform

  #[3] combine fingerprint and one-hot encoded protein features
  X = [ecfp + list(protein) for ecfp, protein in zip(input_df['ecfp'].tolist(), protein_onehot.tolist())]
  X = pd.DataFrame(X, columns = range(0,2051))
  return X


# load model
import pickle
# X_conversion = pickle.load(open('X_conversion_function.sav', 'rb'))
feature_selection = pickle.load(open('X_feature_selection.sav', 'rb'))
best_xgb_model = pickle.load(open('best_xgb_model.sav', 'rb'))
# File uploader widget
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the file using pandas
    input_df = pd.read_csv(uploaded_file)
    input_ori_df = input_df.copy()
    # Display the DataFrame
    # st.write("Uploaded CSV file:")
    # st.dataframe(input_df)
    #Convert input df
    input_df = X_conversion(input_df)
    input_df = feature_selection.transform(input_df)
    #Run model
    prediction = best_xgb_model.predict(input_df)
    input_ori_df = pd.DataFrame(input_ori_df)
    prediction = pd.DataFrame(prediction)
    output_df = pd.concat([input_ori_df,prediction], axis = 1)
    output_df.rename(columns={output_df.columns[2]:'binding'}, inplace = True)
    # output_df = pd.DataFrame(output_df, columns=['molecule_smiles','protein_name','Value'])
    st.write("Here is the prediction for binding! Value 1 means that the molecule is binding to protein while value 0 means no binding.")
    st.dataframe(output_df)