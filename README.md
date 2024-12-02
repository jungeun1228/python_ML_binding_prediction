# python_ML_binding_prediction
2024 Bootcamp final project from Kaggle competition: NeurIPS 2024 - Predict New Medicines with BELKA

This prediction model for new medicine was developed based on "Big Encoded Library for Chemical Assessment (BELKA)". Dataset for SMILES of individual molecule and protein types were provided in Kaggle competition (https://www.kaggle.com/c/leash-BELKA/overview) and used to train this model for predicting molecule-protein interactions. <br/>


1. Training the prediction model using machine learning<br/>
Partial data sets were used due to memory issue (TPU v2-8 was used in Google Colab), hence, 200,000 pairs of molecule-protein were used as input data (original data set: ~ 100 milion molecule with 3 different protein).
Input data was processed into chemical fingerprints (RDkit library) or binary data (one hot encoding).<br/>
80% of the datasets were applied for training model and the rest 20% of the data sets were used for model evaluation. Model was optimized via feature selection and model selection processes, which resulted in 90% accuracy in the final model.<br/>
[Optimization process]<br/>
-model selection: different model compared based on their precision, F1 score (KNN, xgboost, SVM, etc.) -> xgboost selected based on their performance<br/>
-PCA analysis: performed for preprocessing but did not improve the model performance; hence not included in the final model<br/>
<br/>


2. Streamlit application<br/>
Finalizd model was applied for user-friendly protein-medicine binding prediction using Streamlit.
![image](https://github.com/user-attachments/assets/3a534945-0c13-4c6d-93ed-6a5f7a83e9ae)
When you upload csv file as input data, you will get protein binding prediction as a result as shown below.
Input data: (1) SMILES for molecules (molecule_smiles), (2) name of target protein (protein_name).
![image](https://github.com/user-attachments/assets/03484a0d-aa9e-4956-906f-4bfa8a446587)
Output data: binding prediction - column "binding"
![image](https://github.com/user-attachments/assets/ab4e86c9-97db-4012-93b5-7c6ac48a453c)

<br/>
References:<br/>
https://www.kaggle.com/competitions/leash-BELKA<br/>
https://medium.com/@icarusabiding/neurips-2024-belka-challenge-innovating-small-molecule-binding-prediction-593aefc7a310
