from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
import pickle
import os
from sklearn.neighbors import NearestNeighbors
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

app = Flask(__name__)

#data
application_test=pd.read_csv("db/application_test.csv")
application_test_records=application_test.to_dict("records")
application_test_preprocessing=pd.read_csv("db/application_test_preprocessing.csv")
application_test_preprocessing_records=application_test_preprocessing.to_dict("records")
variables_importantes=pd.read_csv("db/variables_importantes.csv")
predictions=pd.read_csv("db/predictions.csv")
shap_values_client=pd.read_csv("db/shap_values_clients.csv")

#modèle

modele = pickle.load(open('db/modele.pkl', 'rb'))

@app.route('/api/projet7/decision', methods=['GET'])
def api_id():
 # Check if an ID was provided as part of the URL.
 # If ID is provided, assign it to a variable.
 # If no ID is provided, display an error in the browser.
  if 'id' in request.args:
      id = int(request.args['id'])
  else:
      return "Error: No id field provided. Please specify an id."
  modele_pred_test=modele.predict(application_test_preprocessing)
  predictions_test=pd.DataFrame()
  predictions_test['SK_ID_CURR']=application_test['SK_ID_CURR']
  predictions_test["TARGET_pred"]=modele_pred_test
  predictions_test.set_index("SK_ID_CURR", inplace=True)
  results=[]
  if predictions_test.loc[id]["TARGET_pred"]== 0:
    results.append("Pret accorde")
  else:
    results.append("Pret refuse")
 # Use the jsonify function from Flask to convert our list of
 # Python dictionaries to the JSON format.
  return jsonify(results)

@app.route('/api/projet7/skids', methods=['GET'])
def get_skids():
  skidcurr_dict=application_test["SK_ID_CURR"].to_dict()
  return jsonify(skidcurr_dict)
 
@app.route('/api/projet7/application_test/all', methods=['GET'])
def json_application_test():
  application_test_records=application_test.to_dict("records")
  return jsonify(application_test_records)

@app.route('/api/projet7/application_test', methods=['GET'])
def json_application_test_id():
 
 if 'id' in request.args:
   id = int(request.args['id'])
 else:
   return "Error: No id field provided. Please specify an id."
 # Create an empty list for our results
 results = []
 # Loop through the data and match results that fit the requested ID.
 # IDs are unique, but other fields might return many results
 for donnees in application_test_records:
   if donnees['SK_ID_CURR'] == id:
     results.append(donnees)
 # Use the jsonify function from Flask to convert our list of
 # Python dictionaries to the JSON format.
 return jsonify(results)
 
@app.route('/api/projet7/application_test_preprocessing/all', methods=['GET'])
def json_application_test_preprocessing():
  application_test_preprocessing_records=application_test_preprocessing.to_dict("records")
  return jsonify(application_test_preprocessing_records)

@app.route('/api/projet7/application_test_preprocessing', methods=['GET'])
def json_application_test_preprocessing_id():
 if 'id' in request.args:
   id = int(request.args['id'])
 else:
   return "Error: No id field provided. Please specify an id."
 # Create an empty list for our results
 results = []
 # Loop through the data and match results that fit the requested ID.
 # IDs are unique, but other fields might return many results
 for donnees in application_test_preprocessing_records:
   if donnees['SK_ID_CURR'] == id:
     results.append(donnees)
 # Use the jsonify function from Flask to convert our list of
 # Python dictionaries to the JSON format.
 return jsonify(results)

@app.route('/api/projet7/variables_importantes', methods=['GET'])
def json_variables_importantes():
  variables_importantes_records=variables_importantes.to_dict("records")
  return jsonify(variables_importantes_records)
 
@app.route('/api/projet7/predictions', methods=['GET'])
def json_predictions():
  predictions_records=predictions.to_dict("records")
  return jsonify(predictions_records)
 
@app.route('/api/projet7/variables_importantes_client', methods=['GET'])
def json_shap_values_client():
  shap_values_clients_records=shap_values_client.to_dict("records")
  return jsonify(shap_values_clients_records)

@app.route('/api/projet7/dataframe_voisins', methods=['GET'])
def json_dataframe_voisins():
  if 'id' in request.args:
   id = int(request.args['id'])
  else:
    return "Error: No id field provided. Please specify an id."
  application_test_preprocessing["TARGET"]=predictions["TARGET_pred"].values
  labels_test=application_test_preprocessing["TARGET"]
  application_test_preprocessing.set_index("SK_ID_CURR", inplace=True)
  neigh = NearestNeighbors(n_neighbors=20)
  neigh.fit(np.array(application_test_preprocessing), labels_test)
  ind=neigh.kneighbors([application_test_preprocessing.loc[id].values])
  ind_neigh=ind[1]
  l=[]
  for i in range(ind_neigh.shape[1]):
    l.append(ind_neigh[0,i])
    application_test_reset_index=application_test_preprocessing.reset_index()
    dataframe_voisins=pd.DataFrame(columns=application_test_reset_index.columns)
    for i in range(len(application_test_reset_index)):
        if i in sorted(l):
            dataframe_voisins=pd.concat([dataframe_voisins,pd.DataFrame(data=np.array([application_test_reset_index.loc[i]]),columns=application_test_reset_index.columns)],ignore_index=True)
            dataframe_voisins["SK_ID_CURR"].astype(int)
    dataframe_voisins.set_index("SK_ID_CURR",inplace=True)
    dataframe_voisins.drop(index=id,inplace=True)
  return jsonify(dataframe_voisins.mean(axis=0).to_dict())
  
if __name__ == '__main__':
    app.run()


#app.run(port=os.environ.get("PORT", 8080))