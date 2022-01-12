import streamlit as st
from PIL import Image
import requests
import json
import pandas as pd
import seaborn as sns
import sys
import os
import shap
import time
import pickle
import matplotlib.pyplot as plt
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objs as go
import pip_api


def main():

    #API_URL = "https://apiprojet7.herokuapp.com/api/projet7/"
    API_URL ="http://127.0.0.1:5000/api/projet7/"
    # Liste des numéros de client
    @st.cache
    def get_sk_id_list():
        # URL de la liste des numéros des clients
        DATA_API_URL = API_URL + "skids"
        # Sauvegarder la réponse de la requête sur l'API
        response = requests.get(DATA_API_URL, auth=('user', 'pass'))
        # Conversion du JSON format en dictionnaire Python
        data = response.json()
        series=pd.Series(data.values())
        SK_IDS=list(series.values)
        return SK_IDS
    
    @st.cache
    def get_application_test():
        # URL de la table application_test
        DATA_API_URL = API_URL + "application_test/all"
        # Sauvegarder la réponse de la requête sur l'API 
        response = requests.get(DATA_API_URL, auth=('user', 'pass'))
        # Conversion du JSON format en dictionnaire Python 
        data = response.json()
        application_test=pd.DataFrame(data)
        return application_test        

    #Données personnelles
    @st.cache
    def get_donnees_client(select_sk_id):
        # URL des données d'un client donné dans la table application_test
        DATA_API_URL = API_URL + "application_test?id=" + str(select_sk_id)
        # Sauvegarder la réponse de la requête sur l'API
        response = requests.get(DATA_API_URL, auth=("user", "pass"))
        # Conversion du JSON format en dictionnaire Python
        data=response.json()  
        data_cust = pd.Series(data[0]).rename(select_sk_id)
        return data_cust
    
    @st.cache
    def get_application_test_preprocessing():
        # URL de la table application_test_preprocessing
        DATA_API_URL = API_URL + "application_test_preprocessing/all"
        # Sauvegarder la réponse de la requête sur l'API
        response = requests.get(DATA_API_URL, auth=('user', 'pass'))
        # Conversion du JSON format en dictionnaire Python
        data = response.json()
        application_test_preprocessing=pd.DataFrame(data)
        #application_test_preprocessing.set_index("SK_ID_CURR", inplace=True)
        return application_test_preprocessing  
        
    @st.cache
    def get_donnees_client_preprocessing(select_sk_id):
        # URL des données d'un client donné dans la table application_test_preprocessing
        DATA_API_URL = API_URL + "application_test_preprocessing?id=" + str(select_sk_id)
        # Sauvegarder la réponse de la requête sur l'API
        response = requests.get(DATA_API_URL, auth=("user", "pass"))
        # Conversion du JSON format en dictionnaire Python
        data=response.json()  
        data_cust_prepro = pd.Series(data[0]).rename(select_sk_id)
        return data_cust_prepro
    
    @st.cache
    def get_predictions():
        # URL de la table prédictions
        DATA_API_URL = API_URL + "predictions"
        # Sauvegarder la réponse de la requête sur l'API
        response = requests.get(DATA_API_URL, auth=("user", "pass"))
        # Conversion du JSON format en dictionnaire Python
        data=response.json()  
        predictions=pd.DataFrame(data)
        predictions.set_index("SK_ID_CURR", inplace=True)
        return predictions
        
    #Importance de chaque variable d'un client donné
    @st.cache
    def get_shap_values():
        # URL de la table de l'importance des variables pour tous les clients
        DATA_API_URL = API_URL + "variables_importantes_client"
        # Sauvegarder la réponse de la requête sur l'API
        response = requests.get(DATA_API_URL, auth=("user", "pass"))
        # Conversion du JSON format en dictionnaire Python
        data=response.json()  
        predictions=pd.DataFrame(data)
        return predictions
    
    #Variables importantes du modèle    
    @st.cache
    def get_variables_importantes():
        # URL de la liste de toutes les variables importantes selon le modèle
        DATA_API_URL = API_URL + "variables_importantes"
        # Sauvegarder la réponse de la requête sur l'API
        response = requests.get(DATA_API_URL, auth=("user", "pass"))
        # Conversion du JSON format en dictionnaire Python
        data=response.json()  
        variables=pd.DataFrame(data)
        return variables
    
    # 20 variables les plus importantes selon le modèle
    @st.cache(allow_output_mutation=True)
    def get_features_importances():
        # URL de la liste de toutes les variables importantes selon le modèle
        DATA_API_URL = API_URL + "variables_importantes"
        # Sauvegarder la réponse de la requête sur l'API
        response = requests.get(DATA_API_URL, auth=("user", "pass"))
        # Conversion du JSON format en dictionnaire Python
        data=response.json() 
        dataframe=pd.DataFrame(data)        
        variables_importantes = list(dataframe["Features"][0:20])
        return variables_importantes  

    # Score d'un client donné
    @st.cache
    def get_cust_scoring(select_sk_id):
        # URL de la table prédictions
        DATA_API_URL = API_URL + "predictions"
        # Sauvegarder la réponse de la requête sur l'API
        response = requests.get(DATA_API_URL, auth=("user", "pass"))
        # Conversion du JSON format en dictionnaire Python
        data=response.json()  
        dataframe=pd.DataFrame(data)
        dataframe.set_index("SK_ID_CURR", inplace=True)
        # Valeur du score: probabilité de défaut, et target: décision du prêt
        score = dataframe.loc[select_sk_id]["prediction"]
        target=dataframe.loc[select_sk_id]["TARGET_pred"]
        return score, target

    #Variables importantes par client
    @st.cache
    def get_features_importances_client(select_sk_id):
        # URL de la table de l'importance des variables pour tous les clients
        DATA_API_URL = API_URL + "variables_importantes_client"
        # Sauvegarder la réponse de la requête sur l'API
        response = requests.get(DATA_API_URL, auth=("user", "pass"))
        # Conversion du JSON format en dictionnaire Python
        data=response.json()  
        dataframe_client=pd.DataFrame(data)
        dataframe_client.rename(columns={"Unnamed: 0":"SK_ID_CURR"},inplace=True)
        vals= abs(dataframe_client[dataframe_client["SK_ID_CURR"]==select_sk_id].iloc[0,1:].values)
        feature_importance = pd.DataFrame(list(zip(dataframe_client.columns,vals)),columns=['col_name','feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
        return feature_importance

    @st.cache
    def get_data_neigh(select_sk_id):
        # URL de la table des voisins d'un client donné
        DATA_API_URL = API_URL + "dataframe_voisins?id=" + str(select_sk_id)
        # Sauvegarder la réponse de la requête sur l'API
        response = requests.get(DATA_API_URL, auth=("user", "pass"))
        # Conversion du JSON format en dictionnaire Python
        data=response.json()  
        data_neigh = pd.Series(data).rename(select_sk_id)
        return data_neigh 
        
    #Liste des 20 variables importantes selon le modèle, ajouté des 20 variables les plus importantes du client donné, 40 variables au total.
    @st.cache
    def get_all_features(select_sk_id):
        feature_importance=get_features_importances()
        features_importance_clients=get_features_importances_client(select_sk_id)
        feature_importance.extend(features_importance_clients.iloc[0:20]["col_name"])
        variables = list(set(feature_importance))
        return variables
            
        
    # Configuration de la page streamlit
    st.set_page_config(page_title='Application prêt bancaire dashboard',
                       page_icon='random',
                       layout='centered',
                       initial_sidebar_state='auto')

    # Titre du dashboard
    st.title('Dashboard application prêt bancaire')
    st.header("Cyril GODET - Projet 7")
    path = "logo.png"
    image = Image.open(path)
    st.sidebar.image(image, width=180)

    
    # Sélection du numéro du client

    SK_IDS = get_sk_id_list()
    select_sk_id = st.sidebar.selectbox('Sélectionner le numéro du client:', SK_IDS)
    st.write('Vous avez sélectionné: ', select_sk_id)

    # Données client (non transformé et transformé)
    feature_importance_client=get_features_importances_client(select_sk_id)
    X_cust= get_donnees_client(select_sk_id)
    X_cust_proc=get_donnees_client_preprocessing(select_sk_id)
    X_neigh=get_data_neigh(select_sk_id)
    application_test_preprocessing_update=get_application_test_preprocessing()

    # Variables importantes du client

    if st.sidebar.checkbox("Données client"):

        st.header("Données client")
        
        st.markdown("***Variables importantes client***")

        format_dict = {'cust prepro': '{:.2f}',
                       '20 neigh (mean)': '{:.2f}',
                       '20k samp (mean)': '{:.2f}'}
                              
        
        disp_cols_client = st.multiselect('Choisir les variables à afficher:',
                                   feature_importance_client.iloc[0:20]["col_name"],
                                   default=feature_importance_client.iloc[0:20]["col_name"])
        
        disp_cols_not_prepro = [col for col in disp_cols_client \
                                if col in X_cust.index.to_list()]
        disp_cols_prepro = [col for col in disp_cols_client \
                            if col in X_cust_proc.index.to_list()]
                            
     # Affichage des données du client
        df_display = pd.concat([X_cust.loc[disp_cols_not_prepro].rename('Valeur client'),
                                X_cust_proc.loc[disp_cols_prepro].rename('Valeur client après transformation'),
                                X_neigh.loc[disp_cols_prepro].rename("Moyenne 20 plus proches voisins, après transformation")], axis=1)

        st.dataframe(df_display.style.format(format_dict)
                                     .background_gradient(cmap='seismic',
                                                          axis=0, subset=None,
                                                          text_color_threshold=0.2,
                                                          vmin=-1, vmax=1)
                                     .highlight_null('lightgrey'))
        
    # Scoring et décision du prêt

    if st.sidebar.checkbox("Scoring et décision du modèle", key=38):

        st.header("Scoring et décision du modèle")
        
        score, target= get_cust_scoring(select_sk_id)
        
        colors = ['green', 'red']
        
        fig=go.Figure(data=[go.Pie(labels=['Remboursement', 'Défaut paiement'],
                        values=[1-score,score],
                        marker=dict(colors=colors),
                        textinfo='label+value',
                        hoverinfo='label+value+percent',
                        textfont=dict(size=13),
                        hole=.7,
                        rotation=45
                        )])
        
        st.plotly_chart(fig)
        

        if target==0:
            decision = "Prêt accepté" 
        else:
            decision = "Prêt refusé"
        
        st.write('Decision:', decision)
        
        expander = st.expander("Concernant le modèle de classification")

        expander.write("La prédiction a été effectuée en utilisant le modèle de classification LightGBM (Light Gradient Boosting Machine)")
    
    all_variables=get_all_features(select_sk_id)
        
    importance_variables=get_variables_importantes()

    if st.sidebar.checkbox("Importance des variables", key=29):

        st.header("Variables importantes pour la prédiction")
        
        fig= plt.figure()
        
        ax = fig.add_subplot(111)

        plt.barh(list(get_variables_importantes().iloc[0:20]['Features']), list(get_variables_importantes().iloc[0:20]['Importances']))
        
        plt.xlabel("Importance variables")
        
        st.pyplot(fig)
    
    if st.sidebar.checkbox("Boxplot des variables principales"):
        
        st.header("Boxplot des variables principales")
        
        st.markdown("***Variables importantes modèle***")
        
        disp_cols = st.multiselect('Choisir les variables à afficher:',
                                   get_features_importances()[0:10],
                                   default=get_features_importances()[0:10])
        
        df=application_test_preprocessing_update
        
        df = df.melt(id_vars=['TARGET'],  
                                   value_vars=disp_cols,
                                   var_name="variables",
                                   value_name="values")
                
        fig, ax = plt.subplots(figsize=(15,8))
        
        sns.boxplot(data=df, x="variables", y="values", hue="TARGET", linewidth=1,
                width=0.4, palette=['tab:green', 'tab:red'], showfliers=False, saturation=0.5,
                ax=ax)
        
        plt.xticks(rotation = 'vertical')
                
        st.pyplot(fig)
        
        expander = st.beta_expander("Explications sur les boxplots")

        expander.write("Les boxplots ont été réalisés sur tout l'échantillon")

    if st.sidebar.checkbox('Nuage de points des variables principales'):

        st.header("Nuage de points des variables principales")
    
        option1 = st.selectbox(
        'Sélectionner la première variable',
        all_variables)
    
        option2= st.selectbox(
        'Sélectionner la deuxième variable',
        all_variables)
        
        df_nuage=application_test_preprocessing_update
    
        df_nuage_boxplot = df_nuage.melt(id_vars=['TARGET'],  
                                   value_vars=option1,
                                   var_name="variables",
                                   value_name="values")
                                   
        if option1==option2:
    
            fig, ax = plt.subplots(figsize=(15,8))
        
            sns.boxplot(data=df_nuage_boxplot, x="variables", y="values", hue="TARGET", linewidth=1,
                width=0.4, palette=['tab:green', 'tab:red'], showfliers=False, saturation=0.5,
                ax=ax)
        
            st.pyplot(fig)
    
        else:
    
            fig, ax = plt.subplots(figsize=(15,8))
    
            sns.scatterplot(data=df_nuage, x=option1, y=option2, hue="TARGET", palette=['tab:green', 'tab:blue'], legend=False, ax=ax)
        
            plt.scatter(data=df_nuage[df_nuage["SK_ID_CURR"]==select_sk_id], x=option1, y=option2, c="red")
        
            red_patch = mpatches.Patch(color='green', label='Prêt accepté')
        
            blue_patch = mpatches.Patch(color='blue', label='Prêt refusé')
        
            green_patch = mpatches.Patch(color='red', label='Client')
        
            patches = [red_patch, blue_patch, green_patch]
        
            ax.legend(handles=patches,loc='upper right')
        
            st.pyplot(fig)
    
if __name__ == '__main__':
    main()