#Importation de la librairie streamlit
import streamlit as st
#Importation de la librairie Pandas
import pandas as pd
#Importation de la librairie seaborn
import seaborn as sns
#Importation de la librairie pyplot
import matplotlib.pyplot as plt
#Importation de la librairie Sklearn
import sklearn as sk
from sklearn import cluster
#Importation de la librairie de l'Analyse de la composante principale
from sklearn.decomposition import PCA
#Importation de la librairie pickle
import pickle
#Importation de la librairie IO
from io import StringIO

def detection_faux_billets():
    X = df_billets.drop(['id'], axis=1)    
    X = pd.DataFrame(std_scale.transform(X), columns=X.columns)

    y = clf.predict(X)
    predictions = []
    for i in range(0, len(y)):
        predictions.append(y[i])
 
    predictions = pd.concat([
        pd.DataFrame([predictions]).rename(index={0: 'Prédiction'}).T.replace({False: 'Faux billet', True: 'Vrai billet'}),
        pd.DataFrame(clf.predict_proba(X)).rename(columns={0: 'Probabilité faux billet', 1: 'Probabilité vrai billet'})], axis=1)
    predictions['id'] = df_billets['id'].unique() 

    n_components = 2
    reduced = pca.transform(X)
    for i in range(0, n_components):
        predictions['PC' + str(i + 1)] = reduced[:, i]   
    plt.figure(figsize=(6, 5))
    ax = sns.scatterplot(data=predictions.sort_values(by=['Prédiction']), x='PC1', y='PC2', markers=['X','o'],
        hue='Prédiction', style='Prédiction', s=100)
    predictions.apply(lambda x: ax.text(x['PC1']+0.1, x['PC2'], x['id']), axis=1)

    plt.title('Projection des billets sur 2 dimensions')
    st.pyplot(plt.gcf())
    st.write(predictions.iloc[:,:-2])

st.image("logo_oncfm.png", use_column_width='always', caption='Organisation nationale contre le faux-monnayage')

st.markdown(
    """
    <style>
    .header {
        text-align: center;
        font-size: xx-large;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<h1 class="header">Programme de détection de faux billets</h1>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df_billets = pd.read_csv(uploaded_file)

    training_dict = pickle.load(open('training_model.pkl', 'rb'))
    std_scale = training_dict[0]
    clf = training_dict[1]
    pca = training_dict[2]

    detection_faux_billets()
