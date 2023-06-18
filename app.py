import streamlit as st
import os
from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import pickle
import pandas as pd


st.title('Detecció automàtica de melanoma')

st.info('Les prediccions del model no substitueixen el diagnòstic d un dermatòleg.', icon="ℹ️")
st.cache_data(show_spinner= False)
def loadModel():
    model = tf.keras.models.load_model('C:\\Users\\Nitropc\\Desktop\\TFG\\jpeg_binary\\models\\EfficientNetV2S_97%.h5')
    model_gaussian = pickle.load(open("C:\\Users\\Nitropc\\Desktop\\TFG\\finalized_modelGaussian.sav", 'rb'))
    return model, model_gaussian


st.header('Dades Pacient')
col1,col2,col3 = st.columns(3)
sexe = col1.selectbox('Selecciona el sexe:',['Dona','Home'])
edat = col2.number_input('Selecciona l edat aproximada',step = 5, min_value = 0)
ubicacio = col3.selectbox('Selecciona el lloc de la lesió:',['Tors','Extremitats superiors','Extremitats inferiors','Coll/Cap','No especificat','Palmells mans/ Plantes dels peus','Zona oral/genital'])

if sexe == 'Dona':
    codificació_sexe = 0
else:
    codificació_sexe = 1
        
if ubicacio == 'Coll/Cap':
    codificacio_lloc = 0
elif ubicacio == 'Extremitats inferiors':
    codificacio_lloc = 1
elif ubicacio == 'Zona oral/genital':
    codificacio_lloc = 2
elif ubicacio == 'Palmells mans/ Plantes dels peus':
    codificacio_lloc = 3
elif ubicacio == 'Tors':
    codificacio_lloc = 4
elif ubicacio == 'No especificat':
    codificacio_lloc = 5
elif ubicacio == 'Extremitats superiors':
    codificacio_lloc = 6
        
pacient = pd.DataFrame()
pacient['sex'] = None
pacient['age_approx'] = None
pacient['anatom_site_general_challenge'] = None
pacient['Probabilitat'] = None

model, model_gaussian = loadModel()

st.header('Imatge Lesió Cutània')
uploaded_file = st.file_uploader("Selecciona la imatge per predir:")
if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    #st.write(bytes_data)
    image_data = uploaded_file.getvalue()
    #st.write("filename:", uploaded_file.name[-3:])
    if uploaded_file.name[-3:] == 'jpg':
        
        st.image(image_data)
        # To read file as bytes:
        im= Image.open(uploaded_file)
        img= np.asarray(im)
        #img = cv2.imread(image_data)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resize = tf.image.resize(img, (256,256))
        yhat = model.predict(np.expand_dims(resize, 0))
        #st.write(yhat[0][0])
        
        
        pacient.loc[len(pacient)] = [codificació_sexe,edat,codificacio_lloc,yhat[0][0]]

        llista_sexe = [codificació_sexe]
        llista_edat = [edat]
        llista_lloc = [codificacio_lloc]
        llista_prob = [yhat[0][0]]

        if st.button('Realitzar diagnòstic'):
            val_show = round(yhat[0][0],2)
            print(val_show)
            st.metric('Probabilitat de la imatge de ser maligne',val_show)
            #st.write(pacient)
            predicció = model_gaussian.predict(pacient)
            if predicció == 0:
                st.success('El diagnòstic ha determinat que la lesió és benigne.')
            else:
                st.error('El diagnòstic ha determinat que la lesió es tracta de melanoma.')





