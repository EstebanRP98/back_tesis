import math
from fractions import Fraction

import numpy as np
from django.urls import reverse
import pandas as pd
import requests
from bs4 import BeautifulSoup #scraping
from pandas import read_csv
from sklearn.pipeline import Pipeline
from tensorflow.python.keras.models import load_model, model_from_json
from keras import backend as K
from appInteligente.Logica import modeloSNN
import pickle

class modeloSNN():
    """Clase modelo Preprocesamiento y SNN"""
    #Función para cargar preprocesador
    def cargarPipeline(self,nombreArchivo):
        with open(nombreArchivo+'.pickle', 'rb') as handle:
            pipeline = pickle.load(handle)
        return pipeline
    #Función para cargar red neuronal
    def cargarNN(self,nombreArchivo):
        model = load_model(nombreArchivo+'.h5')
        print("Red Neuronal Cargada desde Archivo")
        return model

    #funcion para leer el csv
    def procesarCSV(self, nombreArchivo):
        df = pd.read_csv(nombreArchivo+'.csv', sep=';', index_col=0, encoding='latin-1')
        df["fecha"] = pd.to_datetime(df["fecha"])
        df = df.sort_values(["fecha"])
        dfnew = pd.DataFrame(
            list(zip(df["center_id"], df["week"], df["day"], df["meal_id"], df["timestamp"], df["num_orders"])),
            columns=['center_id', 'week', 'day', 'meal_id', 'timestamp', 'Num_Orders'])
        print('Total rows: {}'.format(len(dfnew)))
        return dfnew



    #Función para integrar el preprocesador y la red neuronal en un Pipeline
    def cargarModelo(self):
        #Se carga el Pipeline de Preprocesamiento
        nombreArchivoPreprocesador='Recursos/pipePreprocesadores'
        pipe=self.cargarPipeline(self,nombreArchivoPreprocesador)
        print('Pipeline de Preprocesamiento Cargado')
        cantidadPasos=len(pipe.steps)
        print("Cantidad de pasos: ",cantidadPasos)

        return pipe


    def predecirNuevoCliente(self,ESTADOCUENTACORRIENTE='A12', PLAZOMESESCREDITO=6, HISTORIALCREDITO='A34', PROPOSITOCREDITO='A43',
                                MONTOCREDITO=1169, SALDOCUENTAAHORROS='A65', TIEMPOACTUALEMPLEO='A75', TASAPAGO=4,
                                ESTADOCIVILYSEXO='A93', GARANTE='A101', TIEMPORESIDENCIAACTUAL=4, ACTIVOS='A121', EDAD=67,
                                VIVIENDA='A152', CANTIDADCREDITOSEXISTENTES=2, EMPLEO='A173', CANTIDADPERSONASAMANTENER=2,
                                TRABAJADOREXTRANJERO='A201'):
        pipe=self.cargarModelo(self)
        cnames=['ESTADOCUENTACORRIENTE','PLAZOMESESCREDITO','HISTORIALCREDITO','PROPOSITOCREDITO','MONTOCREDITO',
                'SALDOCUENTAAHORROS','TIEMPOACTUALEMPLEO','TASAPAGO','ESTADOCIVILYSEXO','GARANTE','TIEMPORESIDENCIAACTUAL',
                'ACTIVOS','EDAD','VIVIENDA','CANTIDADCREDITOSEXISTENTES','EMPLEO','CANTIDADPERSONASAMANTENER',
                'TRABAJADOREXTRANJERO']
        Xnew=[ESTADOCUENTACORRIENTE,PLAZOMESESCREDITO,HISTORIALCREDITO,PROPOSITOCREDITO,MONTOCREDITO,SALDOCUENTAAHORROS,
              TIEMPOACTUALEMPLEO,TASAPAGO,ESTADOCIVILYSEXO,GARANTE,TIEMPORESIDENCIAACTUAL,ACTIVOS,EDAD,VIVIENDA,
              CANTIDADCREDITOSEXISTENTES,EMPLEO,CANTIDADPERSONASAMANTENER,TRABAJADOREXTRANJERO]
        Xnew_Dataframe = pd.DataFrame(data=[Xnew],columns=cnames)
        print("mapa",Xnew_Dataframe)
        pred = (pipe.predict(Xnew_Dataframe) > 0.5).astype("int32")
        pred = pred.flatten()[0]# de 2D a 1D
        if pred==1:
            pred='Aprobado. Felicidades =)'
        else:
            pred='Negado. Lo sentimos, intenta en otra ocasión'
        return pred

    def preprocesarNuevoCliente(self,center_id=11, week=1, dia=2, meal_id=2707):
        print("cargando pipe.....")
        pipe = self.cargarModelo(self)
        print("terminado pipe.....")
        cnames = ['center_id', 'week', 'day', 'meal_id']
        Xnew = [center_id, week, dia, meal_id]
        Xnew_Dataframe = pd.DataFrame(data=[Xnew], columns=cnames)

        Xnew_Preprocesado = pipe.transform(Xnew_Dataframe)
        print("preprocesado data...")
        self.procesarCSV(self, "Recursos/versionDataMeal")
        # load model architecture
        print("cargando red neuronal")
        with open('Recursos/modeloRedNeuronalBase.json', 'r') as f:
            model = model_from_json(f.read())
        # load model weights
        model.load_weights('Recursos/modeloRedNeuronalBase.h5')
        print("cargada red")
        print(Xnew_Preprocesado)
        Xnew = Xnew_Preprocesado.reshape((Xnew_Preprocesado.shape[0], 1, Xnew_Preprocesado.shape[1]))
        pred = model.predict(Xnew)[0]
        return pred


    def preprocesarNuevoClienteList(self,mealList):
        print("cargando pipe.....")
        pipe = self.cargarModelo(self)
        print("terminado pipe.....")
        cnames = ['center_id', 'week', 'day', 'meal_id']
        Xnew = mealList
        Xnew_Dataframe = pd.DataFrame(data=Xnew, columns=cnames)
        print(len(Xnew_Dataframe))
        Xnew_Preprocesado = pipe.transform(Xnew_Dataframe)
        print("preprocesado data...")
        self.procesarCSV(self, "Recursos/versionDataMeal")
        # load model architecture
        print("cargando red neuronal")
        with open('Recursos/modeloRedNeuronalBase.json', 'r') as f:
            model = model_from_json(f.read())
        # load model weights
        model.load_weights('Recursos/modeloRedNeuronalBase.h5')
        print("cargada red")
        print(Xnew_Preprocesado)
        Xnew = Xnew_Preprocesado.reshape((Xnew_Preprocesado.shape[0], 1, Xnew_Preprocesado.shape[1]))
        pred = model.predict(Xnew)
        dfpredict = pd.DataFrame(np.ceil(pred), columns=['prediccion'])
        predDF=pd.concat([Xnew_Dataframe, dfpredict], axis=1)
        return predDF

    def replace(text):
        if text is not None:
            text = text.replace('\xa0', ' ')
        return text

    def getRecipeDataframe(self,data):
        recipes_all = []
        for index, row in data.iterrows():
            url = row['attribution.url']
            mealId = row['meal_id']
            r = requests.get(url)
            html = BeautifulSoup(r.content, "html.parser")
            name = html.find('h1', class_="recipe-title font-bold h2-text primary-dark").text
            serving = html.find('div', class_="servings micro-caps font-bold").text
            for i in html.find_all('div', class_="add-ingredient"):
                amount = i.find('span', {"class": "amount"})
                if amount is not None:
                    amount = amount.text
                ingredient = i.find('span', {"class": "ingredient"}).text
                unit = i.find('span', {"class": "unit"})
                if unit is not None:
                    unit = unit.text

                # ing = i.text.strip()
                # ing = i.find('li', {"class":"IngredientLine"}).text
                recipes_all.append({"mealId": mealId, "name": name, "serving": serving, "amount": self.replace(amount),
                                    "ingredients": self.replace(ingredient), "unit": self.replace(unit)})
        base_df = pd.DataFrame(recipes_all)
        base_df = base_df.assign(serving=lambda x: x['serving'].str.extract('(\d+)'))
        base_df.to_excel(
            r'Recursos\export_dataframe.xlsx', index=False,
            header=True)
        return base_df

    def webScrappingRecipe(self,mealId):
        df = pd.read_excel('Recursos/orderComida.xlsx')
        # recipesFrame2 = self.getRecipeDataframe(self,df)
        recipesFrame2 = pd.read_excel('Recursos/export_dataframe.xlsx')
        recipesFrame2["amount"].fillna(1, inplace=True)
        recipesFrame2["amount"] = recipesFrame2["amount"].replace('None', 1, regex=True)
        recipesFrame2["amount"] = recipesFrame2["amount"].replace('', 1, regex=True)
        recipesFrame2['amount'] = recipesFrame2['amount'].astype(str)
        recipesFrame2['amount'] = recipesFrame2.amount.apply(
            lambda x: sum([float(Fraction(i)) for i in x.rstrip().lstrip().split(' ')]))
        recipesFrame2['amount'].replace({None: 1})
        appended_data = []

        for k, v in mealId.items():
            recipesNew = recipesFrame2.loc[recipesFrame2['mealId'] == k]
            recipesNew['serving'] = recipesNew['serving'].astype(float)
            recipesNew['amount'] = recipesNew['amount'].astype(float)
            val = recipesNew['serving'].values[0]
            total = v / val
            total = math.ceil(total)
            recipesNew["amount"] = total * recipesNew["amount"]
            appended_data.append(recipesNew)
        recipes = pd.concat(appended_data)
        return recipes