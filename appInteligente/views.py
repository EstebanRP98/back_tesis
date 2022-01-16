import json

import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from appInteligente.Logica import modeloSNN #para utilizar el método inteligente
from django.views.decorators.csrf import csrf_exempt
from io import StringIO

class Clasificacion():
    def determinarAprobacion(request):
        return render(request, "aprobacionCredito.html")

    def predecir(request):
        try:
            print("predecir")
            #Formato de datos de entrada
            center_id = int(request.POST.get('CENTERID'))
            week = float(request.POST.get('WEEK'))
            dia = float(request.POST.get('DAY'))
            meal_id = int(''+request.POST.get('MEALID'))
            #Consumo de la lógica para predecir si se aprueba o no el crédito
            resul=modeloSNN.modeloSNN.preprocesarNuevoCliente(modeloSNN.modeloSNN,center_id=center_id,week=week,dia=dia,meal_id=meal_id)
        except:
            resul='Datos inválidos'
        return render(request, "informe.html",{"e":resul})

    @csrf_exempt
    @api_view(['GET', 'POST'])
    def buscar2(request):
        print(request)
        print('*****************')
        print(request.body)
        print('*****************')
        body = json.loads(request.body.decode('utf-8'))

        resul = modeloSNN.modeloSNN.procesarCSV(modeloSNN.modeloSNN,"Recursos/versionDataMeal")
        #a = np.asarray(dataset)
        #resul2 = resul[['center_id', 'meal_id']].to_numpy()
        resul2 = resul[['center_id', 'meal_id']].reset_index().values
        # question = body.get("question")
        #print(body)
        print(resul2)
        # resul = modeloSNN.modeloSNN.buscarConPLNyKNN(modeloSNN.modeloSNN, question)
        # data = {'result': resul}
        # resp = JsonResponse(data)
        # resp['Access-Control-Allow-Origin'] = '*'
        predictions = {
            'error': '2',
            "message": "ok"
        }
        return Response(predictions)

    def buscarMealByCenterId(self,centerId):
        resul = modeloSNN.modeloSNN.procesarCSV(modeloSNN.modeloSNN, "Recursos/versionDataMeal")
        newrespPrueba = resul.filter(['center_id', 'meal_id'], axis=1)
        original_list = newrespPrueba.loc[newrespPrueba['center_id'] == centerId]['meal_id'].to_list()
        convert_list_to_set = set(original_list)
        original_list = list(convert_list_to_set)
        return original_list

    @csrf_exempt
    @api_view(['GET', 'POST'])
    def buscarMealList(request):
        try:
            print(request)
            print('*****************')
            print(request.body)
            print('*****************')
            body = json.loads(request.body.decode('utf-8'))

            centerId = body.get("centerId")
            week = body.get("week")
            resul = modeloSNN.modeloSNN.procesarCSV(modeloSNN.modeloSNN, "Recursos/versionDataMeal")
            newrespPrueba = resul.filter(['center_id', 'meal_id'], axis=1)

            original_list = newrespPrueba.loc[newrespPrueba['center_id'] == centerId]['meal_id'].to_list()
            convert_list_to_set = set(original_list)
            original_list = list(convert_list_to_set)
            listMealByCenter = original_list
            recipes_all = []
            weekDay = [1, 2, 3, 4, 5, 6, 7]
            for meal in listMealByCenter:
                for day in weekDay:
                    recipesMeal = []
                    recipesMeal.append(centerId)
                    recipesMeal.append(week)
                    recipesMeal.append(day)
                    recipesMeal.append(meal)
                    recipes_all.append(recipesMeal)

            dfResult = modeloSNN.modeloSNN.preprocesarNuevoClienteList(modeloSNN.modeloSNN, recipes_all)
            # dfResult = dfResult.to_json(orient = "records", lines=True)
            # dfResult = dfResult.to_json('temp.json.gz', orient='records', lines=True, compression='gzip')
            df = pd.read_excel('./Recursos/orderComida.xlsx')
            dfMerge = pd.merge(dfResult, df, on="meal_id")
            dfMerge = dfMerge.drop(['code', 'name', 'nameFile', 'originalingredientLines', 'ingredientLines', 'rating',
                                    'attributes.course', 'attributes.cuisine', 'images', 'attribution.url', 'flavors.Piquant',
                                    'flavors.Sour', 'flavors.Salty', 'flavors.Sweet', 'flavors.Bitter',
                                    'flavors.Meaty','bag_of_words'], axis=1)
            print(dfMerge)
            json_list = json.loads(json.dumps(list(dfMerge.T.to_dict().values())))
            predictions = {
                'error': '0',
                'message': 'Successfull',
                'prediction': json_list
            }
        except Exception as e:
                predictions = {
                    'error': '2',
                    "message": str(e)
                }
        return Response(predictions)



    @csrf_exempt
    @api_view(['GET', 'POST'])
    def buscarIngredientesList(request):
        try:
            print(request)
            print('*****************')
            print(request)
            print('*****************')
            # body = json.loads(request.body.decode('utf-8'))

            mealList={}
            for dict in request.data:
                if dict['meal_id'] not in mealList:
                    mealList[dict['meal_id']] = dict['prediccion']
                else:
                    mealList[dict['meal_id']]=mealList[dict['meal_id']]+dict['prediccion']

            recipes = modeloSNN.modeloSNN.webScrappingRecipe(modeloSNN.modeloSNN,mealList)
            recipes = recipes.fillna('')
            json_list = json.loads(json.dumps(list(recipes.T.to_dict().values())))

            predictions = {
                'error': '0',
                'message': 'Successfull',
                'prediction': json_list
            }
        except Exception as e:
            predictions = {
                'error': '2',
                "message": str(e)
            }
        return Response(predictions)

    @csrf_exempt
    @api_view(['GET', 'POST'])
    def buscarIngredientesInfo(request):
        try:
            mealList = {}
            for dict in request.data:
                if dict['meal_id'] not in mealList:
                    mealList[dict['meal_id']] = dict['prediccion']
                else:
                    mealList[dict['meal_id']] = mealList[dict['meal_id']] + dict['prediccion']

            recipes = modeloSNN.modeloSNN.webScrappingRecipe(modeloSNN.modeloSNN, mealList)
            recipes = recipes.fillna('')
            recipes = recipes.filter(['ingredients', 'amount', 'unit'], axis=1)
            recipes['amount'] = recipes.groupby(['ingredients', 'unit'], dropna=False)['amount'].transform(
                'sum')
            recipes = recipes.drop_duplicates()
            decimals = 2
            recipes['amount'] = recipes['amount'].apply(lambda x: round(x, decimals))
            print("recipes", recipes.T.to_dict().values())

            json_list = json.loads(json.dumps(list(recipes.T.to_dict().values())))

            predictions = {
                'error': '0',
                'message': 'Successfull',
                'prediction': json_list
            }
        except Exception as e:
            predictions = {
                'error': '2',
                "message": str(e)
            }
        return Response(predictions)

    @api_view(["POST"])
    def predecirServ(request):
        try:
            centroid = request.data.get('centroid', None)
            week = request.data.get('week', None)
            day = request.data.get('day', None)
            mealid = request.data.get('mealid', None)
            fields = [centroid, week, day, mealid]
            if not None in fields:
                # Datapreprocessing Convert the values to float
                center_id = float(centroid)
                week = float(week)
                dia = float(day)
                meal_id = float(mealid)
                print("predecir")
                # Consumo de la lógica para predecir si se aprueba o no el crédito
                resul = modeloSNN.modeloSNN.preprocesarNuevoCliente(modeloSNN.modeloSNN, center_id=center_id, week=week,
                                                                    dia=dia, meal_id=meal_id)
                predictions = {
                    'error': '0',
                    'message': 'Successfull',
                    'prediction': resul
                }
            else:
                predictions = {
                    'error': '1',
                    'message': 'Invalid Parameters'
                }
        except Exception as e:
                predictions = {
                    'error': '2',
                    "message": str(e)
                }
        return Response(predictions)