#paquetes necesarios para el funcionamiento del programa
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import Optional
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn import linear_model
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from datetime import datetime
from datetime import timedelta

import requests
import pickle
import psycopg2

app = FastAPI()


#**********************************************************************************************************************************************
#*****************************************************************PETICIONES_GET***************************************************************
#**********************************************************************************************************************************************

@app.get("/")#cuando se realiza un get en la api no se realiza ninguna accion
def read_root():
	return {"No se realiza ninguna acción"}


@app.get("/entrenar/")#get que realiza el entrenamiento del los modelos guardando los mejores de cada tipo y el mejor

def read_root(base_de_datos: str, experimental: Optional[bool] = False):#se debe pasar la base de datos a la que se desea conectar

	if not experimental:
		#creamos el fichero a entrenar con los datos que deseamos de la base de datos, aparte se actualiza la base de datos
		if 200 != crear_fichero_entrenamiento(base_de_datos):#funcion donde se actualiza la base de datos y se crea el fichero con los datos a trabajar
			return {"Error": "Error al crear el fichero con patrones"}

		train_file = "DatasetGeneracionDeBD.csv"
	
	else:
		patrones_get = requests.get("http://obtencion-datos-generacion:8008/experimental")
		df = pd.DataFrame(patrones_get.json())
		df.pop('fecha')
		df.pop('ubicacion')
		df.pop('idema')
		df.to_csv('experimental.csv',index=False)
		train_file = "experimental.csv"
		

	#llamamos a funcion read_data para extraer los datos del fichero csv a un formato con el que podamos trabajar
	Dataset, train_inputs, train_outputs, test_inputs, test_outputs, scaler= read_data(train_file)

	#comprobamos si ha ocurrido algun error
	if train_inputs is None or train_outputs is None or test_inputs is None or test_outputs is None:
		print("Error con el fichero")
		return {"Error con el fichero"}

	listaMSE = np.zeros(shape=(4), dtype=float)

	#creamos una lista con el nombre de todos los modelos
	NombreModelo = ["SVR", "LinearRegression", "RandomForest", "SGDRegressor"]

	#entrenamos con los distintos modelo para ver cual es el mejor
	print("Entrenando " + NombreModelo[0])
	listaMSE[0] = entrenarSVR(scaler, train_inputs, train_outputs, test_inputs, test_outputs)
	print("Entrenando " + NombreModelo[1])
	listaMSE[1] = entrenarRegresionLineal(scaler, train_inputs, train_outputs, test_inputs, test_outputs)
	print("Entrenando " + NombreModelo[2])
	listaMSE[2] = entrenarRandomForest(scaler, train_inputs, train_outputs, test_inputs, test_outputs)
	print("Entrenando " + NombreModelo[3])
	listaMSE[3] = entrenarSDGRegressor(scaler, train_inputs, train_outputs, test_inputs, test_outputs)

	menorMSE = 999999
	mejorModelo = 999999

	for iterador in range(len(listaMSE)):#comprobamos cual genero un MSE menor
		if listaMSE[iterador] < menorMSE:
			menorMSE = listaMSE[iterador]
			mejorModelo = iterador

	print("El mejor modelo es " + NombreModelo[mejorModelo] + " con un MSE de: " + str(menorMSE))

	with open(NombreModelo[iterador]+".pickle", 'rb') as fr: #cargamos el modelo ya entrenado, para guardarlo con el nombre de mejor modelo
		model = pickle.load(fr)

	#se almacena el mejor modelo
	guardar_modelo_y_scaler(model, scaler, "mejorModelo")

	return{"Modelo guardado con exito"}


@app.get("/predecir/")#get que predice la generacion a nivel diario

def read_root(base_de_datos: str, idlocalidad: str, modelo: Optional[str]="mejorModelo"):#pasamos la bd, la id de la localidad y el modelo que queramos utilizar en el caso de no querer usar el mejor

	if idlocalidad != 'exp1':
		train_file = "DatasetGeneracionDeBD.csv"
	else:
		train_file = "experimental.csv"
	#llamamos a funcion read_data para extraer los datos del fichero csv a un formato con el que podamos trabajar
	Dataset, train_inputs_tmp, train_outputs_tmp, test_inputs_tmp, test_outputs_tmp, scaler_tmp= read_data(train_file)
	if train_inputs_tmp is None or train_outputs_tmp is None or test_inputs_tmp is None or test_outputs_tmp is None:
		print("Error con el fichero")
		return {"Error con el fichero"}

	#diccionario
	PrediccionDict = {
		"Name": "Predicción",
		"Description": "",
		"Data": []
	}

	modelo_name = None
	scaler_name = None

	#dependiendo del tipo de modelo se escoge el nombre del fichero correspondiente
	if modelo == "mejorModelo":
		modelo_name = "mejorModelo.pickle"
		scaler_name = "mejorModeloscaler.pickle"
	elif modelo == "SVR":
		modelo_name = "SVR.pickle"
		scaler_name = "SVRscaler.pickle"
	elif modelo == "RegresionLineal":
		modelo_name = "RegresionLineal.pickle"
		scaler_name = "RegresionLinealscaler.pickle"
	elif modelo == "RandomForest":
		modelo_name = "RandomForest.pickle"
		scaler_name = "RandomForestscaler.pickle"
	elif modelo == "SGDRegressor":
		modelo_name = "SGDRegressor.pickle"
		scaler_name = "SGDRegressorscaler.pickle"
	else:
		return {"Error, modelo inexistente"}

	with open(modelo_name, 'rb') as fr: #cargamos el modelo ya entrenado
		model = pickle.load(fr)
	with open(scaler_name, 'rb') as fr: #cargamos el scaler del modelo para poder preprocesar el patron
		scaler = pickle.load(fr)

	#extracion de patron a nivel de dia para poder predecir la generacion del dia siguiente, ejemplo de id 14021

	#parametros para la peticion get
	getParameters_generacion = {
	"localidad": idlocalidad,
	}
	if idlocalidad != 'exp1':
		#get a la api sobre la generacion de energia
		patrones_get = requests.get("http://obtencion-datos-generacion:8008/getPatron",
			params = getParameters_generacion
		)
	else:
		patrones_get = requests.get("http://obtencion-datos-generacion:8008/getPatronExperimental")
	
	if not patrones_get.status_code == 200:#se comprueba si ha fallado
		raise Exception("Incorrect reply from Ree API. Status code: {}. Text: {}".format(patrones_get.status_code, patrones_get.text))
	#se realiza el preprocesamiento necesario para su correcto funcionamiento

	v_patrones = pd.DataFrame()

	print(v_patrones)

	for patron in patrones_get.json():
		#cada patron es guardado en un pandas que se va adjuntando a un pandas que almacena cada uno de los patrones
		temp_df = pd.DataFrame([patron], columns=['precipitaciones', 'velocidad_media_viento', 'velocidad_maxima_viento', 'direccion_media_viento', 'direccion_maxima_viento', 'humedad_relativa', 'insolacion', 'presion', 'temperatura_suelo', 'temperatura', 'temperatura_minima', 'temperatura_maxima', 'visibilidad', 'recorrido_viento', 'nieve', 'energiagenerada'])
		v_patrones = pd.concat([v_patrones,temp_df], ignore_index=True)

	#v_patrones = preprocessing(Dataset,v_patrones,mode='predict')
	#v_patrones = v_patrones.iloc[-24:, :]
	v_patrones = scaler.transform(v_patrones)[:,:-1]#preprocesado y escalado para ajustar los patrones del dia de hoy a predecir
	#se predicen resultados, se extrane los resultados que buscamos y se desescala para que tenga sentido
	prediccion = model.predict(v_patrones)#predecimos la generacion
	prediccion = np.column_stack((v_patrones,prediccion))
	prediccion = scaler.inverse_transform(prediccion)[:,-1]#realizamos el inverso al preprocesado y el escalado
	prediccion = np.round_(prediccion, decimals=2)
	Description = "Predicción diaria de generacion"
	
	#guardamos los resultados de la prediccion para cada hora del dia siguiente

	dia = datetime.now()

	ArrayOfValues = []
	for i in range(0, len(prediccion), 1):
		ArrayOfValues.append({"hour": str(i)+":00", "value": str(prediccion[i])})

	dia = dia + timedelta(days=1)
	DataDict = {
		"Day": dia.strftime("%Y-%m-%-d"),
		"units": "MWh",
		"values": ArrayOfValues
	}

	
	#cambiar formato, poniendo el formato especicado por nosotros en json
	PrediccionDict['Description'] = Description
	PrediccionDict['Data'] = DataDict
	PrediccionDict = jsonable_encoder(PrediccionDict)
	return JSONResponse(content=PrediccionDict)#devolvemos la generacio


#estas llamadas get son creadas para experimentar con un modelo en concreto
@app.get("/SVR/")#get que realiza el entrenamiento del modelo svr guardando el mejor

def read_root(base_de_datos: str):

	#creamos el fichero a entrenar con los datos que deseamos de la base de datos, aparte se actualiza la base de datos
	if 200 != crear_fichero_entrenamiento(base_de_datos):#funcion donde se actualiza la base de datos y se crea el fichero con los datos a trabajar
		return {"Error": "Error al crear el fichero con patrones"}

	train_file = "DatasetGeneracionDeBD.csv"

	#llamamos a funcion read_data para extraer los datos del fichero csv a un formato con el que podamos trabajar
	Dataset, train_inputs, train_outputs, test_inputs, test_outputs, scaler= read_data(train_file)
	if train_inputs is None or train_outputs is None or test_inputs is None or test_outputs is None:
		print("Error con el fichero")
		return {"Error con el fichero"}

	print("Entrenando SVR")#se realiza el entrenamiento de SVR
	MSE = entrenarSVR(scaler, train_inputs, train_outputs, test_inputs, test_outputs)

	return{"Modelo guardado con exito"}


@app.get("/RegresionLineal/")#get que realiza el entrenamiento del modelo RegresionLineal guardando el mejor

def read_root(base_de_datos: str):

	#creamos el fichero a entrenar con los datos que deseamos de la base de datos, aparte se actualiza la base de datos
	if 200 != crear_fichero_entrenamiento(base_de_datos):#funcion donde se actualiza la base de datos y se crea el fichero con los datos a trabajar
		return {"Error": "Error al crear el fichero con patrones"}

	train_file = "DatasetGeneracionDeBD.csv"

	#llamamos a funcion read_data para extraer los datos del fichero csv a un formato con el que podamos trabajar
	Dataset, train_inputs, train_outputs, test_inputs, test_outputs, scaler= read_data(train_file)
	if train_inputs is None or train_outputs is None or test_inputs is None or test_outputs is None:
		print("Error con el fichero")
		return {"Error con el fichero"}

	print("Entrenando RegresionLineal")#se realiza el entrenamiento de RegresionLineal
	MSE = entrenarRegresionLineal(scaler, train_inputs, train_outputs, test_inputs, test_outputs)

	return{"Modelo guardado con exito"}


@app.get("/RandomForest/")#get que realiza el entrenamiento del modelo random forest guardando el mejor

def read_root(base_de_datos: str):

	#creamos el fichero a entrenar con los datos que deseamos de la base de datos, aparte se actualiza la base de datos
	if 200 != crear_fichero_entrenamiento(base_de_datos):#funcion donde se actualiza la base de datos y se crea el fichero con los datos a trabajar
		return {"Error": "Error al crear el fichero con patrones"}

	train_file = "DatasetGeneracionDeBD.csv"

	#llamamos a funcion read_data para extraer los datos del fichero csv a un formato con el que podamos trabajar
	Dataset, train_inputs, train_outputs, test_inputs, test_outputs, scaler= read_data(train_file)
	if train_inputs is None or train_outputs is None or test_inputs is None or test_outputs is None:
		print("Error con el fichero")
		return {"Error con el fichero"}

	print("Entrenando RandomForest")#se realiza el entrenamiento de random forest
	MSE = entrenarRandomForest(scaler, train_inputs, train_outputs, test_inputs, test_outputs)

	return{"Modelo guardado con exito"}


@app.get("/SGDRegressor/")#get que realiza el entrenamiento del modelo sgdregressor guardando el mejor

def read_root(base_de_datos: str):

	#creamos el fichero a entrenar con los datos que deseamos de la base de datos, aparte se actualiza la base de datos
	if 200 != crear_fichero_entrenamiento(base_de_datos):#funcion donde se actualiza la base de datos y se crea el fichero con los datos a trabajar
		return {"Error": "Error al crear el fichero con patrones"}

	train_file = "DatasetGeneracionDeBD.csv"

	#llamamos a funcion read_data para extraer los datos del fichero csv a un formato con el que podamos trabajar
	Dataset, train_inputs, train_outputs, test_inputs, test_outputs, scaler= read_data(train_file)
	if train_inputs is None or train_outputs is None or test_inputs is None or test_outputs is None:
		print("Error con el fichero")
		return {"Error con el fichero"}

	print("Entrenando SGDRegressor")#se realiza el entrenamiento de SGDRegressor
	MSE = entrenarSDGRegressor(scaler, train_inputs, train_outputs, test_inputs, test_outputs)

	return{"Modelo guardado con exito"}

#**********************************************************************************************************************************************
#*************************************************************FIN_PETICIONES_GET***************************************************************
#**********************************************************************************************************************************************



#**********************************************************************************************************************************************
#*************************************************************FUNCIONES_DE_APOYO***************************************************************
#**********************************************************************************************************************************************

#funcion que realiza el entrenamiento de SVR
def entrenarSVR(scaler, train_inputs, train_outputs, test_inputs, test_outputs):

	f=open("ResultadosBusquedaParametrosSVR", "w"); #archivo donde se almacenan los resultados de los parametros

	#inicializamos parametros que usamos en el entrenamiento
	vector = np.array([1e-2,1e-1,1e0,1e1,1e2,1e3])
	mejor_mse = 999999
	mae = 999999
	mejor_C = 999999
	mejor_Gamma = 999999

	for C in vector:#entrenamos probando distintas c y gammas
		for Gamma in vector:
			#print("C=%f y Gamma=%f" % (C, Gamma))

			# Entrenamos el modelo
			model = svm.SVR(kernel='rbf',C=C, gamma=Gamma)
			model.fit(train_inputs, train_outputs)
			y_pred=model.predict(test_inputs)
			test_mse = mean_squared_error(test_outputs, y_pred) #MSE
			test_mae = mean_absolute_error(test_outputs, y_pred) #MAE

			f.write("MSE & MAE Final con C=%f y G=%f: \t%f %f\n" % (C, Gamma, test_mse, test_mae))#lo vamos almacenando en un fichero

			if mejor_mse > test_mse:#se va guardando la mejor combinacion de parametros
				mejor_mse = test_mse
				mejor_C = C
				mejor_Gamma = Gamma
				mae = test_mae

				#se almacena el modelo y el scaler para poder eliminar el preprocesamiento cuando sea necesario o aplicarlo
				guardar_modelo_y_scaler(model,scaler,"SVR")


	f.write("Los mejores parametros son C:"+ str(mejor_C) +" y Gamma:"+ str(mejor_Gamma) +" con un MSE de "+ str(mejor_mse)+" y un MAE de "+str(mae))
	f.close()

	print("******************")
	print("Los mejores parametros son C:"+ str(mejor_C) +" y Gamma:"+ str(mejor_Gamma) +" con un MSE de "+ str(mejor_mse)+" y un MAE de "+str(mae))
	print("******************")

	return mejor_mse


#funcion que realiza el entrenamiento de RegresionLineal
def entrenarRegresionLineal(scaler, train_inputs, train_outputs, test_inputs, test_outputs):

	f=open("ResultadosBusquedaParametrosRegresionLineal", "w"); #archivo donde se almacenan los resultados de los parametros

	#inicializamos parametros que usamos en el entrenamiento
	mejor_mse = 999999
	mae = 999999

	#Entrenamos el modelo
	model = linear_model.LinearRegression()
	model.fit(train_inputs, train_outputs)
	y_pred=model.predict(test_inputs)
	test_mse = mean_squared_error(test_outputs, y_pred) #MSE
	test_mae = mean_absolute_error(test_outputs, y_pred) #MAE
	
	mejor_mse = test_mse
	mae = test_mae

	f.write("MSE & MAE Final con para la regresion lineal: \t%f %f\n" % (test_mse, test_mae))
	f.close()

	#se almacena el modelo y el scaler para poder eliminar el preprocesamiento cuando sea necesario o aplicarlo
	guardar_modelo_y_scaler(model,scaler,"RegresionLineal")
	
	print("******************")
	print("La regresion lineal no tiene parametros que buscar, siendo su MSE: "+ str(mejor_mse)+" y un MAE de "+str(mae))
	print("******************")

	return mejor_mse


#funcion que realiza el entrenamiento de RandomForest
def entrenarRandomForest(scaler, train_inputs, train_outputs, test_inputs, test_outputs):

	f=open("ResultadosBusquedaParametrosRandomForest", "w"); #archivo donde se almacenan los resultados de los parametros

	#inicializamos parametros que usamos en el entrenamiento
	v_n_estimators = {100,300}
	v_max_features = {0.01, 0.05, 0.1, 0.2, 0.3}
	v_min_samples_split = {2, 50, 100}
	v_min_samples_leaf = {2, 50, 100}
	mejor_mse = 99999
	mae = 999999
	mejor_n_estimators = 999999
	mejor_min_samples_split = 999999
	mejor_min_samples_leaf = 999999
	mejor_max_features = 999999

	#entrenamos probando distintas combinaciones
	for n_estimators in v_n_estimators:
		for max_features in v_max_features:
			for min_samples_split in v_min_samples_split:
				for min_samples_leaf in v_min_samples_leaf:
					#print("n_estimators=%d, max_features=%f, min_samples_split=%f, min_samples_leaf=%f \n" % (n_estimators, max_features, min_samples_split, min_samples_leaf))

					#Entrenamos el modelo
					model = ensemble.RandomForestRegressor(min_weight_fraction_leaf=0.2, max_samples= 0.5,n_estimators=n_estimators, max_features=max_features, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, bootstrap=True, n_jobs=4)
					model.fit(train_inputs, train_outputs)
					y_pred=model.predict(test_inputs)
					test_mse = mean_squared_error(test_outputs, y_pred) #MSE
					test_mae = mean_absolute_error(test_outputs, y_pred) #MAE

					f.write("MSE & MAE Final con n_estimators=%d, v_max_features=%f, min_samples_split=%f, min_samples_leaf=%f= \t%f %f\n" % (n_estimators, max_features, min_samples_split, min_samples_leaf, test_mse, test_mae))

					if mejor_mse > test_mse:#se va guardando la mejor combinacion de parametros
						mejor_mse = test_mse
						mae = test_mae
						mejor_n_estimators = n_estimators
						mejor_min_samples_split = min_samples_split
						mejor_min_samples_leaf = min_samples_leaf
						mejor_max_features = max_features

						#se almacena el modelo y el scaler para poder eliminar el preprocesamiento cuando sea necesario o aplicarlo
						guardar_modelo_y_scaler(model,scaler,"RandomForest")

	f.write("Los mejores parametros para el random forest es n_estimators->"+ str(n_estimators) + "max_features->"+ str(mejor_max_features) + ", min_samples_split->"+ str(mejor_min_samples_split) +", min_samples_leaf->"+ str(mejor_min_samples_leaf) + " con un MSE de:"+ str(mejor_mse)+" y un MAE de "+str(mae))
	f.close()

	print("******************")
	print("Los mejores parametros para el random forest es n_estimators->"+ str(n_estimators) + "max_features->"+ str(mejor_max_features) + ", min_samples_split->"+ str(mejor_min_samples_split) +", min_samples_leaf->"+ str(mejor_min_samples_leaf) + " con un MSE de:"+ str(mejor_mse)+" y un MAE de "+str(mae))
	print("******************")

	return mejor_mse


#funcion que realiza el entrenamiento de SGDRegressor
def entrenarSDGRegressor(scaler, train_inputs, train_outputs, test_inputs, test_outputs):

	f=open("ResultadosBusquedaParametrosSGCRegressor", "w"); #archivo donde se almacenan los resultados de los parametros

	#inicializamos parametros que usamos en el entrenamiento
	v_validation = {0.1, 0.2, 0.3, 0.4}
	v_early_stopping = {True, False}
	v_shuffle = {True, False}
	#v_l1_ratio = {0, 0.1, 0.2, 0.3 ,0.4 ,0.5, 0.6 ,0.7 ,0.8 ,0.9 ,1}
	mejor_mse = 99999
	mae = 999999
	mejor_validation = 999999
	mejor_early_stopping = None
	mejor_shuffle = None
	#mejor_l1_ratio = 999999
	mejor_alpha = 999999

	#entrenamos probando distintas combinaciones
	for early_stopping in v_early_stopping:
		for validation in v_validation:
			for shuffle in v_shuffle:
				#for l1_ratio in v_l1_ratio:
					for alpha in range(1,21,1):
						#print("early_stopping=%r, validation=%f, shuffle=%r, alpha=%f \n" % (early_stopping, validation, shuffle, alpha/10000))

						#Entrenamos el modelo
						model = linear_model.SGDRegressor(early_stopping=early_stopping, validation_fraction=validation, shuffle=shuffle,alpha=alpha/10000)
						model.fit(train_inputs, train_outputs)
						y_pred=model.predict(test_inputs)
						test_mse = mean_squared_error(test_outputs, y_pred)
						test_mae = mean_absolute_error(test_outputs, y_pred) #MAE
						
						f.write("MSE & MAE Final con early_stopping=%r, validation=%f, shuffle=%r, alpha=%f= \t%f %f\n \n" % (early_stopping, validation, shuffle, alpha/10000, test_mse, test_mae))

						if mejor_mse > test_mse:#se va guardando la mejor combinacion de parametros
							mejor_mse = test_mse
							mae = test_mae
							mejor_early_stopping = early_stopping
							mejor_validation = validation
							mejor_shuffle = shuffle
							mejor_alpha = alpha/10000

							#se almacena el modelo y el scaler para poder eliminar el preprocesamiento cuando sea necesario o aplicarlo
							guardar_modelo_y_scaler(model,scaler,"SGDRegressor")
							

	f.write("Los mejores parametros para el SGDRegressor es early_stopping->"+str(mejor_early_stopping)+ ", validation->"+ str(mejor_validation)+ ", shuffle->"+ str(mejor_shuffle)+ ", alpha->"+ str(mejor_alpha)+" con un MSE de:"+ str(mejor_mse)+" y un MAE de "+str(mae))
	f.close()

	print("******************")
	print("Los mejores parametros para el SGDRegressor es early_stopping->"+str(mejor_early_stopping)+ ", validation->"+ str(mejor_validation)+ ", shuffle->"+ str(mejor_shuffle)+ ", alpha->"+ str(mejor_alpha)+" con un MSE de:"+ str(mejor_mse)+" y un MAE de "+str(mae))
	print("******************")

	return mejor_mse



#funcion para leer los datos del fichero
def read_data(train_file):

		#guardamos los datos en un dataset de tipo string
		Dataset = pd.read_csv(train_file)

		#guardamos una copia de los datos sin procesar
		Dataset_sin_procesar = Dataset.copy()

		Dataset = preprocessing(Dataset, mode='train')


		#dividimos en 70% de datos para entreno y el resto para test
		train, test = train_test_split(Dataset, test_size= 0.10)

		#realizamos un preprocesamiento para estandarizar los datos
		scaler = MinMaxScaler().fit(train)
		train = scaler.transform(train)
		test = scaler.transform(test)

		#finalmente se dividen los datos en inputs y outputs
		train_inputs=train[:, :-1]
		train_outputs=train[:, -1:]
		test_inputs=test[:, :-1]
		test_outputs=test[:, -1:]

		#transforma a una unica columna para evitar warnings (solo en el caso de una salida)
		train_outputs = train_outputs.ravel() 
		test_outputs = test_outputs.ravel()

		return Dataset_sin_procesar, train_inputs, train_outputs, test_inputs, test_outputs, scaler



#preprocesamiento de los datos donde se eliminan patrones que no nos sirven, en el caso de predecir unimos el dataset con los valores outputs pasados
def preprocessing(Dataset, outputs=None, mode='train'):

	if  mode == 'predict':
		Dataset = pd.concat((Dataset, outputs))

	Dataset = Dataset.fillna(0)
	
	#no para este Dataset
	"""ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), list(range(5)))], remainder= 'passthrough')#realización del encoder 
	Dataset = ct.fit_transform(Dataset)"""
		
	return Dataset


#funcion para leer de la base de datos
def crear_fichero_entrenamiento(base_de_datos):

	try:
		#primero se intenta actualizar la base de datos
		actualizar = requests.get("http://obtencion-datos-generacion:8008/actualizar")
		#se comprueba si ha fallado
		if not actualizar.status_code == 200:
			raise Exception("Incorrect reply from Ree API. Status code: {}. Text: {}".format(actualizar.status_code, actualizar.text))
			return {"Error al actualizar"}

		# se crea la conexion a la base de datos para extraer los datos a un fichero
		conn = psycopg2.connect(database=base_de_datos, user="postgres", password="password", host="timescaledb", port="5432")
		cur = conn.cursor()#se crea cursor
 
		#se realiza la peticion de sql con los atributos que queremos usar para extraer los datos de la bd y exportarlos a fichero csv
		sql = "COPY (SELECT precipitaciones, velocidad_media_viento, velocidad_maxima_viento, direccion_media_viento, direccion_maxima_viento, humedad_relativa, insolacion, presion, temperatura_suelo, temperatura, temperatura_minima, temperatura_maxima, visibilidad, recorrido_viento, nieve, energiagenerada FROM datos_generacion) TO STDOUT WITH CSV DELIMITER ','" 

		#guardamos los resultados de la peticion en un fichero .csv
		with open("/usr/src/app/DatasetGeneracionDeBD.csv", "w") as file: 
			#en la primera linea guardamos el nombre de las columnas
			file.write("precipitaciones,velocidad_media_viento,velocidad_maxima_viento,direccion_media_viento,direccion_maxima_viento,humedad_relativa,insolacion,presion,temperatura_suelo,temperatura,temperatura_minima,temperatura_maxima,visibilidad,recorrido_viento,nieve,energiagenerada\n")
			cur.copy_expert(sql, file)

		conn.close()#cerramos conexion con la base de datos

		return 200
	except:
		return 300


#funcion para guardar modelo
def guardar_modelo_y_scaler(model, scaler, name):

	#guardamos el modelo y el scaler
	with open(name+'.pickle', 'wb') as fw:
		pickle.dump(model, fw)
	with open(name+'scaler.pickle', 'wb') as fw:
		pickle.dump(scaler, fw)


#**********************************************************************************************************************************************
#*********************************************************FIN_FUNCIONES_DE_APOYO***************************************************************
#**********************************************************************************************************************************************