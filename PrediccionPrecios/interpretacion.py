# Import tools needed for visualization
from cmath import nan
import pickle
from matplotlib import pyplot as plt
from sklearn import linear_model, svm, tree
from sklearn.datasets import fetch_olivetti_faces
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import export_graphviz
import numpy as np
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
import psycopg2
from statsmodels.tsa.stattools import adfuller
from numpy import log

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm

def main():

	crear_fichero_test('modelosbd')

	#llamamos a funcion read_data para extraer los datos del fichero csv a un formato con el que podamos trabajar
	test, train_inputs_tmp, train_outputs_tmp, test_inputs_tmp, test_outputs_tmp, scaler_tmp= read_data("test.csv", 1)
	if train_inputs_tmp is None or train_outputs_tmp is None or test_inputs_tmp is None or test_outputs_tmp is None:
		print("Error con el fichero")
		return {"Error con el fichero"}
	
	precios_reales = test[:,-1]
	
	train_file = "bd.csv"
	#llamamos a funcion read_data para extraer los datos del fichero csv a un formato con el que podamos trabajar
	Dataset, train_inputs_tmp, train_outputs_tmp, test_inputs_tmp, test_outputs_tmp, scaler_tmp= read_data(train_file, 1)
	if train_inputs_tmp is None or train_outputs_tmp is None or test_inputs_tmp is None or test_outputs_tmp is None:
		print("Error con el fichero")
		return {"Error con el fichero"}

	#print(v_patrones)

	modelos = ["ARIMA"]

	for modelo in modelos:
		print(modelo)
		if modelo == "SVR":
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
		elif modelo == "ARIMA":
			modelo_name = "ARIMA.pickle"
			scaler_name = "ARIMAscaler.pickle"

		with open(modelo_name, 'rb') as fr: #cargamos el modelo ya entrenado
			model = pickle.load(fr)
		if modelo != "ARIMA":
			with open(scaler_name, 'rb') as fr: #cargamos el scaler del modelo para poder preprocesar el patron
				scaler = pickle.load(fr)

		if modelo != "ARIMA":
			v_patrones = preprocessing(Dataset,test,mode='predict')[-24:, :]
			v_patrones = scaler.transform(v_patrones)[:,:-1]#preprocesado y escalado para ajustar los patrones del dia de hoy a predecir
			#se predicen resultados, se extrane los resultados que buscamos y se desescala para que tenga sentido
			prediccion = model.predict(v_patrones)#predecimos los precios
			prediccion = np.column_stack((v_patrones,prediccion))
			prediccion = scaler.inverse_transform(prediccion)[:,-1]#realizamos el inverso al preprocesado y el escalado
			
		else:
			prediccion = model.predict(n_periods=24)#predecimos los precios

		prediccion = np.round_(prediccion, decimals=2)
			




		fig, ax = plt.subplots(figsize=(20, 10))
		ax.autoscale(enable=None, axis="y", tight=True)
		#ax.set_ylim()
		#ax.set_xlim(left=8000, right=9000)
		ax.plot( prediccion, 'r-', label='predicho')
		ax.plot( precios_reales, 'b-', label='real')
		plt.ylabel('precios')
		plt.xlabel('fecha')
		plt.xticks(rotation = '90'); 
		plt.legend()
		plt.show()



	
	
	


#funcion para leer los datos del fichero
def read_data(train_file, outputs):
		Inputs = None
		Outputs = None

		#guardamos los datos en un dataset de tipo string
		Dataset = np.genfromtxt(train_file, delimiter=',', usecols=range(1,19), dtype='str')

		#cambiamos lass variables 't' o 'f' por '1' o '0' respectivamente
		i = 0
		for x in Dataset[:,5]:
			if x == 't':
				x = '1'
			else:
				x = '0'
			Dataset[i,5] = x
			i+=1

		#pasamos el tipo a float
		Dataset = Dataset.astype(float)


		Dataset_sin_procesar = np.copy(Dataset)
		Dataset = preprocessing(Dataset, mode='train')

		#dividimos en 70% de datos para entreno y el resto para test
		train, test = train_test_split(Dataset, test_size= 0.10, random_state=0)

		#realizamos un preprocesamiento para estandarizar los datos
		scaler = MinMaxScaler().fit(train)
		train = scaler.transform(train)
		test = scaler.transform(test)

		#finalmente se dividen los datos en inputs y outputs
		train_inputs=train[:, :-outputs]
		train_outputs=train[:, -outputs:]
		test_inputs=test[:, :-outputs]
		test_outputs=test[:, -outputs:]


		return Dataset_sin_procesar, train_inputs, train_outputs, test_inputs, test_outputs, scaler



def preprocessing(Dataset, outputs=None, mode='train'):
	if  mode == 'predict':
		Dataset = np.concatenate((Dataset, outputs))




	nan_cols = np.where(np.isnan(Dataset))
	
	Dataset[nan_cols] = 0

	Dataset = np.delete(Dataset, np.where(Dataset == nan)[0], axis=1)

	Dataset = Dataset[:, ~np.isnan(Dataset).any(axis=0)]

	ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), list(range(4)))], remainder= 'passthrough')
	Dataset = np.array(ct.fit_transform(Dataset))
		
	return Dataset



def crear_fichero_test(base_de_datos):
	#conexion a la base de datos
	try: 
		conn = psycopg2.connect(database=base_de_datos, user="postgres", password="password", host="127.0.0.1", port="5432")
		cur = conn.cursor()#se crea cursor
		#se realiza la peticion de sql para extraer los datos de la bd y exportarlos a fichero csv
		sql = "COPY (SELECT * FROM precio ORDER BY fecha DESC limit 24) TO STDOUT WITH CSV DELIMITER ','" 
		with open("./test.csv", "w") as file: 
			cur.copy_expert(sql, file)
		conn.close()#cerramos conexion con la base de datos
	except:
		print('no bd')


if __name__ == "__main__":
	main()