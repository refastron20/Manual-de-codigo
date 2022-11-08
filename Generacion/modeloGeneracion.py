from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA



def main():

	Dataset, train_inputs, train_outputs, test_inputs, test_outputs, scaler= read_data('./Datasets/Weather_and_energy_Final_2020_2021.csv')

	




	

#funcion para leer datos del fichero
#funcion para leer los datos del fichero
def read_data(train_file):

		Inputs = None
		Outputs = None

		#guardamos los datos en un dataset de tipo string
		Dataset = pd.read_csv(train_file)

		#cambiamos la columna de bool a int
		col = Dataset.pop("Energy Discharged (Wh)")
		Dataset.insert(len(Dataset.columns), col.name, col)


		Dataset_sin_procesar = Dataset.copy()

		Dataset = preprocessing(Dataset, mode='train')

		"""pd.set_option('display.max_rows', 0)
		pd.set_option('display.max_columns', None)
		pd.set_option('display.date_dayfirst', True)
		pd.set_option('display.expand_frame_repr', True)
		pd.set_option('display.large_repr', 'truncate')
		
		print(Dataset.head())"""


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





def preprocessing(Dataset, outputs=None, mode='train'):#preprocesamiento de los datos donde se eliminan patrones que no nos sirver y se realiza codificacion de parametros categoricos

	if  mode == 'predict':
		Dataset = pd.concat((Dataset, outputs))



	#eliminamos columnas descriptivas
	Dataset.pop("Name")
	Dataset.pop("Date time")
	Dataset.pop("Conditions")
	#eliminamos columnas que no aportan info
	Dataset.pop("Heat Index")
	#hay que ver
	Dataset.pop("Minimum Temperature")
	Dataset.pop("Snow")
	Dataset.pop("Snow Depth")

	Dataset.pop("Wind Gust")

	Dataset = Dataset.fillna(0)
	
	#no es necesario por ahora
	"""ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), list(range(5)))], remainder= 'passthrough')#realización del encoder 
	Dataset = ct.fit_transform(Dataset)"""
		
	return Dataset


if __name__ == "__main__":
	main()