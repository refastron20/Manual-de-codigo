import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
from cmath import nan
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import sys

@click.command()
@click.option('--train_file', '-d', default=None, required=True,
			help=u'Name of the file with training data.')
@click.option('--outputs', '-o', default=1, required=False, 
			show_default=True, help=u'Number of columns that will be used as target variables (all at the end).')

def pruebas(train_file, outputs):
	Dataset = read_data(train_file, outputs)

	if Dataset is None:
		print("Error con el fichero")
		return
	with pd.option_context('display.max_rows', None, 'display.max_columns', None):
		with open("randomfile.txt", "w") as external_file:
			print(Dataset, file=external_file)
			external_file.close()

	Dataset.hist(figsize=(25,20))
	plt.show()

	
	x = Dataset.values #returns a numpy array
	DatasetNormalizado=x[:, :-outputs]
	Estandarizado=x[:, :-outputs]
	auxiliarEstandar = pd.DataFrame(Estandarizado)

	min_max_scaler = MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(DatasetNormalizado)
	DatasetNormalizado = pd.DataFrame(x_scaled)
	DatasetNormalizado.hist()
	plt.show()

	Estandarizado_scaled= StandardScaler() 
	EstandarizadoFinal= Estandarizado_scaled.fit_transform(Estandarizado)
	names=auxiliarEstandar.columns
	EstandarizadoFinal =pd.DataFrame(EstandarizadoFinal, columns=names)
	EstandarizadoFinal.hist()
	plt.show()

	"""
	pd.plotting.scatter_matrix(Dataset)
	plt.show()

	pd.plotting.scatter_matrix(DatasetNormalizado)
	plt.show()

	pd.plotting.scatter_matrix(EstandarizadoFinal)
	plt.show()"""

	corr = Dataset.corr()
	sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns)
	plt.show()

#funcion para leer datos del fichero
def read_data(train_file, outputs):

	Dataset = None

	#guardamos los datos en un dataset de tipo string
	np_array = np.genfromtxt(train_file, delimiter=',', usecols=range(1,19), dtype='str')

	#cambiamos lass variables 't' o 'f' por '1' o '0' respectivamente
	i = 0
	for x in np_array[:,5]:
		if x == 't':
			x = '1'
		else:
			x = '0'
		np_array[i,5] = x
		i+=1

	#las filas con valores nan se vacian
	#np_array = np.delete(np_array, np.where(np_array == '')[0], axis=0)

	#pasamos el tipo a float
	np_array = np_array.astype(float)
	
	np_array = preprocessing(np_array)
	#np.set_printoptions(threshold=sys.maxsize)
	#print (np_array)
	#realizamos un preprocesamiento para estandarizar los datos
	#scaler = MinMaxScaler().fit(np_array)
	#np_array = scaler.transform(np_array)

	#Dataset = pd.DataFrame(np_array,columns= ["anio2021", "anio2022","mes1","mes2","mes3","mes4","mes5","mes6","mes7","mes8","mes9","mes10","mes11","mes12","dia1"      ,"dia2"      ,"dia3"      ,"dia4"      ,"dia5"      ,"dia6"      ,"dia7"      ,"dia8"      ,"dia9"      ,"dia10"     ,"dia11"     ,"dia12"     ,"dia13"     ,"dia14"     ,"dia15"     ,"dia16"     ,"dia17"     ,"dia18"     ,"dia19"     ,"dia20"     ,"dia21"     ,"dia22"     ,"dia23"     ,"dia24"     ,"dia25"     ,"dia26"     ,"dia27"     ,"dia28"     ,"dia29"     ,"dia30"     ,"dia31"     ,"hora0"     ,"hora1"     ,"hora2"     ,"hora3"     ,"hora4"     ,"hora5"     ,"hora6"     ,"hora7"     ,"hora8"     ,"hora9"     ,"hora10"    ,"hora11"    ,"hora12"    ,"hora13"    ,"hora14"    ,"hora15"    ,"hora16"    ,"hora17"    ,"hora18"    ,"hora19","hora20","hora21","hora22","hora23","num_dia0","num_dia1","num_dia2","num_dia3","num_dia4","num_dia5","num_dia6","festivo","energia_hidraulica", "energia_eolica", "energia_fv", "energia_nuclear", "energia_carbon", "energia_fuelgas", "energia_ciclocombinado", "demanda","precio_dos_horas_antes", "precio_hora_antes", "precio_actual","precio_dia_siguiente"])
	Dataset = pd.DataFrame(np_array,columns= ["anio","mes","dia","hora","num_dia","festivo","energia_hidraulica", "energia_eolica", "energia_fv", "energia_nuclear", "energia_carbon", "energia_fuelgas", "energia_ciclocombinado", "demanda","precio_dos_horas_antes", "precio_hora_antes", "precio_actual","precio_dia_siguiente"])

	return Dataset



def preprocessing(Dataset, outputs=None):

	col_mean = np.nanmean(Dataset, axis=0)

	nan_cols = np.where(np.isnan(Dataset))

	Dataset[nan_cols] = np.take(col_mean, nan_cols[1])

	Dataset = np.delete(Dataset, np.where(Dataset == nan)[0], axis=1)

	Dataset = Dataset[:, ~np.isnan(Dataset).any(axis=0)]

	#ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), list(range(5)))], remainder= 'passthrough')
	#Dataset = np.array(ct.fit_transform(Dataset))

	return Dataset


if __name__ == "__main__":
	pruebas()
