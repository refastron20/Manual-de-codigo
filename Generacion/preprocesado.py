from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def main():
	#opciones de pandas para imprimir
	pd.set_option('display.max_rows', 0)
	pd.set_option('display.max_columns', None)
	pd.set_option('display.date_dayfirst', True)
	pd.set_option('display.expand_frame_repr', True)
	pd.set_option('display.large_repr', 'truncate')
	
	#pd.set_option('display.max_colwidth', None)
	#leemos el dataset
	data = pd.read_csv('./Datasets/Weather_and_energy_Final_2020_2021.csv')
	col = data.pop("Energy Discharged (Wh)")
	data.insert(len(data.columns), col.name, col)
	#imprimimos una pequeña descripción del dataset
	print(data.describe())

	#imprimimos histogramas para ver las distribuciones generales de las variables
	data.hist(figsize=(20,20))
	plt.show()

	#y un histograma de correlaciones para ver la correlación entre variables
	fig, ax = plt.subplots(figsize=(15,20))
	ax.autoscale(enable=None, axis="x", tight=True)
	#fig.tight_layout()
	sns.heatmap(data.corr(), cmap='RdBu_r', annot=True)
	plt.setp( ax.xaxis.get_majorticklabels(), rotation=45, ha="right" )
	plt.subplots_adjust(left=0.2, bottom=0.2, right=1, top=0.9, wspace=0, hspace=1)
	plt.show()

	



if __name__ == "__main__":
	main()
