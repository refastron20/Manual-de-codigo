#import necesarios para el funcionamiento de la aplicación
from xml.etree.ElementTree import tostring
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd

import psycopg2
import requests
from random import seed
from random import random
from datetime import datetime

app = FastAPI()



#**********************************************************************************************************************************************
#*****************************************************************PETICIONES_GET***************************************************************
#**********************************************************************************************************************************************

@app.get("/")#cuando se realiza un get en la api no se realiza ninguna accion
def read_root():

	return {"No se realiza ninguna acción"}

@app.get("/experimental/")#get para obtener datos experimentales
def read_root():

	#guardamos los datos en un dataset de tipo string
	Dataset = pd.read_csv("Weather_and_energy_Final_2020_2021.csv")

	#cambiamos la columna de energía generada al final
	col = Dataset.pop("Energy Discharged (Wh)")
	Dataset.insert(len(Dataset.columns), col.name, col)

	#eliminamos columnas descriptivas
	Dataset.pop("Name")
	Dataset.pop("Conditions")
	#eliminamos columnas que no aportan info
	Dataset.pop("Heat Index")
	Dataset.pop("Snow Depth")

	Dataset['Date time'] = pd.to_datetime(Dataset['Date time'])
	Dataset['Date time'] = Dataset['Date time'].dt.strftime('%Y-%m-%d')


	#rellenamos los nan con 0s
	Dataset = Dataset.fillna(0)



	json_final = {
		'fecha': '',
		'ubicacion':'',
		'idema':'',
		'precipitaciones':'',
		'velocidad_media_viento':'',
		'velocidad_maxima_viento':'',
		'direccion_media_viento':'',
		'direccion_maxima_viento':'',
		'humedad_relativa':'',
		'insolacion':'',
		'presion':'',
		'temperatura_suelo':'',
		'temperatura':'',
		'temperatura_minima':'',
		'temperatura_maxima':'',
		'visibilidad':'',
		'recorrido_viento':'',
		'nieve':'',
		'energia_generada' : ''
	}

	lista_final = []

	for i in range(0,len(Dataset),1):
		json_final['fecha'] = Dataset['Date time'][i]
		json_final['ubicacion'] = 'experimental'
		json_final['idema'] = 'exp1'
		json_final['precipitaciones'] = Dataset['Precipitation'][i]
		json_final['velocidad_media_viento'] = Dataset['Wind Speed'][i]
		json_final['velocidad_maxima_viento'] = 0
		json_final['direccion_media_viento'] = Dataset['Wind Direction'][i]
		json_final['direccion_maxima_viento'] = 0
		json_final['humedad_relativa'] = Dataset['Relative Humidity'][i]
		json_final['insolacion'] = 0
		json_final['presion'] = 0
		json_final['temperatura_suelo'] = Dataset['Temperature'][i]
		json_final['temperatura'] = Dataset['Temperature'][i]
		json_final['temperatura_minima'] = Dataset['Maximum Temperature'][i]
		json_final['temperatura_maxima'] = Dataset['Minimum Temperature'][i]
		json_final['visibilidad'] = Dataset['Visibility'][i]
		json_final['recorrido_viento'] = Dataset['Wind Chill'][i]
		json_final['nieve'] = Dataset['Snow'][i]
		json_final['energia_generada'] = int(Dataset['Energy Discharged (Wh)'][i])
		lista_final.append(json_final.copy())

	
	
	return JSONResponse(content=lista_final)

@app.get("/actualizar/")#get que comprueba si la bd esta actualizada y en el caso de que no inserta nuevos patrones

def read_root():

	#conexion a la base de datos
	conn = psycopg2.connect(database="modelosbd", user="postgres", password="password", host="timescaledb", port="5432")
	cur = conn.cursor()#se crea cursor

	#Actualizacion de la base de datos
	cur.execute("Select fecha from datos_generacion order by fecha desc limit 1")
	v_ultima_fecha_bd = None
	v_ultima_fecha_bd = cur.fetchall()#sacamos la ultima fecha de la base de datos

	#esta lista se deberá ir ampliando con los ids de los municipios donde haya placas solares
	lista_ids = ['5402', '6084X']
	lista_json = []

	for i in lista_ids:

		tiempo = pedir_tiempo(i)#llamamos a la funcion que extrae los datos, respecto a tiempo
		leng = len(tiempo)
		placas = pedir_generacion(i, leng)#llamamos a la funcion que extrae los datos, respecto a la generacion

		json_final = fusionar_datos(tiempo, placas)#se fusionan los datos anteriores
		for i in range(0,leng,1):
			#print(json_final[i].values())
			#print(v_ultima_fecha_bd[0][0])
			#print(json_final[i]['fecha'])
			fechajson = datetime.strptime(json_final[i]['fecha'], '%Y-%m-%dT%H:%M:%S')
			#print(fechajson)
			if v_ultima_fecha_bd[0][0] < fechajson:
				insert_Pattern(conn, list(json_final[i].values()))#se inserta el patron
		lista_json.append(json_final.copy())#cada patron se va insertando en un json para su posible visualizacion

	conn.close()#cerramos conexion con la base de datos

	return JSONResponse(content=lista_json)


#68.1               ,53.6               ,62.8       ,         0,         0.05,   0,      12.8,        178.25,       30,       9.6,       44.6,            76.68
#Maximum Temperature,Minimum Temperature,Temperature,Wind Chill,Precipitation,Snow,Wind Speed,Wind Direction,Wind Gust,Visibility,Cloud Cover,Relative Humidity
@app.get("/getPatronExperimental/")#get para coger patron de generacion

def read_root():

	json_final = {
		'precipitaciones':'',
		'velocidad_media_viento':'',
		'velocidad_maxima_viento':'',
		'direccion_media_viento':'',
		'direccion_maxima_viento':'',
		'humedad_relativa':'',
		'insolacion':'',
		'presion':'',
		'temperatura_suelo':'',
		'temperatura':'',
		'temperatura_minima':'',
		'temperatura_maxima':'',
		'visibilidad':'',
		'recorrido_viento':'',
		'nieve':'',
		'energia_generada' : ''
	}
	json_final['precipitaciones'] = 0.05
	json_final['velocidad_media_viento'] = 12.8
	json_final['velocidad_maxima_viento'] = 0
	json_final['direccion_media_viento'] = 178.25
	json_final['direccion_maxima_viento'] = 0
	json_final['humedad_relativa'] = 76.68
	json_final['insolacion'] = 0
	json_final['presion'] = 0
	json_final['temperatura_suelo'] = 62.8
	json_final['temperatura'] = 62.8
	json_final['temperatura_minima'] = 68.1
	json_final['temperatura_maxima'] = 53.6 
	json_final['visibilidad'] = 9.6
	json_final['recorrido_viento'] = 0
	json_final['nieve'] = 0
	json_final['energia_generada'] = 0

	lista_final = [json_final]
	#res = 16190
	return JSONResponse(content=lista_final)


@app.get("/getPatron/")#get para coger patron de generacion

def read_root(localidad: int):#ejecuta el codigo para el id pasado en el get

	municipio = localidad

	#se realiza el get pasando la clave url y el header
	url = "https://opendata.aemet.es/opendata/api/prediccion/especifica/municipio/horaria/"+str(municipio)

	querystring = {
		"api_key":"eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJyaG9ybWlnbzk5QGdtYWlsLmNvbSIsImp0aSI6IjVjZTNkMjMxLTFlZWYtNDIzMC1hMGIyLTNiM2Q4ZWNiMWVjMSIsImlzcyI6IkFFTUVUIiwiaWF0IjoxNjU4OTk5NTAzLCJ1c2VySWQiOiI1Y2UzZDIzMS0xZWVmLTQyMzAtYTBiMi0zYjNkOGVjYjFlYzEiLCJyb2xlIjoiIn0.FL0Erp4mvRqo6uo_4Rp21bUaTod26dl08fMoDiPUHfM"}

	headers = {
		'cache-control': "no-cache"
		}
	

	resp_aemet = requests.request("GET", url, headers=headers, params=querystring)

	#se comprueba si ha fallado
	if not resp_aemet.status_code == 200:
		raise Exception("Incorrect reply from Aemet API. Status code: {}. Text: {}".format(resp_aemet.status_code, resp_aemet.text))

	url = resp_aemet.json()['datos']#si ha funcionado correctamente, extraemos los datos de la peticion
	resp_aemet = (requests.request("GET", url, headers=headers)).json()[0]

	num_patrones = len(resp_aemet['prediccion']['dia'][1]['precipitacion'])#guardamos la cantidad de patrones extraidos

	patronesDict = {#creacion del diccionario con los valores que utilizamos
        'precipitaciones':'',
        'velocidad_media_viento':'',
        'velocidad_maxima_viento':'',
        'direccion_media_viento':'',
        'direccion_maxima_viento':'',
        'humedad_relativa':'',
        'insolacion':'',
        'presion':'',
        'temperatura_suelo':'',
        'temperatura':'',
        'temperatura_minima':'',
        'temperatura_maxima':'',
        'visibilidad':'',
        'recorrido_viento':'',
        'nieve':'',
        'energia_generada': ''
    }

	patrones = []

	for i in range(0,num_patrones,1):#para los patrones recibididos, se intenta extraer la informacion posible y adjuntarla al diccionario

		"""try:
			patronesDict["fecha"] = resp_aemet['elaborado']
		except:
			patronesDict["fecha"] = None
		try:
			patronesDict["ubicacion"] = resp_aemet['nombre']
		except:
			patronesDict["ubicacion"] = None
		patronesDict["codigo_municipio"] = municipio
		"""

		#los patrones que no se puedan coges se pondran a -1 uno para evitar problemas
		try:
			patronesDict["precipitaciones"] = resp_aemet['prediccion']['dia'][1]['precipitacion'][i]["value"]
		except:
			patronesDict["precipitaciones"] = -1
		patronesDict["velocidad_media_viento"] = -1
		try:
			patronesDict["velocidad_maxima_viento"] = resp_aemet['prediccion']['dia'][1]["vientoAndRachaMax"][i]["velocidad"][0]
		except:
			patronesDict["velocidad_maxima_viento"] = -1
		patronesDict["direccion_media_viento"] = -1
		try:
			#patronesDict["direccion_maxima_viento"] = resp_aemet['prediccion']['dia'][1]["vientoAndRachaMax"][i]["direccion"][0]
			patronesDict["direccion_maxima_viento"] = -1#temporalmente a none es necesario cambia de norte sur este oeste y calma a float
		except:
			patronesDict["direccion_maxima_viento"] = -1
		try:
			patronesDict["humedad_relativa"] = resp_aemet['prediccion']['dia'][1]["humedadRelativa"][i]["value"]
		except:
			patronesDict["humedad_relativa"] = -1
		patronesDict["insolacion"] = -1
		patronesDict["presion"] = -1
		patronesDict["temperatura_suelo"] = -1
		try:
			patronesDict["temperatura"] = resp_aemet['prediccion']['dia'][1]["temperatura"][i]["value"]
		except:
			patronesDict["temperatura"] = -1
		patronesDict["temperatura_minima"] = -1
		patronesDict["temperatura_maxima"] = -1
		patronesDict["visibilidad"] = -1
		patronesDict["recorrido_viento"] = -1
		try:
			patronesDict["nieve"] = resp_aemet['prediccion']['dia'][1]["nieve"][i]["value"]
		except:
			patronesDict["nieve"] = -1

		patronesDict["energia_generada"] = 0

		patrones.append(patronesDict.copy())#cada patron se va almacenando en una lista que posteriormente se devuelve en un json

	return JSONResponse(content=patrones)


#**********************************************************************************************************************************************
#*************************************************************FIN_PETICIONES_GET***************************************************************
#**********************************************************************************************************************************************



#**********************************************************************************************************************************************
#*************************************************************FUNCIONES_DE_APOYO***************************************************************
#**********************************************************************************************************************************************


#funcion que realiza las peticiones get a la api ree e inserta en la bd
def pedir_tiempo(idema):

	#se realiza el get pasando la clave url y el header
	url = "https://opendata.aemet.es/opendata/api/observacion/convencional/datos/estacion/"+str(idema)

	querystring = {
		"api_key":"eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJyaG9ybWlnbzk5QGdtYWlsLmNvbSIsImp0aSI6IjVjZTNkMjMxLTFlZWYtNDIzMC1hMGIyLTNiM2Q4ZWNiMWVjMSIsImlzcyI6IkFFTUVUIiwiaWF0IjoxNjU4OTk5NTAzLCJ1c2VySWQiOiI1Y2UzZDIzMS0xZWVmLTQyMzAtYTBiMi0zYjNkOGVjYjFlYzEiLCJyb2xlIjoiIn0.FL0Erp4mvRqo6uo_4Rp21bUaTod26dl08fMoDiPUHfM"
		}

	headers = {
		'cache-control': "no-cache"
		}
	

	resp_aemet = requests.request("GET", url, headers=headers, params=querystring)

	#se comprueba si ha fallado
	if not resp_aemet.status_code == 200:
		raise Exception("Incorrect reply from Aemet API. Status code: {}. Text: {}".format(resp_aemet.status_code, resp_aemet.text))

	#si ha funcionado correctamente, extraemos los datos de la peticion
	url = resp_aemet.json()['datos']
	resp_aemet = (requests.request("GET", url, headers=headers)).json()

	num_patrones = len(resp_aemet)#guardamos la cantidad de patrones extraidos

	patronesDict = {#creacion del diccionario con los valores que utilizamos
		'fecha': '',
		'ubicacion':'',
		'idema':'',
		'precipitaciones':'',
		'velocidad_media_viento':'',
		'velocidad_maxima_viento':'',
		'direccion_media_viento':'',
		'direccion_maxima_viento':'',
		'humedad_relativa':'',
		'insolacion':'',
		'presion':'',
		'temperatura_suelo':'',
		'temperatura':'',
		'temperatura_minima':'',
		'temperatura_maxima':'',
		'visibilidad':'',
		'recorrido_viento':'',
		'nieve':''
	}

	listaPatrones = []
	
	for i in range(0,num_patrones,1):#recorremos todo el json extrayendo los datos que necesitamos

		try:
			patronesDict['fecha'] = resp_aemet[i]['fint']
			try:
				patronesDict['ubicacion'] = resp_aemet[i]['ubi']
				try:
					patronesDict['idema'] = resp_aemet[i]['idema']
					try:
						patronesDict['precipitaciones'] = resp_aemet[i]['prec']
					except:
						patronesDict['precipitaciones'] = 0
					try:
						patronesDict['velocidad_media_viento'] = resp_aemet[i]['vv']
					except:
						patronesDict['velocidad_media_viento'] = -1
					try:
						patronesDict['velocidad_maxima_viento'] = resp_aemet[i]['vmax']
					except:
						patronesDict['velocidad_maxima_viento'] = -1
					try:
						patronesDict['direccion_media_viento'] = resp_aemet[i]['dv']
					except:
						patronesDict['direccion_media_viento'] = -1
					try:
						patronesDict['direccion_maxima_viento'] = resp_aemet[i]['dmax']
					except:
						patronesDict['direccion_maxima_viento'] = -1
					try:
						patronesDict['humedad_relativa'] = resp_aemet[i]['hr']
					except:
						patronesDict['humedad_relativa'] = 0
					try:
						patronesDict['insolacion'] = resp_aemet[i]['inso']
					except:
						patronesDict['insolacion'] = 0
					try:
						patronesDict['presion'] = resp_aemet[i]['pres']
					except:
						patronesDict['presion'] = -1
					try:
						patronesDict['temperatura_suelo'] = resp_aemet[i]['ts']
					except:
						patronesDict['temperatura_suelo'] = -1
					try:
						patronesDict['temperatura'] = resp_aemet[i]['ta']
					except:
						patronesDict['temperatura'] = -1
					try:
						patronesDict['temperatura_minima'] = resp_aemet[i]['tamin']
					except:
						patronesDict['temperatura_minima'] = -1
					try:
						patronesDict['temperatura_maxima'] = resp_aemet[i]['tamax']
					except:
						patronesDict['temperatura_maxima'] = -1
					try:
						patronesDict['visibilidad'] = resp_aemet[i]['vis']
					except:
						patronesDict['visibilidad'] = -1
					try:
						patronesDict['recorrido_viento'] = resp_aemet[i]['rviento']
					except:
						patronesDict['recorrido_viento'] = -1
					try:
						patronesDict['nieve'] = resp_aemet[i]['nieve']
					except:
						patronesDict['nieve'] = 0
					
				except:
					print('Error obteniendo patron')
			
			except:
				print('Error obteniendo patron')
			
		except:
			print('Error obteniendo patron')

		listaPatrones.append(patronesDict.copy())#cada patron se va almacenando en una lista que posteriormente se devuelve 

	return listaPatrones

	
def pedir_generacion(id, leng):
	"""
		Aquí se recolectarían los datos de las placas solares, como no disponemos de ellos, generaremos los datos aleatoriamente y los guardaremos en un JSON con el formato simulado.
		Para mayor sencillez asumiremos que el id de cada planta es igual al idema de aemet
	"""
	
	placasDict = {
		'ubicacion' : '',
		'id' : '',
		'fecha': '',
		'energia_generada':''
	}

	placasLista = []

	seed(1)

	for i in range (0,leng,1):
		placasDict['ubicacion'] = 'Córdoba'
		placasDict['id'] = id
		placasDict['energia_generada'] = random()*5000 #generamos número entre 0 y 5000
		placasLista.append(placasDict.copy())


	return placasLista


def fusionar_datos(tiempo, placas):
	"""
		En el programa final habría que añadir comprobaciones de que las fechas coinciden
	"""

	if len(tiempo) != len(placas):
		print('lentiempo:',len(tiempo),', len placas:',len(placas))
		return -1
	elif str(tiempo[0]['idema']) != str(placas[0]['id']):
		print('idtiempo:',tiempo[0]['idema'] ,', idplacas:',placas[0]['id'])
		return -1

	json_final = {
		'fecha': '',
		'ubicacion':'',
		'idema':'',
		'precipitaciones':'',
		'velocidad_media_viento':'',
		'velocidad_maxima_viento':'',
		'direccion_media_viento':'',
		'direccion_maxima_viento':'',
		'humedad_relativa':'',
		'insolacion':'',
		'presion':'',
		'temperatura_suelo':'',
		'temperatura':'',
		'temperatura_minima':'',
		'temperatura_maxima':'',
		'visibilidad':'',
		'recorrido_viento':'',
		'nieve':'',
		'energia_generada' : ''
	}

	lista_final = []

	for i in range(0, len(tiempo), 1):
		json_final['fecha'] = tiempo[i]['fecha']
		json_final['ubicacion'] = tiempo[i]['ubicacion']
		json_final['idema'] = tiempo[i]['idema']
		json_final['precipitaciones'] = tiempo[i]['precipitaciones']
		json_final['velocidad_media_viento'] = tiempo[i]['velocidad_media_viento']
		json_final['velocidad_maxima_viento'] = tiempo[i]['velocidad_maxima_viento']
		json_final['direccion_media_viento'] = tiempo[i]['direccion_media_viento']
		json_final['direccion_maxima_viento'] = tiempo[i]['direccion_maxima_viento']
		json_final['humedad_relativa'] = tiempo[i]['humedad_relativa']
		json_final['insolacion'] = tiempo[i]['insolacion']
		json_final['presion'] = tiempo[i]['presion']
		json_final['temperatura_suelo'] = tiempo[i]['temperatura_suelo']
		json_final['temperatura'] = tiempo[i]['temperatura']
		json_final['temperatura_minima'] = tiempo[i]['temperatura_minima']
		json_final['temperatura_maxima'] = tiempo[i]['temperatura_maxima']
		json_final['visibilidad'] = tiempo[i]['visibilidad']
		json_final['recorrido_viento'] = tiempo[i]['recorrido_viento']
		json_final['nieve'] = tiempo[i]['nieve']
		json_final['energia_generada'] = placas[i]['energia_generada']
		lista_final.append(json_final.copy())

	return lista_final

	

#esta funcion inserta el patron en la bd
def insert_Pattern(conn, variable_list):
	#insertar un nuevo patron en la tabla datos_generacion 
	sql = """INSERT INTO datos_generacion(fecha, ubicacion, idema, precipitaciones, velocidad_media_viento, velocidad_maxima_viento, direccion_media_viento, 
	direccion_maxima_viento, humedad_relativa, insolacion, presion, temperatura_suelo, temperatura, temperatura_minima, temperatura_maxima, visibilidad,recorrido_viento, nieve, energiagenerada)
						 VALUES(%s,      %s,  %s,  %s,   %s,         %s,      %s,                 %s,             %s,         %s,              %s,             %s,              %s,                     %s,      %s,            %s,                                   %s,                              %s,                   %s) ;"""
	try:
		# se crea un nuevo cursor
		cur = conn.cursor()
		# se ejecuta la declaracion sql creada anteriormente
		cur.execute(sql ,(variable_list[0],variable_list[1],variable_list[2],variable_list[3],variable_list[4],variable_list[5],variable_list[6],variable_list[7],variable_list[8],variable_list[9],variable_list[10],variable_list[11],variable_list[12], variable_list[13], variable_list[14], variable_list[15],variable_list[16],variable_list[17],variable_list[18],))
		# guardamos los cambios en la base de datos
		conn.commit()
		# se cierra comunicación con la base de datos
		cur.close()
	except (Exception, psycopg2.DatabaseError) as error:
		print("Error: ", error)

	return

#**********************************************************************************************************************************************
#*********************************************************FIN_FUNCIONES_DE_APOYO***************************************************************

#**********************************************************************************************************************************************