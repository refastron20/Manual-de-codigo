#codigo que permite la extraccion de datos de la api del estado ree a nuestra base de datos y funciones similares


#import necesarios para el funcionamiento de la aplicación
from fastapi import FastAPI
from datetime import datetime
from datetime import timedelta
import numpy as np

import psycopg2
import requests
import json
import holidays


app = FastAPI()



#**********************************************************************************************************************************************
#*****************************************************************PETICIONES_GET***************************************************************
#**********************************************************************************************************************************************

@app.get("/")#cuando se realiza un get en la api no se realiza ninguna accion
def read_root():

	return {"No se realiza ninguna acción"}



@app.get("/obtener_patron_dia/")#realizamos get que nos da los patrones del dia actual para obtener los precios el dia siguiente

def read_root():
	#creamos el los patrones con los que vamos a trabajar
	fecha_actual = datetime.now()#vemos la fecha actual
	fecha_actual = fecha_actual.replace(hour=0, minute=0, second=0, microsecond=0)#ajustamos a la hora 0:0
	fecha_dia_siguiente = fecha_actual.replace(hour=23, minute=55)#adelantamos un dia


	#parametros para todas las peticiones get
	getParameters_generacion = {
	"start_date": fecha_actual,
	"end_date": fecha_dia_siguiente,
	"time_trunc": "day",
	}

	#peticion a la API del Estado demanda energia
	getParameters_demanda = {
		"start_date": fecha_actual,
		"end_date": fecha_dia_siguiente,
		"time_trunc": "hour",
	}

	#peticion a la API del Estado mercado de la energia
	getParameters_mercado = {
		"start_date": fecha_actual - timedelta(hours = 2),
		"end_date": fecha_dia_siguiente,
		"time_trunc": "hour",
	}

	#get a la api sobre la demanda
	resp_demanda = requests.get("https://apidatos.ree.es/es/datos/demanda/demanda-tiempo-real",
		params = getParameters_demanda
		)

	#se comprueba si ha fallado
	if not resp_demanda.status_code == 200:
		raise Exception("Incorrect reply from Ree API. Status code: {}. Text: {}".format(resp_demanda.status_code, resp_demanda.text))

	#get a la api sobre la generacion de energia
	resp_generacion_energia = requests.get("https://apidatos.ree.es/es/datos/generacion/estructura-generacion",
		params = getParameters_generacion
		)

	#se comprueba si ha fallado
	if not resp_generacion_energia.status_code == 200:
		raise Exception("Incorrect reply from Ree API. Status code: {}. Text: {}".format(resp_generacion_energia.status_code, resp_generacion_energia.text))

	#get a la api sobre los precios de la energia
	resp_mercado_precios = requests.get("https://apidatos.ree.es/es/datos/mercados/precios-mercados-tiempo-real",
		params = getParameters_mercado
		)
	#se comprueba si ha fallado
	if not resp_mercado_precios.status_code == 200:
		raise Exception("Incorrect reply from Ree API. Status code: {}. Text: {}".format(resp_mercado_precios.status_code, resp_mercado_precios.text))


	num_patrones = len(resp_mercado_precios.json()["included"][1]["attributes"]["values"])#comprobamos la cantidad de patrones que tenemos
	v_patrones = np.zeros(shape=(num_patrones-2, 18))#se crea vector donde se almacenaran todos los patrones
	iterador_demanda = 0


	for hora in range(0,num_patrones-2,1):#recorremos los patrones para darle el formato que deseamos
		
		v_patrones[hora][0] = int(fecha_actual.strftime("%Y"))
		v_patrones[hora][1] = int(fecha_actual.strftime("%m"))
		v_patrones[hora][2] = int(fecha_actual.strftime("%d"))
		v_patrones[hora][3] = hora
		v_patrones[hora][4] = fecha_actual.weekday()
		v_patrones[hora][5] = fecha_actual in holidays.ES()
		try:
			v_patrones[hora][6] = resp_generacion_energia.json()["included"][0]["attributes"]["values"][0]["value"]
		except:
			v_patrones[hora][6] = None
		try:
			v_patrones[hora][7] = resp_generacion_energia.json()["included"][9]["attributes"]["values"][0]["value"]
		except:
			v_patrones[hora][7] = None
		try:
			v_patrones[hora][8] = resp_generacion_energia.json()["included"][10]["attributes"]["values"][0]["value"]
		except:
			v_patrones[hora][8] = None
		try:
			v_patrones[hora][9] = resp_generacion_energia.json()["included"][2]["attributes"]["values"][0]["value"]
		except:
			v_patrones[hora][9] = None
		try:
			v_patrones[hora][10] = resp_generacion_energia.json()["included"][3]["attributes"]["values"][0]["value"]
		except:
			v_patrones[hora][10] = None
		try:
			v_patrones[hora][11] = resp_generacion_energia.json()["included"][4]["attributes"]["values"][0]["value"]
		except:
			v_patrones[hora][11] = None
		try:
			v_patrones[hora][12] = resp_generacion_energia.json()["included"][7]["attributes"]["values"][0]["value"]
		except:
			v_patrones[hora][12] = None
		
		v_patrones[hora][13] = 0
		hora_demanda = v_patrones[hora][3]

		divisor = 0
		
		while hora_demanda == v_patrones[hora][3]:
			try:
				v_patrones[hora][13] += resp_demanda.json()["included"][2]["attributes"]["values"][iterador_demanda]["value"]
			except:
				v_patrones[hora][13] += 0

			iterador_demanda+=1
			divisor+=1
			try:
				hora_demanda = resp_demanda.json()["included"][2]["attributes"]["values"][iterador_demanda]["datetime"]
			except:
				hora_demanda = '0000-00-00000'
			hora_demanda = hora_demanda.split("-")
			hora_demanda = int(hora_demanda[2][3]+hora_demanda[2][4])
	
			v_patrones[hora][13] = v_patrones[hora][12]/divisor #se realihoraa la media de los 6 datos que hay en cada hora
			v_patrones[hora][14] = resp_mercado_precios.json()["included"][1]["attributes"]["values"][hora]["value"]
			v_patrones[hora][15] = resp_mercado_precios.json()["included"][1]["attributes"]["values"][hora+1]["value"]
			v_patrones[hora][16] = resp_mercado_precios.json()["included"][1]["attributes"]["values"][hora+2]["value"]
			v_patrones[hora][17] = 0

	return json.dumps(v_patrones.tolist())#devolvemos una lista con los datos



@app.get("/items/")#get que dadas dos fechas obtiene los patrones disponibles de la api ree

def read_root(fecha_inicio_get: str, fecha_fin_get: str):#ejecuta el codigo para las fechas pasadas en el get

	#conexion a la base de datos
	conn = psycopg2.connect(database="modelosbd", user="postgres", password="password", host="timescaledb", port="5432")
	
	#variables que indican desde donde hasta cuando se extraen datos de la api ree
	fecha_inicio = fecha_inicio_get
	fecha_fin = fecha_fin_get

	fecha_inicio = fecha_inicio.split("-")
	fecha_fin = fecha_fin.split("-")

	date1 = datetime(int(fecha_inicio[0]), int(fecha_inicio[1]), int(fecha_inicio[2]))
	date2 = datetime(int(fecha_fin[0]), int(fecha_fin[1]), int(fecha_fin[2]))
	date2 = date2 + timedelta(days=1)

	date3 = date2 - date1

	inicio = date1

	#como las peticiones de ree tienen cierto limite, dividimos las peticiones en dias 20 y con un bucle for se completa el total de dias indicado anteriormente
	for i in range(1, date3.days-20, 20):
		fin = inicio + timedelta(days = 20)

		pedir_api(conn, inicio, fin, ((fin-inicio).days+1)*24)#llamamos a la funcion que extrae los datos
		inicio = fin

	fin = date2
	
	pedir_api(conn, inicio, fin, ((fin-inicio).days+1)*24)#esta llamada ocurre cuando quedan menos de 20 dias, de tal forma que se hace la peticion con los dias que se necesiten

	conn.close()#cerramos conexion con la base de datos

	return {"Exito al añadir patrones de la api a la base de datos"}



@app.get("/actualizar/")#get que actualiza la base de datos, actualiza hasta el dia anterior al actual

def read_root():#actualiza la base de datos

	#conexion a la base de datos
	conn = psycopg2.connect(database="modelosbd", user="postgres", password="password", host="timescaledb", port="5432")
	cur = conn.cursor()#se crea cursor

	fecha_actual = datetime.now()#vemos la fecha actual
	fecha_actual = fecha_actual - timedelta(days=1)#le restamos uno porque actualizamos hasta el dia anterior

	#Actualizacion de la base de datos
	cur.execute("Select anio, mes, dia, hora from precio order by fecha desc limit 1")
	v_ultima_fecha_bd = None
	v_ultima_fecha_bd = cur.fetchall()#sacamos la ultima fecha de la base de datos

	if not v_ultima_fecha_bd:#comprobamos si esta vacia la base de datos
		fecha_inicial_adquisicion_datos = datetime(2021, 6, 1)#si la base de datos esta vacia cogemos a partir de esta fecha
		#peticion a la nuestra API bdconexion
		fecha_actual.replace(hour = 23, minute=50)
		getParameters_fechas = {
			"fecha_inicio_get": str(fecha_inicial_adquisicion_datos.strftime("%Y-%m-%d")),
			"fecha_fin_get": str(fecha_actual.strftime("%Y-%m-%d")),
		}
		requests.get("http://bdconapi:8001/items",
			params = getParameters_fechas
			)
	else:#si no esta vacia, cogemos desde la ultima fecha de la base de datos hasta la actual - 1
		ultima_fecha_bd = datetime(v_ultima_fecha_bd[0][0], v_ultima_fecha_bd[0][1], v_ultima_fecha_bd[0][2], v_ultima_fecha_bd[0][3])#cogemos la fecha mas nueva
		
		if ultima_fecha_bd.replace(hour = 0, minute=0, second=0, microsecond=0) < fecha_actual.replace(hour = 0, minute=0, second=0, microsecond=0):#vemos si la base de datos se encuentra actualizada
			ultima_fecha_bd = ultima_fecha_bd + timedelta(days=1)#incrementamos en un dia para empezar desde ahi
			#peticion a la nuestra API bdconexion
			fecha_actual.replace(hour = 23, minute=50)#la fecha final llegara hasta el final del dia
			getParameters_fechas = {
				"fecha_inicio_get": str(ultima_fecha_bd.strftime("%Y-%m-%d")),#necesario eliminar los minutos y segundos para que funcione correctamente
				"fecha_fin_get": str(fecha_actual.strftime("%Y-%m-%d")),
			}
			requests.get("http://bdconapi:8001/items",
				params = getParameters_fechas
				)

	return {"actualizar": "Ok"}



#**********************************************************************************************************************************************
#*************************************************************FIN_PETICIONES_GET***************************************************************
#**********************************************************************************************************************************************



#**********************************************************************************************************************************************
#*************************************************************FUNCIONES_DE_APOYO***************************************************************
#**********************************************************************************************************************************************


#funcion que realiza las peticiones get a la api ree e inserta en la bd
def pedir_api(conn, fecha_inicio, fecha_fin, horas):
	#peticion a la API del Estado generacion energia

	fiestas = holidays.ES()
	if fecha_inicio.strftime(("%Y-%m-%d")) == "2021-06-01" :
		fecha_inicio = fecha_inicio.replace(hour=2, minute=0)
	else:
		fecha_inicio.replace(hour=0, minute=0)

	fecha_inicio = fecha_inicio - timedelta(hours=2)
	fecha_inicio_str = fecha_inicio.strftime("%Y-%m-%dT%H:%M")
	fecha_fin_str = fecha_fin.strftime("%Y-%m-%d")+"T23:55"


	#peticion a la API del Estado generacion enegia
	getParameters_generacion = {
		"start_date": fecha_inicio_str,
		"end_date": fecha_fin_str,
		"time_trunc": "day",
	}
	#peticion a la API del Estado demanda energia
	getParameters_demanda = {
		"start_date": fecha_inicio_str,
		"end_date": fecha_fin_str,
		"time_trunc": "hour",
	}
	#peticion a la API del Estado mercado de la energia
	getParameters_mercado = {
		"start_date": fecha_inicio_str,
		"end_date": fecha_fin_str,
		"time_trunc": "hour",
	}

	#get a la api sobre la demanda
	resp_demanda = requests.get("https://apidatos.ree.es/es/datos/demanda/demanda-tiempo-real",
		params = getParameters_demanda
		)

	#se comprueba si ha fallado
	if not resp_demanda.status_code == 200:
		raise Exception("Incorrect reply from Ree API. Status code: {}. Text: {}".format(resp_demanda.status_code, resp_demanda.text))

	#get a la api sobre la generacion de energia
	resp_generacion_energia = requests.get("https://apidatos.ree.es/es/datos/generacion/estructura-generacion",
		params = getParameters_generacion
		)

	#se comprueba si ha fallado
	if not resp_generacion_energia.status_code == 200:
		raise Exception("Incorrect reply from Ree API. Status code: {}. Text: {}".format(resp_generacion_energia.status_code, resp_generacion_energia.text))

	#get a la api sobre los precios de la energia
	resp_mercado_precios = requests.get("https://apidatos.ree.es/es/datos/mercados/precios-mercados-tiempo-real",
		params = getParameters_mercado
		)
	#se comprueba si ha fallado
	if not resp_mercado_precios.status_code == 200:
		raise Exception("Incorrect reply from Ree API. Status code: {}. Text: {}".format(resp_mercado_precios.status_code, resp_mercado_precios.text))

	dia = 0
	dia_anterior = int(fecha_inicio.strftime("%d"))
	iterador_demanda = 0

	num_patrones = len(resp_mercado_precios.json()["included"][0]["attributes"]["values"])

	patrones = np.empty((num_patrones,1),dtype=[('fecha',np.unicode_,25), ('anio',np.int_), ('mes',np.int_), ('dia',np.int_), ('hora',np.int_), ('dia_semana',np.int_), ('festivo',np.bool_), 
	('energia_hidraulica',np.float_), ('energia_eolica',np.float_), ('energia_fv',np.float_), ('energia_nuclear',np.float_), ('energia_carbon',np.float_), ('energia_fuelgas',np.float_), 
	('energia_ciclocombinado',np.float_), ('demanda',np.float_),('precio_dos_horas_antes',np.float_), ('precio_hora_antes',np.float_),  ('precio_actual',np.float_), 
	('precio_dia_siguiente',np.float_)])

	for i in range(2,num_patrones+2,1):#este bucle se recorre en horas para permitir realizar los calculos en horas y ajustar los parametros de precios
		#se intentan extraer todos los datos de la api necesarios para la bd, en el caso de fallo se extraen None y en el caso de la fecha se descarta el patron
		x=i-2
		try:
			#se añade la informacion relacionada con el tiempo al patron
			patrones['fecha'][x] = resp_mercado_precios.json()["included"][1]["attributes"]["values"][i]["datetime"]
			fecha = str(patrones['fecha'][x][0]).split("-")
			patrones['hora'][x] = int(fecha[2][3]+fecha[2][4])
			fecha = datetime(int(fecha[0]), int(fecha[1]), int(fecha[2][0]+fecha[2][1]))
			patrones['anio'][x] = int(fecha.strftime("%Y"))
			patrones['mes'][x] = int(fecha.strftime("%m"))
			patrones['dia'][x] = int(fecha.strftime("%d"))

			if patrones['dia'][x] != dia_anterior:
				dia+=1
			dia_anterior = patrones['dia'][x]
			patrones['dia_semana'][x] = fecha.weekday()
			patrones['festivo'][x] = fecha in fiestas

			#se añaden enegias al patron
			try:
				patrones['energia_hidraulica'][x] = resp_generacion_energia.json()["included"][0]["attributes"]["values"][dia]["value"]
			except:
				patrones['energia_hidraulica'][x] = None
			try:
				patrones['energia_eolica'][x] = resp_generacion_energia.json()["included"][9]["attributes"]["values"][dia]["value"]
			except:
				patrones['energia_eolica'][x] = None
			try:
				patrones['energia_fv'][x] = resp_generacion_energia.json()["included"][10]["attributes"]["values"][dia]["value"]
			except:
				patrones['energia_fv'][x] = None
			try:
				patrones['energia_nuclear'][x] = resp_generacion_energia.json()["included"][2]["attributes"]["values"][dia]["value"]
			except:
				patrones['energia_nuclear'][x] = None
			try:
				patrones['energia_carbon'][x] = resp_generacion_energia.json()["included"][3]["attributes"]["values"][dia]["value"]
			except:
				patrones['energia_carbon'][x] = None
			try:
				patrones['energia_fuelgas'][x] = resp_generacion_energia.json()["included"][4]["attributes"]["values"][dia]["value"]
			except:
				patrones['energia_fuelgas'][x] = None
			try:
				patrones['energia_ciclocombinado'][x] = resp_generacion_energia.json()["included"][7]["attributes"]["values"][dia]["value"]
			except:
				patrones['energia_ciclocombinado'][x] = None

			#se añaden precios al patron
			try:
				patrones['precio_actual'][x] = resp_mercado_precios.json()["included"][1]["attributes"]["values"][i]["value"]
			except:
				patrones['precio_actual'][x] = -1
			try:
				patrones['precio_dos_horas_antes'][x] = resp_mercado_precios.json()["included"][1]["attributes"]["values"][i-2]["value"]
			except:
				patrones['precio_dos_horas_antes'][x] = -1
			try:
				patrones['precio_hora_antes'][x] = resp_mercado_precios.json()["included"][1]["attributes"]["values"][i-1]["value"]
			except:
				patrones['precio_hora_antes'][x] = -1
			
			#se añade la demanda aparte
			patrones['demanda'][x] = 0
			hora_demanda = patrones['hora'][x]

			divisor = 0
			
			while hora_demanda == patrones['hora'][x]:
				try:
					patrones['demanda'][x] += resp_demanda.json()["included"][0]["attributes"]["values"][iterador_demanda]["value"]
				except:
					patrones['demanda'][x] += 0

				iterador_demanda+=1
				divisor+=1
				try:
					hora_demanda = resp_demanda.json()["included"][0]["attributes"]["values"][iterador_demanda]["datetime"]
				except:
					hora_demanda = '0000-00-00000'
				hora_demanda = hora_demanda.split("-")
				hora_demanda = int(hora_demanda[2][3]+hora_demanda[2][4])
				
			patrones['demanda'][x] = patrones['demanda'][x]/divisor #se realiza la media de los 6 datos que hay en cada hora

			#los parametros obtenidos se insertan en la base de datos
			
		except:
			print("No se encontró fecha")

	#esta parte del codigo añade el precio del dia siguiente al patron, se tiene en cuenta si el dia tiene 23, 24 o 25 horas
	x = 0
	while patrones[x]['dia'] != patrones[num_patrones-3]['dia'] and x<num_patrones-3:
		if(patrones[x]['hora']==patrones[x+23]['hora']):
			patrones[x]['precio_dia_siguiente'] = patrones[x+23]['precio_actual']
		elif(patrones[x]['hora']==patrones[x+24]['hora']):
			patrones[x]['precio_dia_siguiente'] = patrones[x+24]['precio_actual']
		elif(patrones[x]['hora']==patrones[x+25]['hora']):
			patrones[x]['precio_dia_siguiente'] = patrones[x+25]['precio_actual']

		#insertamos a la base de datos, poniendo el formato deseado
		lista_valores = patrones[x].tolist()
		lista_valores_str = str(lista_valores).replace('[','')
		lista_valores_str = lista_valores_str.replace('(','')
		lista_valores_str = lista_valores_str.replace(')','')
		lista_valores_str = lista_valores_str.replace(']','')
		lista_valores = lista_valores_str.split(',')
		insert_Pattern(conn, lista_valores)
		x=x+1



#esta funcion inserta el patron en la bd
def insert_Pattern(conn, variable_list):
	#insertar un nuevo patron en la tabla precio 
	sql = """INSERT INTO precio(fecha, anio, mes, dia, hora, dia_semana, festivo, energia_hidraulica, energia_eolica, energia_fv, energia_nuclear, energia_carbon, energia_fuelgas, energia_ciclocombinado, demanda, precio_dos_horas_antes, precio_hora_antes,precio_actual, precio_dia_siguiente)
						 VALUES(%s,      %s,  %s,  %s,   %s,         %s,      %s,                 %s,             %s,         %s,              %s,             %s,              %s,                     %s,      %s,            %s,                                   %s,                              %s,                   %s) ;"""
	try:
		# sse crea un nuevo cursor
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