--------------------------------------------------------------------
Este fichero contiene información sobre el esquema de la base de datos.
--------------------------------------------------------------------

1. Tabla de mercado

Esta tabla contiene información importante sobre el mercado energético.

- Variables:
	
	+ Fecha de extraccion de los datos(año/mes/dia/hora)
	+ Año de extraccion de los datos
	+ Mes de extraccion de los datos
	+ Dia de extraccion de los datos
	+ Hora de extraccion de los datos
	+ Dia de la semana representado del 0 al 6(dia de la semana)
	+ Se recoge si es un dia festivo o no(True->Sí, False->No)
	+ Generación de energía hidrálica(Mw/h)
	+ Generación de energía eólica(Mw/h)
	+ Generación de energía solar fotovoltáica(Mw/h)
	+ Generación de energía nuclear(Mw/h)
	+ Generación de energía carbon(Mw/h)
	+ Generación de energía fuelgas(Mw/h)
	+ Generación de energía ciclocombinado(Mw/h)
	+ Demanda de energía(Mw/h): demanda de energía a nivel nacional
	+ Precio de la energía dos horas antes (€/Mw)
	+ Precio de la energía la hora anterior (€/Mw)
	+ Precio de la energía actual(€/Mw)
	+ Precio de la energía del dia siguiente(€/Mw)
	
	    Column			|           Type		| Collation	| Nullable | Default 
--------------------------------------+------------------------------+--------------+----------+---------
 fecha					| timestamp with time zone 	|		| not null |
 anio					| integer			|		| not null |
 mes					| integer			|		| not null |
 dia					| integer			|		| not null |
 hora					| integer			|		| not null |
 dia_semana				| integer			| 		| not null | 
 festivo				| boolean			| 		| not null | 
 energia_hidraulica			| real				| 		|          | 
 energia_eolica			| real				| 		|          | 
 energia_fv				| real				| 		|          | 
 energia_nuclear			| real				| 		|          | 
 energia_carbon			| real				| 		|          | 
 energia_fuelgas			| real				| 		|          | 
 energia_ciclocombinado		| real				| 		|          | 
 demanda				| real				|		| not null | 
 precio_actual				| real				|		| not null | 
 precio_dos_horas_antes		| real				|		| not null | 
 precio_hora_antes			| real				|		| not null | 
 precio_dia_siguiente			| real				|		| not null | 
Indexes:
    "precio_pkey" PRIMARY KEY, btree (fecha)
    
    
    
    
