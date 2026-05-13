# Tarea 6: Mathernal health Risks Prediction

## Definición del problema
El objetivo del proyecto es predecir el riesgo de problemas de salud materna según variables de presión arterial, edad, temperatura corporal.

## Definición de experimento
El experimento consite en entrenar 3 modelos con los datos para que predigan el nivel de riesgo en base a los valores de evaluación.
Los valores manejados por el dataset son: low risk, mid risk, high risk. 
Para realizar el experimento se debe utilizar un peso balanceado para cada nivel de riesgo, para evitar el sesgo hacia el nivel de riesgo con mayor cantidad de datos. Para cada nivel de riesgo se utilizarán 272 muestras.

## Modelos a usar
- Random Forest
- 
- 

## Criterio de selección de modelo
Para seleccionar el modelo a utilizar, Se priorizará en valor de recall para poder identificar correctamente a las personas con riesgo medio y alto, seguido del puntaje f1 para discriminar los modelos por la armonia que posee entre presición y recall, seguido de un analisis de la matriz de confusión para confirmar el buen desempeño del modelo.