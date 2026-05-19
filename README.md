# Automobile Price Prediction

Proyecto final de Machine Learning orientado a predecir el precio de automóviles a partir de características técnicas y comerciales.

## Objetivo

Construir y evaluar modelos de regresión supervisada para estimar la variable `price`.

## Dataset

El dataset contiene información de automóviles, incluyendo marca, tipo de combustible, dimensiones, peso, tamaño del motor, potencia, consumo y precio.

## Técnicas aplicadas

- Limpieza de datos
- Tratamiento de valores faltantes
- Análisis exploratorio
- Ingeniería de atributos
- Pipelines de preprocesamiento
- One-Hot Encoding
- Estandarización
- Modelos de regresión
- Comparación por MAE, RMSE y R²
- Importancia de variables

## Modelo final

El modelo seleccionado fue `Gradient Boosting`, con buen desempeño predictivo sobre el conjunto de testeo.

## Cómo ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
