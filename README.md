# Aprendizaje profundo: clasificación de fraude y regresión de viviendas

Este proyecto reúne dos aplicaciones de **aprendizaje profundo** y machine learning: la detección de transacciones fraudulentas y la estimación del precio de viviendas.

## ¿Qué es el aprendizaje profundo?

El **deep learning** es una rama del machine learning basada en redes neuronales con múltiples capas. Estas capas aprenden representaciones no lineales de los datos y permiten abordar problemas de clasificación y regresión con arquitecturas flexibles.

## Casos analizados

### 1. Clasificación de fraude

Se trabaja con transacciones de tarjetas de crédito altamente desbalanceadas: 284.807 registros, de los cuales 492 corresponden a fraude. Se comparan Decision Tree, Random Forest, Gradient Boosting y dos arquitecturas Keras. En el análisis realizado, Random Forest alcanza el mejor F1 entre los modelos clásicos (0,9087), mientras la versión secuencial de Keras registra precision 0,9720 y recall 0,8740.

### 2. Regresión de precios de viviendas

Se comparan modelos lineales, ensambles y una red neuronal funcional. El conjunto descrito en el informe corresponde a 1.728 viviendas de Saratoga County. La red Keras obtiene el menor RMSE reportado (0,3877), seguida por LGBM (0,6329).

## Estructura

- `notebooks/`: notebook principal con rutas relativas y datos preparados para ejecutarse desde el repositorio.
- `data/`: dataset de fraude comprimido y dividido en partes menores a 25 MB, una muestra rápida y la base completa de viviendas `boston_housing_esp.csv`.
- `assets/figures/`: gráficos extraídos de los resultados originales.
- `results/`: métricas consolidadas en CSV.
- `reports/`: informe profesional en PDF y Word.

## Ejecución

```bash
pip install -r requirements.txt
jupyter notebook notebooks/deep-learning-fraud-housing-models.ipynb
```

El notebook reconstruye automáticamente `creditcard.csv.gz` desde las partes pequeñas incluidas en `data/` y carga la base de viviendas desde `data/boston_housing_esp.csv`. Las partes se dividieron en archivos de menos de 10 MB para permitir su carga desde el navegador de GitHub. Esta base contiene 1.728 registros y 16 variables, y corresponde al conjunto de viviendas descrito como `SaratogaHouses` en el informe. 

## Autor

**Lea Lambrecht**  
LinkedIn: linkedin.com/in/lealambrecht  
GitHub: github.com/leanlambrecht
