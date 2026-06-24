# Proyecto Python: Análisis Inferencial, Correlaciones y Pruebas de Hipótesis en Tendencias de la NBA

Este repositorio contiene un proyecto práctico desarrollado en Python utilizando las librerías **Pandas**, **NumPy**, **SciPy** (módulo `stats`), **Seaborn** y **Matplotlib** para realizar un Análisis Exploratorio de Datos (EDA) y evaluar la existencia de asociaciones estadísticas significativas en el contexto de la National Basketball Association (NBA). El script implementa análisis de diferencias de medias basándose en desgloses temporales, matrices de covarianza y correlación para evaluar variables predictivas, tablas de contingencia cruzadas y modelos probabilísticos de **Pruebas de Chi-cuadrado de Independencia** para validar la ventaja de la localía en los resultados deportivos.

---

## Código Python del Proyecto

El programa realiza el procesamiento de las series temporales por franquicias, calcula métricas de dispersión lineal y ejecuta las pruebas de hipótesis inferenciales:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Estandarización de visualización numérica para arreglos de NumPy
np.set_printoptions(suppress=True, precision=2)

# --- 1. Ingesta y Segmentación Temporal ---
nba = pd.read_csv('nba_games.csv')
nba_2010 = nba[nba.year_id == 2010]
nba_2014 = nba[nba.year_id == 2014]

# --- 2. Análisis Comparativo Univariado (Knicks vs Nets - 2010) ---
knicks_pts_10 = nba_2010[nba_2010['fran_id'] == 'Knicks']['pts']
nets_pts_10 = nba_2010[nba_2010['fran_id'] == 'Nets']['pts']

diff_means_2010 = knicks_pts_10.mean() - nets_pts_10.mean()
print("Diferencia de medias (2010): " + str(diff_means_2010))

# Visualización de histogramas superpuestos
plt.hist(knicks_pts_10, alpha=0.5, label='Knicks')
plt.hist(nets_pts_10, alpha=0.5, label='Nets')
plt.legend()
plt.title("Distribución de Puntos (2010)")
plt.xlabel("Puntos")
plt.ylabel("Frecuencia")
plt.savefig("hist_puntos_2010.png")
plt.close()

# --- 3. Análisis de Tendencia Longitudinal (Knicks vs Nets - 2014) ---
knicks_pts_14 = nba_2014[nba_2014['fran_id'] == 'Knicks']['pts']
nets_pts_14 = nba_2014[nba_2014['fran_id'] == 'Nets']['pts']
diff_means_2014 = knicks_pts_14.mean() - nets_pts_14.mean()

# --- 4. Distribución General por Franquicias (Boxplot) ---
sns.boxplot(data=nba_2010, x='fran_id', y='pts')
plt.title("Puntos por Partido por Equipo (2010)")
plt.xlabel("Franquicia")
plt.ylabel("Puntos")
plt.xticks(rotation=45)
plt.savefig("boxplot_franquicias_2010.png")
plt.close()

# --- 5. Análisis de Variables Categóricas (Localía vs Resultado) ---
# Tabla de frecuencias cruzadas (Matriz de Contingencia)
location_result_freq = pd.crosstab(nba_2010['game_location'], nba_2010['game_result'])
location_result_proportions = location_result_freq / location_result_freq.to_numpy().sum()

# Prueba de Independencia de Chi-cuadrado
chi2, pval, dof, expected = chi2_contingency(location_result_freq)
print("Estadístico Chi-cuadrado: " + str(chi2))
print("P-valor de la prueba: " + str(pval))

# --- 6. Análisis Bivariado Continuo (Forecast vs Diferencia de Puntos) ---
# Cálculo del Coeficiente de Covarianza y de Correlación de Pearson
point_diff_forecast_cov = nba_2010[['forecast', 'point_diff']].cov().iloc[0, 1]
point_diff_forecast_corr = nba_2010[['forecast', 'point_diff']].corr().iloc[0, 1]

print("Covarianza Calculada: " + str(point_diff_forecast_cov))
print("Correlación de Pearson: " + str(point_diff_forecast_corr))

# Diagrama de Dispersión (Scatter Plot)
plt.scatter(nba_2010['forecast'], nba_2010['point_diff'], alpha=0.5)
plt.title("Forecast vs Diferencia de Puntos (2010)")
plt.xlabel("Forecast (Probabilidad de ganar)")
plt.ylabel("Diferencia de puntos")
plt.savefig("scatter_forecast_vs_diff.png")
plt.close()

```

---

## Análisis Metodológico e Interpretación Estadística

La ejecución de los modelos descriptivos e inferenciales sobre el set de datos de la NBA arroja conclusiones analíticas profundas estructuradas bajo tres ejes relacionales:

### 1. Variables Numéricas vs Categóricas (Distribución de Desempeño)

Al contrastar el rendimiento anotador de los Knicks frente a los Nets, el cálculo de las diferencias de las medias de los puntajes revela fluctuaciones en el dominio de las franquicias a lo largo del tiempo:

* **Temporada 2010:** Los Knicks muestran un promedio de anotación superior respecto a los Nets, desplazando el centro de la campana en el histograma hacia la derecha.
* **Temporada 2014:** La brecha se modifica debido a las reestructuraciones técnicas de las plantillas de jugadores de los equipos, alterando el sesgo de la distribución analizada en los gráficos.

### 2. Variables Categóricas vs Categóricas (Ventaja de Localía)

La tabla de contingencia cruzada mapea las interacciones entre la ubicación del partido (`game_location`: *H = Home, A = Away*) y el resultado final obtenido (`game_result`: *W = Win, L = Loss*).

Para determinar si la localía influye directamente en la probabilidad de ganar (es decir, si las variables están asociadas o son independientes), la función `chi2_contingency` evalúa las frecuencias observadas frente a una distribución teórica esperada bajo hipótesis nula de total independencia. El cálculo matemático responde a la ecuación del estadístico:

$$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$$

Donde $O_i$ representa los valores observados en tu `crosstab` y $E_i$ los valores esperados calculados de forma matemática. Un valor alto del estadístico $\chi^2$ junto con un **P-valor inferior al umbral crítico estándar de $0.05$** permite rechazar la hipótesis nula, concluyendo con rigurosidad científica que la ubicación del juego afecta significativamente los resultados de los partidos.

### 3. Variables Numéricas vs Numéricas (Precisión de Modelos)

El análisis lineal bivariado evalúa la relación entre el índice matemático predictivo de victoria (`forecast`) calculado previamente por los analistas y la diferencia de puntos real final (`point_diff`) obtenida en la cancha:

* **Covarianza:** Al arrojar un signo positivo, certifica que ambas variables se mueven en la misma dirección general (a mayor probabilidad teórica asignada, mayor margen de puntos real obtenido).
* **Correlación de Pearson:** Al aproximarse a valores cercanos a $1.0$, determina la fuerza y el grado de asociación lineal del modelo, cuantificando la exactitud analítica de las métricas predictivas de la NBA.

---

## Galería de Visualizaciones Analíticas

Para desplegar los gráficos generados directamente en la documentación, guarda las imágenes exportadas (`.png`) en la raíz del repositorio de GitHub:

| Distribución de Puntos por Franquicias | Correlación Lineal (Predicción vs Realidad) |
| --- | --- |
| <img width="642" height="522" alt="image" src="https://github.com/user-attachments/assets/02956169-1ee0-4a32-8cdf-501519842691" /> | <img width="630" height="507" alt="image" src="https://github.com/user-attachments/assets/d7a950b7-833d-4a75-8f4e-171142477a2b" /> |

---

## Conceptos Técnicos Aplicados

* **Prueba de Independencia de Chi-cuadrado ($\chi^2$)**: Prueba no paramétrica utilizada para evaluar la significancia estadística de la relación entre dos variables cualitativas nominales u ordinales en una población.
* **Coeficiente de Correlación de Pearson ($r$)**: Medida estadística que cuantifica la fuerza y la dirección de la relación lineal entre dos variables aleatorias continuas, oscilando estrictamente en un intervalo cerrado de $[-1, 1]$.
* **Tablas de Contingencia (`pd.crosstab`)**: Operador matricial que reorganiza series planas estructurando tablas de frecuencias bidimensionales para auditar la distribución conjunta de múltiples factores categóricos.

```
