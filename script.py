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
