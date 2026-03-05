import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. CARGAR LOS DATOS
df = pd.read_csv('house_prices.csv')

# 2. ESTADÍSTICA DESCRIPTIVA BÁSICA
# Imprime un resumen con la media, min, max, y cuartiles de cada columna
print("--- Estadística Descriptiva ---")
print(df.describe())

# 3. PREPROCESAMIENTO
# Eliminar columnas que no nos sirven para predecir ('id' es un identificador y 'date' es la fecha)
df_model = df.drop(['id', 'date'], axis=1)

# Convertir variables de texto a números.
# Si tu dataset tiene 'waterfront' como 'Y'/'N' o 'condition' como texto, esto las vuelve numéricas
df_model = pd.get_dummies(df_model, columns=['waterfront', 'condition'], drop_first=True)

# Separar características (X) y el objetivo a predecir (y)
X = df_model.drop('price', axis=1)
y = df_model['price']

# Dividir en datos de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos: Las Redes Neuronales (MLP) necesitan que todos los datos estén en la misma escala
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. ENTRENAR MODELOS
print("\nEntrenando Random Forest...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("Entrenando MLP (Red Neuronal)...")
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
mlp.fit(X_train_scaled, y_train)
mlp_pred = mlp.predict(X_test_scaled)

# 5. FUNCIÓN DE EVALUACIÓN
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'Model': model_name, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

rf_metrics = evaluate_model(y_test, rf_pred, 'Random Forest')
mlp_metrics = evaluate_model(y_test, mlp_pred, 'MLP')

# Mostrar las métricas
metrics_df = pd.DataFrame([rf_metrics, mlp_metrics])
print("\n--- Resultados de Evaluación ---")
print(metrics_df)

# 6. GRAFICAR
plt.figure(figsize=(14, 6))

# Gráfico del Random Forest
plt.subplot(1, 2, 1)
plt.scatter(y_test, rf_pred, alpha=0.3, color='blue')
# Línea ideal (donde la predicción es exactamente igual al precio real)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title(f'Random Forest\n$R^2$: {rf_metrics["R2"]:.3f}')
plt.xlabel('Precio Real')
plt.ylabel('Precio Predicho')

# Gráfico del MLP
plt.subplot(1, 2, 2)
plt.scatter(y_test, mlp_pred, alpha=0.3, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title(f'MLP (Red Neuronal)\n$R^2$: {mlp_metrics["R2"]:.3f}')
plt.xlabel('Precio Real')
plt.ylabel('Precio Predicho')

plt.tight_layout()
plt.savefig('comparacion_modelos.png')
print("\nEl gráfico se ha guardado como 'comparacion_modelos.png'")