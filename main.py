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

pd.set_option('display.float_format', '{:.2f}'.format)  # Mostrar números sin notación científica
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# 2. PREPROCESAMIENTO
# Eliminar columnas que no nos sirven para predecir.
df_model = df.drop(['id', 'date'], axis=1)

# 3. ESTADÍSTICA DESCRIPTIVA BÁSICA
# Imprime un resumen con la media, min, max de cada columna
print("--- Estadística Descriptiva ---")
descripcion = df_model.describe()
descripcion.rename(index={
    'count': 'Conteo',
    'mean': 'Promedio',
    'std': 'Desv. Estándar',
    'min': 'Mínimo',
    '25%': '25%',
    '50%': 'Mediana',
    '75%': '75%',
    'max': 'Máximo'
}, inplace=True)
print(descripcion)

# Convertir variables de texto a números.
df_model = pd.get_dummies(df_model, columns=['waterfront', 'condition'], drop_first=True)

# 3.1 CORRELACIÓN CON EL PRECIO
print("\n--- Correlación con el Precio ---")
# Calcular la correlación de cada columna con 'price' y ordenar de mayor a menor
correlacion_precio = df_model.corr()['price'].sort_values(ascending=False)
print(correlacion_precio)

# Comenzar a preparar los datos para el entrenamiento.

# Separar características (X) y el objetivo a predecir (y)
X = df_model.drop('price', axis=1)
y = df_model['price']

# Dividir los datos en 80% para entrenamiento y 20% para prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. ENTRENAR MODELOS
print("\nEntrenando Random Forest...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("Entrenando MLP (Red Neuronal)...")
mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    max_iter=2000,  # Aumentado a 2000 iteraciones
    random_state=42,
    early_stopping=True,  # Detiene el entrenamiento si no hay mejora
    validation_fraction=0.2,  # Usa 20% de los datos para validación
    n_iter_no_change=50,  # Se detiene si no mejora en 50 iteraciones
    learning_rate_init=0.001  # Tasa de aprendizaje inicial
)
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