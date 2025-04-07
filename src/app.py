from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd

#importando la data:
url_1='https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv'
df=pd.read_csv(url_1)



#transfiriendo la data hacia un .csv
df.to_csv('raw_data.csv')



#Revisando la data:
df.info()



df.drop(['id', 'name', 'host_name','last_review', 'reviews_per_month','latitude', 'longitude' ], axis=1, inplace= True)
df.info()



#seleccion de columnas para estadisticas:
columnas_seleccionadas=['neighbourhood_group','price', 'minimum_nights', 'availability_365']

df[columnas_seleccionadas].describe()



import pandas as pd
import matplotlib.pyplot as plt

# Seleccionar solo las columnas necesarias para el analisis por localidad
df_filtered = df[['neighbourhood_group', 'room_type', 'price', 'minimum_nights', 'availability_365']]

# Comprobar si las columnas existen en el DataFrame
missing_columns = [x_col for x_col in ['neighbourhood_group', 'room_type', 'price', 'minimum_nights', 'availability_365'] if x_col not in df.columns]
if missing_columns:
    print(f"Las siguientes columnas no se encontraron en el archivo: {missing_columns}")
else:
    # Agrupar por neighbourhood_group y room_type, calculando la media de cada columna numérica
    grouped_data = df_filtered.groupby(['neighbourhood_group', 'room_type']).agg({
        'price': 'mean',
        'minimum_nights': 'mean',
        'availability_365': 'mean'
    })
    
    # Comprobar si el objeto 'grouped_data' es un DataFrame o un objeto inválido
    if isinstance(grouped_data, pd.DataFrame):
        # Convertir el índice a columnas normales
        grouped_data.reset_index(inplace=True)

        # Mostrar el DataFrame agrupado
        print("Datos agrupados:")
        print(grouped_data)

        # Graficar los datos agrupados
        plt.figure(figsize=(12, 6))
        for room_type in grouped_data['room_type'].unique():
            subset = grouped_data[grouped_data['room_type'] == room_type]
            plt.bar(subset['neighbourhood_group'], subset['price'], label=room_type)

        plt.title('Precio promedio por tipo de habitación y grupo de vecindario')
        plt.xlabel('Neighbourhood Group')
        plt.ylabel('Precio Promedio (USD)')
        plt.xticks(rotation=45)
        plt.legend(title='Room Type')
        plt.show()
    else:
        print("Ocurrió un problema al realizar la agrupación. Verifique que el DataFrame tenga las columnas adecuadas.")



df['precio_por_noche']= df['price'] / df['minimum_nights']
df['precio_por_dia_disponible']=df['price']/df['availability_365'].replace(0,1) # Para evitar división por cero
df['precio_por_review']= df['price']/df['number_of_reviews'].replace(0,1)

#Mapeo de tipo de habitacion a puntaje numerico:

room_type_mapping={'Entire home/apt': 3, 'Private room': 2, 'Shared room': 1}
df['room_type_score']= df['room_type'].map(room_type_mapping)

#Creacion de variables booleanas para validar si un alojamiento esta en Manhattan.

df['es_manhattan'] = (df['neighbourhood_group'] == 'Manhattan').astype(int)

# Mostrar algunas filas para verificar las nuevas variables

df.head()


# Variables categóricas que queremos convertir a numéricas
columnas_categoricas = ['neighbourhood_group', 'neighbourhood', 'room_type']

# Aplicamos One-Hot Encoding a estas columnas
df = pd.get_dummies(df, columns=columnas_categoricas, drop_first=True)

# Mostrar columnas nuevas para confirmar la conversión
print("Nuevas columnas generadas tras One-Hot Encoding:")
print(df.columns)




import seaborn as sns
import matplotlib.pyplot as plt
# Filtramos las columnas que nos interesan para visualizar correlaciones
columnas_interes = [
    'price', 'minimum_nights', 'availability_365', 'number_of_reviews', 
    'calculated_host_listings_count', 'room_type_Private room', 
    'room_type_Shared room'
]
# Agregar columnas que contienen 'neighbourhood_group' porque son más generales que 'neighbourhood'
columnas_interes += [col for col in df.columns if 'neighbourhood_group' in col]
# Crear un nuevo DataFrame con las columnas seleccionadas
df_reducido = df[columnas_interes]

# Generar el mapa de calor
plt.figure(figsize=(12, 8))
sns.heatmap(df_reducido.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación Reducida')
plt.show()



# Crear nuevas características
df['precio_por_noche'] = df['price'] / df['minimum_nights']
df['precio_por_dia_disponible'] = df['price'] / (df['availability_365'] + 1)  # +1 para evitar divisiones por cero

# Crear interacciones entre 'room_type' y 'neighbourhood_group'
df['Manhattan_Private_Room'] = df['neighbourhood_group_Manhattan'] * df['room_type_Private room']
df['Manhattan_Shared_Room'] = df['neighbourhood_group_Manhattan'] * df['room_type_Shared room']
df['Brooklyn_Private_Room'] = df['neighbourhood_group_Brooklyn'] * df['room_type_Private room']
df['Brooklyn_Shared_Room'] = df['neighbourhood_group_Brooklyn'] * df['room_type_Shared room']

# Confirmar que las nuevas columnas se han añadido correctamente
print("Nuevas columnas agregadas:", [col for col in df.columns if 'precio' in col or 'Room' in col])



import seaborn as sns
import matplotlib.pyplot as plt

# Seleccionar solo las columnas que son numéricas y que no incluyen vecindarios específicos
columnas_interes = [
    'price', 'minimum_nights', 'availability_365', 'number_of_reviews',
    'calculated_host_listings_count', 'room_type_Private room', 
    'room_type_Shared room', 'neighbourhood_group_Brooklyn', 
    'neighbourhood_group_Manhattan', 'neighbourhood_group_Queens', 
    'neighbourhood_group_Staten Island', 'precio_por_noche', 'precio_por_dia_disponible'
]

df_reducido = df[columnas_interes]

# Generar la nueva matriz de correlación
matriz_correlacion_reducida = df_reducido.corr()

# Crear un heatmap con las columnas seleccionadas
plt.figure(figsize=(12, 10))
sns.heatmap(matriz_correlacion_reducida, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación Reducida - Variables Relevantes')
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separar las características (X) de la variable objetivo (y)
X = df.drop(['price'], axis=1)
y = df['price']

# División en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado de características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Mostrar resultados de las dimensiones
print(f"X_train escalado - Forma: {X_train_scaled.shape}")
print(f"X_test escalado - Forma: {X_test_scaled.shape}")
print(f"y_train - Forma: {y_train.shape}")
print(f"y_test - Forma: {y_test.shape}")



from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd

# Obtener las columnas originales antes del escalado
columnas_originales = X_train.columns  # <- Este debe ser tu X_train original antes de escalarlo

# Selección de las mejores características (Top K)
selector = SelectKBest(score_func=f_regression, k=20)  # Selección de las 20 mejores características

# Ajustar el selector con X_train_scaled y y_train
X_train_selected = selector.fit_transform(X_train_scaled, y_train)

# Aplicar la transformación a X_test_scaled
X_test_selected = selector.transform(X_test_scaled)

# Extraer las columnas seleccionadas
columnas_seleccionadas = columnas_originales[selector.get_support()]

# Convertir a DataFrame para visualizarlo mejor
X_train_selected = pd.DataFrame(X_train_selected, columns=columnas_seleccionadas)
X_test_selected = pd.DataFrame(X_test_selected, columns=columnas_seleccionadas)

# Mostrar resultados tras la selección
print(f"X_train seleccionado - Forma: {X_train_selected.shape}")
print(f"X_test seleccionado - Forma: {X_test_selected.shape}")
print(f"Columnas seleccionadas: {columnas_seleccionadas.tolist()}")



from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Inicializar el modelo
modelo_lr = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
modelo_lr.fit(X_train_selected, y_train)

# Realizar predicciones en el conjunto de entrenamiento y prueba
y_train_pred = modelo_lr.predict(X_train_selected)
y_test_pred = modelo_lr.predict(X_test_selected)

# Calcular métricas de rendimiento
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

# Imprimir resultados
print("📊 Rendimiento del Modelo - Regresión Lineal:")
print(f"Error Cuadrático Medio (MSE) en Train: {mse_train:.2f}")
print(f"Error Cuadrático Medio (MSE) en Test: {mse_test:.2f}")
print(f"Error Absoluto Medio (MAE) en Test: {mae_test:.2f}")
print(f"Coeficiente de Determinación (R2 Score) en Test: {r2_test:.2f}")


