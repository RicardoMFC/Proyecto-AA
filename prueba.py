import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# 1. Cargar el archivo ARFF
data, meta = arff.loadarff("Sapfile1.arff")
df = pd.DataFrame(data)

# 2. Decodificar bytes a strings
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.decode("utf-8")

# 3. Limpieza básica
df_modificado = df.copy()

# Eliminar columnas sin variabilidad
columnas_sin_variabilidad = df_modificado.columns[df_modificado.nunique() == 1].tolist()
df_modificado.drop(columns=columnas_sin_variabilidad, inplace=True)

# Eliminar columna 'ge' si existe
if 'ge' in df_modificado.columns:
    df_modificado.drop(columns=['ge'], inplace=True)

# Agrupar categorías raras en 'cst'
frecuencias_cst = df_modificado['cst'].value_counts()
categorias_comunes = frecuencias_cst[frecuencias_cst >= 10].index.tolist()
df_modificado['cst'] = df_modificado['cst'].apply(lambda x: x if x in categorias_comunes else 'Other')

# 4. Codificación
df_encoded = df_modificado.apply(LabelEncoder().fit_transform)
X = df_encoded.drop(columns=['esp'])
y = df_encoded['esp']

# 5. División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# 6. Aplicar SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 7. Definir modelos
modelos_res = {
    "Regresión Logística": LogisticRegression(max_iter=1000),
    "SVM (RBF)": SVC(kernel='rbf', probability=True),
    "Árbol de Decisión": DecisionTreeClassifier(),
    "Red Neuronal (MLP)": MLPClassifier(max_iter=1000),
    "k-Vecinos más Cercanos": KNeighborsClassifier()
}

# 8. Entrenar y evaluar
resultados_res = {}
for nombre, modelo in modelos_res.items():
    modelo.fit(X_train_res, y_train_res)
    pred = modelo.predict(X_test)
    resultados_res[nombre] = {
        "Accuracy": accuracy_score(y_test, pred),
        "Reporte": classification_report(y_test, pred)
    }

# 9. Mostrar resultados
print("=== Comparación de Accuracy con SMOTE ===")
for nombre, resultado in resultados_res.items():
    print(f"\n{nombre}:")
    print("Accuracy:", round(resultado["Accuracy"], 3))
    print("Reporte de clasificación:")
    print(resultado["Reporte"])
