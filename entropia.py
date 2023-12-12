import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

# Cargar el conjunto de datos del Titanic
titanic = pd.read_csv('./dataset/iris.csv')

# Separar las características (X) y la variable objetivo (y)
X, y = titanic.drop('class', axis=1), titanic['class']


# Crear un objeto DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)

# Ajustar el clasificador a nuestros datos. Esto calculará la importancia de las características utilizando el índice Gini.
clf.fit(X, y)

# Obtener la importancia de las características
feature_importances = clf.feature_importances_

# Normalizar cada importancia dividiéndola por la suma total
normalized_importances = [
    importance / sum(feature_importances) for importance in feature_importances]

# Crear un DataFrame para almacenar las características y sus importancias
features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': clf.feature_importances_
})

# Ordenar el DataFrame por importancia
features_df = features_df.sort_values(by='Importance', ascending=False)

# Seleccionar las dos características más importantes
top_features = features_df.iloc[:2]

print("Las dos características más importantes son:")
print(top_features)
print("Las caracteristicas normalizadas: ", normalized_importances)


for feature_name, importance, normalized in zip(X.columns, feature_importances, normalized_importances):
    print(
        f"Característica: {feature_name}, Importancia: {importance}, Importancia normalizada: {normalized}")
