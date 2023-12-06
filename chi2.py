# Importar bibliotecas
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif


data = pd.read_csv("./datasetPreprocesado/lymph.csv")
X = data.drop('class', axis=1)
y = data['class']


selector = SelectKBest(score_func=chi2, k=2)

X_new = selector.fit_transform(X, y)

selected_features = selector.get_support()


selected_feature_names = [feature for feature, selected in zip(
    X.columns, selected_features) if selected]


total_score = sum(selector.scores_)


normalized_scores = [score / total_score for score in selector.scores_]

normalized_scores_rounded = [round(score, 10) for score in normalized_scores]

print("Características seleccionadas:", selected_feature_names)


print("Scores:", selector.scores_)


print("Características seleccionadas:", X_new)


print("Puntuacion normalizada:", normalized_scores)

print("Puntuacion normalizada a 10 decimales:", normalized_scores_rounded)
