
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_graphviz
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import shap
import graphviz
import scipy.stats as stats
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from IPython.display import display
from sklearn.base import clone


def afficher_voisins(modele, X_train, y_train, X_test, y_test, logement=0):

    preprocessor = modele.named_steps["preprocessing"]

    knn = modele.named_steps["regression"]

    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    distances, indices = knn.kneighbors(
        X_test_trans[logement].reshape(1,-1)
    )

    indice_test = X_test.index[logement]

    print("="*70)
    print(f"Logement étudié : {indice_test}")
    print("="*70)

    display(X_test.iloc[[logement]])

    print()

    print("Voisins les plus proches")

    voisins = pd.DataFrame({
        "Indice": X_train.index[indices[0]],
        "Distance": distances[0],
        "Prix": y_train.iloc[indices[0]].values
    })

    display(voisins)

    prediction = modele.predict(
        X_test.iloc[[logement]]
    )[0]

    print()

    print(f"Prix réel     : {y_test.iloc[logement]:,.0f} €")
    print(f"Prix prédit   : {prediction:,.0f} €")

    print("\nCaractéristiques des voisins :")

    display(
        X_train.iloc[indices[0]]
    )

    plt.figure(figsize=(6,4))

    plt.bar(
        range(1, len(distances[0])+1),
        distances[0]
    )

    plt.xlabel("Voisin")
    plt.ylabel("Distance")
    plt.title("Distance des K plus proches voisins")

    plt.grid(True)

    plt.show()



def afficher_predictions(y_test, y_pred):

    plt.figure(figsize=(6,6))

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.7
    )

    # droite idéale
    mini = min(y_test.min(), y_pred.min())
    maxi = max(y_test.max(), y_pred.max())

    plt.plot(
        [mini, maxi],
        [mini, maxi],
        "r--",
        label="Prédiction parfaite"
    )

    plt.xlabel("Prix réel (€)")
    plt.ylabel("Prix prédit (€)")
    plt.title("Valeurs réelles vs prédictions")

    plt.legend()
    plt.grid(True)

    plt.show()


def titre(txt):
    largeur = 90
    print()
    print("=" * largeur)
    print(txt.center(largeur))
    print("=" * largeur)


def dependence_plot(shap_values, interaction_values, X_test_trans, feature):

    # Indice de la variable
    i = list(X_test_trans.columns).index(feature)

    # Recherche de la variable la plus en interaction
    interaction_strength = np.abs(
        interaction_values[:, i, :]
    ).mean(axis=0)

    # On ignore l'interaction avec elle-même
    interaction_strength[i] = -1

    j = np.argmax(interaction_strength)

    print(f"Interaction la plus forte avec {feature} : "
          f"{X_test_trans.columns[j]}")

    shap.plots.scatter(
        shap_values[:, feature],
        color=shap_values[:, X_test_trans.columns[j]]
    )


def afficher_shap(modele, X_test, y_test, logement=0):


    indice_original = X_test.index[logement]

    titre(f"Explication du logement n°{indice_original}")

    display(X_test.iloc[[logement]])

    prediction = modele.predict(X_test.iloc[[logement]])[0]

    print(f"Prix prédit : {prediction:,.0f} €")

    print(f"Prix réel : {y_test.iloc[logement]:,.0f} €")
    print(f"Erreur : {prediction - y_test.iloc[logement]:,.0f} €")


    # ==========================================================
    # Préparation des données
    # ==========================================================

    preprocessor = modele.named_steps["preprocessing"]
    modele_tree = modele.named_steps["Tree_method"]

    X_test_trans = pd.DataFrame(
        preprocessor.transform(X_test),
        columns=preprocessor.get_feature_names_out(),
        index=X_test.index
    )

    # ==========================================================
    # Calcul des valeurs SHAP
    # ==========================================================

    explainer = shap.TreeExplainer(modele_tree)

    shap_values = explainer(X_test_trans)

    interaction_values = explainer.shap_interaction_values(
        X_test_trans
    )

    # nécessaire pour les notebooks
    shap.initjs()

    # ==========================================================
    # Importance globale
    # ==========================================================

    titre("SHAP Beeswarm")
    shap.plots.beeswarm(shap_values)

    # ==========================================================
    # Importance moyenne
    # ==========================================================

    titre("SHAP Bar")
    shap.plots.bar(shap_values)

    # ==========================================================
    # Explication locale
    # ==========================================================

    titre("Waterfall")
    shap.plots.waterfall(shap_values[logement])

    # ==========================================================
    # Force Plot
    # ==========================================================

    titre("Force Plot")

    shap.force_plot(
        base_value=shap_values.base_values[logement],
        shap_values=shap_values.values[logement],
        features=X_test_trans.iloc[logement],
        matplotlib=True,
        show=True
    )

    # ==========================================================
    # Dépendance
    # ==========================================================

    titre("Dépendance Surface")
    dependence_plot(
        shap_values,
        interaction_values,
        X_test_trans,
        "num__Surface"
    )

    titre("Dépendance DistanceCentre")
    dependence_plot(
        shap_values,
        interaction_values,
        X_test_trans,
        "num__DistanceCentre"
    )

    # ==========================================================
    # Decision Plot
    # ==========================================================

    titre("Decision Plot")

    shap.decision_plot(
        base_value=shap_values.base_values[0],
        shap_values=shap_values.values[:20],
        features=X_test_trans.iloc[:20],
        feature_names=list(X_test_trans.columns)
    )

    # ==========================================================
    # Heatmap des interactions
    # ==========================================================

    titre("Heatmap des interactions")

    interaction_mean = np.abs(interaction_values).mean(axis=0)

    np.fill_diagonal(interaction_mean, 0)

    plt.figure(figsize=(8,8))

    plt.imshow(interaction_mean)

    plt.xticks(
        range(len(X_test_trans.columns)),
        X_test_trans.columns,
        rotation=90
    )

    plt.yticks(
        range(len(X_test_trans.columns)),
        X_test_trans.columns
    )

    plt.colorbar(label="Interaction SHAP moyenne")

    plt.tight_layout()

    plt.show()

    # ==========================================================
    # Tableau des interactions
    # ==========================================================

    interactions = []

    noms = X_test_trans.columns

    for i in range(len(noms)):
        for j in range(i+1, len(noms)):

            interactions.append(
                (
                    noms[i],
                    noms[j],
                    np.abs(interaction_values[:, i, j]).mean()
                )
            )

    interactions = sorted(
        interactions,
        key=lambda x: x[2],
        reverse=True
    )

    df = pd.DataFrame(
        interactions,
        columns=[
            "Variable 1",
            "Variable 2",
            "Interaction moyenne"
        ]
    )

    display(df.head(15))

def afficher_permutation_importance(modele, X_test, y_test):

    r = permutation_importance(
        modele,
        X_test,
        y_test,
        n_repeats=20,
        random_state=42
    )

    importance = pd.DataFrame({
        "Variable": X_test.columns,
        "Importance": r.importances_mean,
        "Ecart-type": r.importances_std
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    display(importance)

    importance.plot.bar(x="Variable", y="Importance")
    plt.yscale("symlog", linthresh=1e-4)
    plt.ylim(r.importances_mean.min(), r.importances_mean.max())
    plt.show()

def evolution_parametre(modele, parametre, valeurs, X_train, y_train, X_test, y_test):

    rmse_train = []
    rmse_test = []
    r2_train = []
    r2_test = []

    for valeur in valeurs:

        m = clone(modele)

        m.set_params(**{parametre: valeur})

        m.fit(X_train, y_train)

        y_train_pred = m.predict(X_train)
        y_test_pred = m.predict(X_test)

        rmse_train.append(root_mean_squared_error(y_train, y_train_pred))
        rmse_test.append(root_mean_squared_error(y_test, y_test_pred))

        r2_train.append(r2_score(y_train, y_train_pred))
        r2_test.append(r2_score(y_test, y_test_pred))

    plt.figure(figsize=(7,4))
    plt.plot(valeurs, rmse_train, label="Train")
    plt.plot(valeurs, rmse_test, label="Test")
    plt.xlabel(parametre)
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(7,4))
    plt.plot(valeurs, r2_train, label="Train")
    plt.plot(valeurs, r2_test, label="Test")
    plt.xlabel(parametre)
    plt.ylabel("$R^2$")
    plt.legend()
    plt.grid()
    plt.show()


def creation_arbre_decision_pdf(modele):

    dot = export_graphviz(
                modele.named_steps["Tree_method"],
                out_file=None,
                feature_names=modele.named_steps["preprocessing"].get_feature_names_out(),
                filled=True,
                rounded=True
            )

    graph = graphviz.Source(dot)

    graph.render("arbre_decision")

def afficher_feature_importances(modele):

    coef_df = pd.DataFrame({
        "Variable": modele.named_steps["preprocessing"].get_feature_names_out(),
        "Coefficient": modele.named_steps["Tree_method"].feature_importances_
    })

    return coef_df.sort_values(by="Coefficient", ascending=False)

def afficher_residus(y_test, y_pred):

    residus = y_test - y_pred

    print(f"Moyenne : {residus.mean():.2f}")
    print(f"Écart-type : {residus.std():.2f}")
    print(f"Minimum : {residus.min():.2f}")
    print(f"Maximum : {residus.max():.2f}")

    plt.figure(figsize=(6,5))
    plt.scatter(y_pred, residus, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Prix prédit")
    plt.ylabel("Résidus")
    plt.title("Résidus en fonction des prédictions")
    plt.show()


    plt.figure(figsize=(6,5))
    plt.hist(residus, bins=20)
    plt.xlabel("Résidus")
    plt.ylabel("Effectif")
    plt.title("Distribution des résidus")
    plt.show()


    

def afficher_coefficients(modele):

    # ============================
    # Récupération du pipeline
    # ============================

    preprocessor = modele.named_steps["preprocessing"]
    regression = modele.named_steps["regression"]

    variables = preprocessor.get_feature_names_out()
    coef_std = regression.coef_


    # ============================
    # Récupération du scaler
    # ============================

    scaler = (preprocessor.named_transformers_["num"].named_steps["scaler"])

    ecarts_type = scaler.scale_
    moyennes = scaler.mean_


    # ============================
    # Construction du tableau
    # ============================

    coef_interpretables = []
    unites = []

    colonnes_num = None

    for nom, transfo, colonnes in preprocessor.transformers_:
        if nom == "num":
            colonnes_num = colonnes
            break

    names = []        
    
    for variable, coef in zip(variables, coef_std):

        # Variables numériques
        if variable.startswith("num__"):

            nom = variable.replace("num__", "")

            names.append(nom)

            indice = colonnes_num.get_loc(nom)

            coef_original = coef / ecarts_type[indice]

            coef_interpretables.append(coef_original)

            if nom == "Surface":
                unites.append("€/m²")

            elif nom == "Chambres":
                unites.append("€/chambre")

            elif nom == "SallesDeBain":
                unites.append("€/salle")

            elif nom == "DistanceCentre":
                unites.append("€/km")

            elif nom == "AnneeConstruction":
                unites.append("€/an")

            else:
                unites.append("")

        # Variables catégorielles
        else:

            nom = variable.replace("cat__", "")

            names.append(nom)

            coef_interpretables.append(coef)

            unites.append("€")


    coef_generation = [3200, 8000, 12000, 500, -2500, 80000, 20000, 180000, 40000, 20000, 30000, 15000, 18000, -930000]
    names.append("Intercept")
    coef_std = np.append(coef_std, regression.intercept_)

    intercept = regression.intercept_

    for coef, mu, sigma in zip(coef_std[:len(colonnes_num)], moyennes, ecarts_type):
        intercept -= coef * mu / sigma

    coef_interpretables.append(intercept)
    unites.append("€")

    df = pd.DataFrame({
        "Variable": names,
        "Coef. standardisé": coef_std,
        "Coef. interprétable": coef_interpretables,
        "Coef. génération": coef_generation,
        "Unité": unites
    })

    display(df)  

    names.pop()
    coef = pd.Series(
        modele.named_steps["regression"].coef_,
        index=names
    )

    coef.sort_values().plot.barh(figsize=(8,5))
    plt.title("Coefficients du modèle")
    plt.show()


def generer_donnees():

    # Chargement des données

    df = pd.read_csv("projet_prix_maisons.csv")

    # On isole la variable cible

    X = df.drop(columns="Prix")
    y = df["Prix"]   

    return df, X, y


def resume_modele(estimator):
    print("Modèle :", estimator.__class__.__name__)

    for nom, valeur in estimator.get_params().items():
        print(f"{nom:20s}: {valeur}")


def optimize_model(preprocessor, estimator, param_grid, X_train, X_test, y_train, y_test):
    model = Pipeline([
        ("preprocessing", preprocessor),
        ("regression", estimator)
    ])

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Meilleurs paramètres :", grid.best_params_)
    print("RMSE CV :", -grid.best_score_)

    return grid.best_estimator_, pd.DataFrame(grid.cv_results_)


def regression_metrics(y_test, y_pred, verbose=True):
    # (Mean Absolute Error) est 1/n Sum |y_i-\bar{y_i}| où \bar{y_i} est la valeur prédite
    # En moyenne, le modèle se trompe de MAE euros.

    mae = mean_absolute_error(y_test, y_pred)


    # RMSE (Root Mean Squared Error) est sqrt{1/n Sum (y_i-\bar{y_i})^2}
    # Le RMSE pénalise davantage les grosses erreurs que le MAE.

    rmse = root_mean_squared_error(y_test, y_pred)

    # R^2 score
    # {\displaystyle R^{2}=1-{\dfrac {\sum _{i=1}^{n}\left(y_{i}-{\hat {y_{i}}}\right)^{2}}{\sum _{i=1}^{n}\left(y_{i}-{\bar {y}}\right)^{2}}}} où hat valeur prédite et bar moyenne

    r2 = r2_score(y_test, y_pred)


    if verbose:
        print(f"MAE  : {mae:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print(f"R²   : {r2:.4f}")


    return {
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
    }


def train_model(preprocessor, model, X_train, X_test, y_train):

    # On ajoute le modèle
    modele = Pipeline([
    ("preprocessing", preprocessor),
    ("regression", model)
    ])
    
    # X_train_trans = modele.named_steps["preprocessing"].fit_transform(X_train)

    modele.fit(X_train, y_train)

    # Prédiction

    return modele.predict(X_test)



def pipeline_regression(colonnes_num, colonnes_cat):

    # Pipeline pour les modèles linéaires

    # Construction du pipeline numérique 
    # On remplit les varleurs manquantes par la valeur médiane  
    # On renormalise les données avec le StandardScaler qui remplace par le z-score
    # On impute jamais la variable cible (on préfère même supprimer les lignes correspondante avec df = df.dropna(subset=["Prix"]))
    # On peut laisser le prix en euros mais pour (réseaux de neurones, certaines méthodes d'optimisation), on standardise aussi la cible.

    pipeline_num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Construction du pipeline catégoritielle 
    # On remplit avec l'occurence la plus fréquente (On aurait pu faire un remplissage aléatoire avec les probas observés)
    # On encode les variables catégoritielles avec des tableaux One-Hot
    # handle_unknown="ignore" permet de ne pas avoir d'erreur si il y a une catégorie manquante dans le train (exemple la ville Bordeaux n'apparait pas dans le train)
    # Lorsqu'une catégorie inconnue est rencontrée lors de la transformation, les colonnes obtenues par codage « one-hot » pour cette caractéristique seront toutes composées de zéros. Lors de la transformation inverse, une catégorie inconnue sera désignée par « None ».
    # drop="first" est un bon choix pour la régression linéaire et Ridge (évite la multicolinéarité parfaite)

    pipeline_cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    # On fusionne les deux pipelines
    # Elle signifie :
    # - applique pipeline_num aux colonnes numériques ;
    # - applique pipeline_cat aux colonnes catégorielles ;
    # - rassemble ensuite les résultats.

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", pipeline_num, colonnes_num),
            ("cat", pipeline_cat, colonnes_cat)
        ]
    )

    return preprocessor

def pipeline_arbre(colonnes_num, colonnes_cat):

# Pipeline pour les arbres

    pipeline_num = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    pipeline_cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", pipeline_num, colonnes_num),
        ("cat", pipeline_cat, colonnes_cat)
    ])


    return preprocessor

"""
param_grid = {
    "regression__kernel": ["rbf", "linear"],
    "regression__C": [100000],
    "regression__epsilon": [0.01, 0.1, 1],
    "regression__gamma": ["scale", 0.01, 0.1, 1]
}


optimize_model(SVR(), param_grid)


##############################################
# Validation croisée (Cross Validation)
##############################################

Principe
---------

Au lieu d'effectuer un unique découpage
Entraînement / Test, on répète plusieurs
expériences sur différents découpages des données.

Le but est d'obtenir une estimation plus robuste
des performances du modèle.

K-Fold
------

Les données sont découpées en K parties
(en général K = 5 ou 10).

Pour chaque itération :

    un fold sert de jeu de test ;

    les K-1 autres servent à entraîner
    le modèle.

Le processus est répété K fois de sorte que
chaque observation soit utilisée une fois
comme donnée de test.

On calcule ensuite la moyenne des métriques.

Pseudo-code

Découper les données en K folds

pour i = 1 ... K

    entraîner sur K-1 folds

    tester sur le fold restant

    calculer la métrique

Retourner la moyenne et l'écart-type.

Avantages
----------

Utilise toutes les données.

Estimation plus stable qu'un unique train/test.

Permet une comparaison plus fiable des modèles.

Inconvénients
-------------

Plus coûteux en temps de calcul.

Un modèle est entraîné K fois.

Mesures obtenues
----------------

Pour chaque métrique (RMSE, MAE, R²)

on obtient :

moyenne

écart-type

Un faible écart-type signifie que le modèle est
stable quel que soit le découpage des données.

for mtd in [LinearRegression(), ridge, lasso, elastic, svr]:
    modele = Pipeline([
        ("preprocessing", preprocessor),
        ("regression", mtd)
        ])

    scores = cross_validate(
            modele,
                X,
                y,
                cv=5,
                scoring=[
                    "neg_mean_absolute_error",
                    "neg_root_mean_squared_error",
                    "r2"
                ]
            )
        
    print("MAE moyen :", -scores["test_neg_mean_absolute_error"].mean())
    print("RMSE moyen :", -scores["test_neg_root_mean_squared_error"].mean())
    print("R² moyen :", scores["test_r2"].mean())
    print(scores["test_r2"].std(ddof=1)) """



###########################################################################################################################################################################

"""
for mtd in [rf, gbr, xgb, cat]:
    modele = Pipeline([
        ("preprocessing", preprocessor),
        ("regression", mtd)
        ])

    scores = cross_validate(
            modele,
                X,
                y,
                cv=5,
                scoring=[
                    "neg_mean_absolute_error",
                    "neg_root_mean_squared_error",
                    "r2"
                ]
            )
        
    print("MAE moyen :", -scores["test_neg_mean_absolute_error"].mean())
    print("RMSE moyen :", -scores["test_neg_root_mean_squared_error"].mean())
    print("R² moyen :", scores["test_r2"].mean())
    print(scores["test_r2"].std(ddof=1))

    
resultats = {}
print(pd.DataFrame(resultats).T)"""
