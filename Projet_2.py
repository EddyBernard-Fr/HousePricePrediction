
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

    """

    reg = modele.named_steps["regression"]
    coef = reg.coef_


    coef_df = pd.DataFrame({
            "Variable": noms,
            "Coefficient standardisé": coef,
            "Coefficient interprétable": coef_originaux
        })

    coef_df["Abs"] = coef_df["Coefficient"].abs()

    coef_df = coef_df.sort_values(
            "Abs",
            ascending=False
        )
    

    print(coef_df)
    print(f"Valeur à l'origine: {reg.intercept_}")"""

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

    # On récupère les colonnes numériques d'une part, et catégoritielle d'autre part (sans le prix)

    """colonnes_num = X.select_dtypes(include="number").columns
    colonnes_cat = X.select_dtypes(include="str").columns

    # Séparation train/test avec 20% des données qui sont gardées pour le test

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )"""

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

    """residus = y_test - y_pred
    #plt.scatter(y_pred, residus)
    #plt.axhline(0, color="red")
    #plt.xlabel("Prix prédit")
    #plt.ylabel("Résidu")
    plt.hist(residus, bins=20)

    plt.xlabel("Résidu")
    plt.ylabel("Nombre")
    plt.title("Distribution des résidus")

    plt.show()

    stats.probplot(residus, dist="norm", plot=plt)

    plt.show()

    plt.scatter(
        X_test["DistanceCentre"],
        residus
        )
    plt.show()

    plt.scatter(
        y_pred,
        abs(residus)
    )

    plt.xlabel("Prix prédit")
    plt.ylabel("|Résidu|")
    plt.show()

    # En moyenne, le modèle commet une erreur d'environ x % sur le prix des logements.
    erreur = 100 * residus / y_test
    plt.hist(erreur, bins=20)
    plt.show()"""

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

    """# Coefficient
    if numero == 0:
        noms = modele.named_steps["preprocessing"].get_feature_names_out() # On récupère le nom des colonne après transformation

        reg = modele.named_steps["regression"]
        coef = reg.coef_

        coef_df = pd.DataFrame({
            "Variable": noms,
            "Coefficient": coef
        })

        coef_df["Abs"] = coef_df["Coefficient"].abs()

        coef_df = coef_df.sort_values(
            "Abs",
            ascending=False
        )

        print(coef_df)
        print(f"Valeur à l'origine: {reg.intercept_}")
    
    elif model.__class__.__name__ == DecisionTreeRegressor:
        noms = modele.named_steps["preprocessing"].get_feature_names_out()

        coef_df = pd.DataFrame({
            "Variable": noms,
            "Coefficient": model.feature_importances_
        })

        print(coef_df)

        # À chaque nœud, l'arbre choisit une variable qui diminue l'erreur sur le prix.
        # Il calcule le gain obtenu grâce à cette séparation.
        # Pour un arbre de régression, le critère est la diminution de variance (ou de manière équivalente la diminution de la somme des carrés des erreurs à la moyenne du noeud).
        # A la fin on addtionne tous les gains pour chaque variable, puis on normalise et ça donne arbre.feature_importances_
        # Si coef = 0, la variable n'a jamais été utilisé pour séparer les noeuds
        # Si coef proche de 1, la varaible a beaucoup contribué au gain.

        if model.max_depth == 3:
            plt.figure(figsize=(18,10))

            plot_tree(
                model,
                feature_names=noms,
                filled=True,
                rounded=True,
                fontsize=8
            )

            plt.show()

        if model.max_depth == 6:

            dot = export_graphviz(
                model,
                out_file=None,
                feature_names=noms,
                filled=True,
                rounded=True
            )

            graph = graphviz.Source(dot)

            graph.render("arbre_decision")

    elif numero == 2:
        noms = modele.named_steps["preprocessing"].get_feature_names_out() 
     
        rf = modele.named_steps["regression"]


        coef = rf.feature_importances_
        coef = coef / coef.sum()

        coef_df = pd.DataFrame({
                "Variable": noms,
                "Importance": coef
            }).sort_values("Importance", ascending=False)

        print(coef_df)

        

# =============================================================================
# SHAP (SHapley Additive exPlanations)
# =============================================================================
#
# SHAP est une méthode d'interprétation issue de la théorie des jeux.
#
# Chaque variable est considérée comme un joueur qui contribue à la
# prédiction finale.
#
# La valeur SHAP mesure la contribution exacte d'une variable à une
# prédiction donnée.
#
# Pour une observation :
#
#     prédiction =
#         valeur moyenne du modèle
#       + somme des valeurs SHAP
#
# Chaque variable peut :
#
#     - augmenter la prédiction (SHAP positif)
#     - diminuer la prédiction (SHAP négatif)
#
# SHAP permet donc d'expliquer individuellement chaque prédiction.
#
# Contrairement aux feature_importances_, SHAP indique également le sens
# de l'influence.
#
# Les principaux graphiques sont :
#
# summary_plot :
#     importance globale des variables.
#
# beeswarm :
#     distribution des contributions pour toutes les observations.
#
# waterfall_plot :
#     explication détaillée d'une observation.
#
# force_plot :
#     visualisation des contributions positives et négatives.
#
# dependence_plot :
#     effet d'une variable selon sa valeur et ses interactions.
#
# Avantages :
#
# - interprétation locale et globale ;
# - indique le sens de l'effet ;
# - très utilisé en recherche et en industrie ;
# - particulièrement adapté aux modèles d'arbres.
#
# Inconvénients :
#
# - calcul parfois coûteux ;
# - interprétation plus complexe ;
# - nécessite une bibliothèque supplémentaire.
#
# Idée fondamentale :
#
# La prédiction est répartie équitablement entre toutes les variables,
# chacune recevant une contribution appelée valeur SHAP.
#
# =============================================================================

# =============================================================================
# Calcul des valeurs SHAP (SHapley Additive exPlanations)
# =============================================================================
#
# Les valeurs SHAP proviennent de la théorie des jeux coopératifs.
#
# Les variables explicatives sont vues comme des joueurs qui coopèrent afin
# d'obtenir une prédiction. On cherche à répartir équitablement la prédiction
# finale entre toutes les variables.
#
# Soit :
#
#   N : ensemble des variables explicatives ;
#   i : une variable donnée ;
#   S : un sous-ensemble de variables ne contenant pas i ;
#   f(S) : prédiction obtenue en utilisant uniquement les variables de S.
#
# Pour chaque sous-ensemble S, on mesure la contribution marginale de la
# variable i :
#
#       f(S ∪ {i}) - f(S)
#
# Cette contribution dépend du contexte (c'est-à-dire des autres variables
# déjà présentes). Afin d'obtenir une contribution unique, on calcule une
# moyenne pondérée sur tous les sous-ensembles possibles.
#
# La valeur de Shapley est donnée par
#
#                    |S|! (|N|-|S|-1)!
# φ_i = Σ -------------------------------- [f(S∪{i}) - f(S)]
#       S inclu N privé de {i}            |N|!
#
# où
#
#     |S| est le nombre de variables dans S ;
#     |N| est le nombre total de variables.
#
# Les coefficients
#
#       |S|!(|N|-|S|-1)! / |N|!
#
# correspondent à la probabilité qu'un sous-ensemble apparaisse dans une
# permutation aléatoire des variables. Ils garantissent une répartition
# équitable des contributions.
#
# Les valeurs SHAP possèdent plusieurs propriétés importantes :
#
# - Additivité :
#
#       prédiction =
#           valeur moyenne du modèle
#         + somme des valeurs SHAP.
#
# - Symétrie :
#
#       deux variables ayant exactement le même rôle reçoivent la même
#       contribution.
#
# - Variable inutile :
#
#       une variable n'ayant aucune influence possède une valeur SHAP nulle.
#
# - Efficacité :
#
#       toute la prédiction est répartie entre les variables.
#
# =============================================================================

# =============================================================================
# TreeSHAP
# =============================================================================
#
# Calculer directement les valeurs de Shapley nécessite d'examiner tous les
# sous-ensembles de variables.
#
# Avec p variables, cela représente :
#
#       2^p sous-ensembles.
#
# Le coût devient donc exponentiel et rapidement impossible en pratique.
#
# Les modèles d'arbres (Decision Tree, Random Forest, Gradient Boosting,
# XGBoost, LightGBM...) possèdent cependant une structure particulière.
#
# TreeSHAP exploite cette structure afin de calculer exactement les valeurs
# SHAP sans énumérer tous les sous-ensembles.
#
# Pour chaque arbre :
#
#   1) l'algorithme parcourt récursivement les chemins de la racine vers les
#      feuilles ;
#
#   2) il maintient les probabilités qu'une observation emprunte chacun des
#      chemins possibles lorsque certaines variables sont inconnues ;
#
#   3) lorsqu'une variable intervient dans une séparation, sa contribution
#      est propagée le long du chemin avec les poids de Shapley appropriés ;
#
#   4) arrivé dans une feuille, la valeur prédite est redistribuée entre les
#      variables ayant participé au chemin.
#
# Les contributions sont ensuite sommées sur tous les arbres de la forêt ou
# du modèle de boosting.
#
# TreeSHAP fournit exactement les mêmes valeurs que la définition théorique
# des valeurs de Shapley, mais avec un coût polynomial au lieu d'un coût
# exponentiel.
#
# C'est cette amélioration algorithmique qui rend SHAP utilisable sur des
# modèles comportant des centaines d'arbres.
#
# =============================================================================

# récupérer les données transformées

        X_train_trans = modele.named_steps["preprocessing"].transform(X_train)
        X_test_trans = pd.DataFrame(
            modele.named_steps["preprocessing"].transform(X_test),
            columns=noms,
            index=X_test.index
        )

        # créer l'explainer 
        explainer = shap.TreeExplainer(modele.named_steps["regression"])

        # calcul des valeurs SHAP       
        shap_values = explainer(X_test_trans)
            
        # Importance globale
        shap.plots.beeswarm(shap_values)

        # Importance moyenne
        shap.plots.bar(shap_values)

        # Waterfall pour le premier logement
        shap.plots.waterfall(shap_values[0])

        # Dépendance de Surface
        interaction_values = explainer.shap_interaction_values(X_test_trans)
        
        feature = "num__Surface"
        i = list(X_test_trans.columns).index(feature)

        interaction_strength = np.abs(interaction_values[:, i, :]).mean(axis=0)

        # on ignore l'interaction avec elle-même
        interaction_strength[i] = -1

        j = np.argmax(interaction_strength)

        shap.plots.scatter(shap_values[:, "num__Surface"], color=shap_values[:, X_test_trans.columns[j]])

        # Dépendance de DistanceCentre

        # On calcule la variable qui intéragit le plus avec num__DistanceCentre
        feature = "num__DistanceCentre"
        i = list(X_test_trans.columns).index(feature)

        interaction_strength = np.abs(interaction_values[:, i, :]).mean(axis=0)

        # on ignore l'interaction avec elle-même
        interaction_strength[i] = -1

        j = np.argmax(interaction_strength)
        shap.plots.scatter(shap_values[:, "num__DistanceCentre"], color=shap_values[:, X_test_trans.columns[j]])


        # Decision plot (Le decision plot montre comment chaque variable “construit” la prédiction en partant de la moyenne)

        shap.decision_plot(
            base_value=shap_values.base_values[0],
            shap_values=shap_values.values[:20],
            features=X_test_trans.iloc[:20],
            feature_names=list(X_test_trans.columns)
        )


        # heatmap des interractions

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

        # Affichage des plus forte interraction


        noms1 = X_test_trans.columns

        interactions = []

        for i in range(len(noms1)):
            for j in range(i+1, len(noms1)):
                interactions.append((
                    noms1[i],
                    noms1[j],
                    np.abs(interaction_values[:, i, j]).mean()
                ))

        interactions = sorted(
            interactions,
            key=lambda x: x[2],
            reverse=True
        )

        for a,b,v in interactions[:15]:
            print(a,b,v)
          
        LightGBM: Sert a voir le nombre de num_leaves utilisé pour le premier arbre
        print(rf.booster_.dump_model()["tree_info"][0]["num_leaves"])
        
                            
    elif numero == 3:

        if model.__class__.__name__ != SVR:

            noms = modele.named_steps["preprocessing"].get_feature_names_out()

            coef_df = pd.DataFrame({
                "Variable": noms,
                "Coefficient": modele.named_steps["regression"].feature_importances_
            })

            print(coef_df)

        

# =============================================================================
# Permutation Importance
# =============================================================================
#
# Les feature_importances_ d'un arbre ou d'une forêt sont calculées pendant
# l'apprentissage. Elles peuvent être biaisées (variables continues,
# variables ayant beaucoup de valeurs distinctes, variables corrélées...).
#
# La permutation importance est une méthode indépendante du modèle.
#
# Principe :
#
# 1) On mesure d'abord les performances du modèle sur le jeu de test
#    (R², RMSE, MAE...).
#
# 2) Pour chaque variable :
#
#       - on mélange aléatoirement uniquement cette colonne
#       - toutes les autres restent inchangées
#       - on recalcule les performances
#
# Si les performances chutent fortement, cela signifie que le modèle
# utilisait beaucoup cette variable.
#
# Si elles changent très peu, cette variable apporte peu d'information.
#
# L'opération est répétée plusieurs fois (n_repeats) afin de diminuer
# l'influence du hasard.
#
# importances_mean :
#     perte moyenne de performance.
#
# importances_std :
#     variabilité de cette perte.
#
# Avantages :
# - fonctionne avec presque tous les modèles ;
# - directement interprétable ;
# - moins biaisée que feature_importances_ ;
# - permet de détecter les variables réellement utiles.
#
# Inconvénients :
# - nécessite plusieurs prédictions ;
# - plus lente ;
# - les variables fortement corrélées peuvent sembler moins importantes
#   car elles se remplacent mutuellement.
#
# Idée fondamentale :
#
# Une variable est importante si le modèle devient nettement moins bon
# lorsqu'on détruit uniquement l'information contenue dans cette variable.
#
# =============================================================================

        r = permutation_importance(modele, X_test, y_test)
        print(r.importances_mean)
        print(r.importances_std)
        

        df_imp = pd.DataFrame({
            "variable": X_test.columns,
            "importance": r.importances_mean,
            "std": r.importances_std
        }).sort_values("importance", ascending=False)

        df_imp.plot.bar(x="variable", y="importance")
        plt.yscale("log")
        plt.ylim(0.00001, 10)
        plt.show()


    # Accuracy (Précision)

    return regression_metrics(y_test, y_pred)"""




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


"""print("=== Régression linéaire ===")

# la régression linéaire cherche les coeff beta qui minimise ∑​(yi - hat{yi}^2) (La méthode des moindres carrées est utilisé ici)

#resultats["Regression"] = train_model(LinearRegression(), 0)

print("=== Méthode Ridge ===")

# La méthode Ridge cherche les coeff qui minimise ∑​(yi-hat{yi}^2) + alpha x ∑ beta_j^2
# Solution explicite Beta = (X^tX+alpha I)^{-1}X^ty
# On cherche un compromis entre bien ajuster les données et garder des coefficients petits.
# L'erreur augmente légèrement sur l'entraînement, mais la généralisation est meilleure.
# Quand utiliser Ridge ? Quand
# - il y a beaucoup de variables ;
# - les variables sont corrélées ;
# - les coefficients deviennent instables ;
# - on souhaite diminuer le surapprentissage.


ridge = RidgeCV(
    alphas=[0.01, 0.1, 1, 10, 100],
    cv=5
)

resultats["Ridge"] = train_model(ridge, 0)

print(f"Pour la méthode Ridge, le meilleur coefficient alpha: {ridge.alpha_}")




print("=== Méthode Lasso ===")

# La méthode Lasso (Least Absolute Shrinkage and Selection Operator)
# cherche les coefficients β qui minimisent
#
#     Σ(yi - ŷi)² + alpha Σ |βj|
#
# On cherche un compromis entre :
# - bien ajuster les données ;
# - garder des coefficients petits.
#
# Contrairement à Ridge, la pénalisation porte sur la norme L1
# (somme des valeurs absolues des coefficients).
#
# Cette pénalisation peut rendre certains coefficients exactement
# égaux à zéro : Lasso effectue donc une sélection automatique
# des variables.
#
# Lorsque alpha = 0, on retrouve la régression linéaire.
# Plus alpha est grand, plus les coefficients sont pénalisés.
#
# On utilise principalement Lasso lorsque :
# - il existe beaucoup de variables ;
# - certaines variables sont peu informatives ;
# - on souhaite obtenir un modèle plus simple et plus interprétable.

# Lasso fait intervenir des fonctions convexes non différentiables. Pour le traiter rigoureusement, 
# on entre dans le domaine de l'analyse convexe, avec les sous-gradients, 
# les opérateurs proximaux et les méthodes d'optimisation itératives (Coordinate Descent, ISTA, FISTA, ADMM...)

lasso = LassoCV(
    alphas=[0.01, 0.1, 1, 10, 100],
    cv=5,
    random_state=42
)

resultats["Lasso"] = train_model(lasso, 0)

print(f"Pour la méthode Lasso, le meilleur coefficient alpha: {lasso.alpha_}")

print("=== Elastic Net ===")


# Elastic Net combine les pénalisations Ridge (L2) et Lasso (L1).
#
# Il minimise :
#
#     Σ(yi - ŷi)²
#     + alpha[(1-l1_ratio)/2 * Σβj² + l1_ratio * Σ|βj|]
#
# alpha contrôle l'intensité de la régularisation.
# l1_ratio contrôle le mélange entre Ridge et Lasso.
#
# l1_ratio = 0  -> Ridge
# l1_ratio = 1  -> Lasso
#
# Elastic Net est particulièrement adapté lorsque plusieurs variables
# sont fortement corrélées. Il permet de stabiliser les coefficients
# tout en pouvant effectuer une sélection de variables.
# La partie L1 (Lasso) permet de supprimer des variables.
# La partie L2 (Ridge) stabilise les coefficients lorsqu'il existe de fortes corrélations entre les variables.

elastic = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
    alphas=[0.01, 0.1, 1, 10, 100],
    cv=5,
    random_state=42
)

resultats["Elastic"] = train_model(elastic, 0)

print(f"Pour la méthode Elastic net, le meilleur coefficient alpha est {elastic.alpha_} et le meilleur l1_ratio est {elastic.l1_ratio_}")


print("=== K-Nearest Neighbors (KNN) ===")

# ============================================================================
# K-Nearest Neighbors (KNN)
# ============================================================================
#
# Le KNN est une méthode non paramétrique et paresseuse (lazy learning).
#
# Contrairement à la régression linéaire ou aux arbres, il n'apprend pas
# explicitement une fonction reliant les variables explicatives à la cible.
# Il mémorise simplement toutes les observations de l'ensemble d'entraînement.
#
# Pour prédire une nouvelle observation :
#
#    1. calculer sa distance à toutes les observations d'entraînement ;
#    2. sélectionner les k plus proches voisins ;
#    3. prédire la moyenne (ou moyenne pondérée) de leurs valeurs.
#
# Si les voisins sont (x_i,y_i), la prédiction est
#
#                   1
#   y_hat = --------------- Σ y_i
#             k
#
# Lorsque weights="distance", les voisins les plus proches ont davantage
# d'influence :
#
#            Σ w_i y_i
#   y_hat = ----------
#             Σ w_i
#
# où généralement
#
#          1
#   w_i = ------
#         d_i^p
#
# avec d_i la distance entre l'observation et le voisin.
#
# Les distances les plus utilisées sont :
#
# Euclidienne :
#
#      d(x,y)=√Σ(x_i-y_i)²
#
# Manhattan :
#
#      d(x,y)=Σ|x_i-y_i|
#
# Paramètres importants :
#
# n_neighbors :
#     nombre de voisins.
#
# weights :
#     "uniform"  -> tous les voisins ont le même poids.
#     "distance" -> les plus proches comptent davantage.
#
# metric :
#     distance utilisée (euclidienne, manhattan, ...).
#
# Avantages :
#
# - extrêmement simple ;
# - aucune phase d'apprentissage ;
# - peut modéliser des relations très non linéaires ;
# - fonctionne bien sur de petits jeux de données.
#
# Inconvénients :
#
# - prédiction lente (comparaison avec tout l'ensemble d'entraînement) ;
# - sensible au choix de k ;
# - nécessite une normalisation des variables ;
# - souffre de la malédiction de la dimension ;
# - devient moins performant lorsque le nombre de variables augmente.
#
# Complexité :
#
# Apprentissage :
#
#     O(n)
#
# Prédiction :
#
#     O(np)
#
# où
#
# n = nombre d'observations
# p = nombre de variables.
#
# ============================================================================

knn = KNeighborsRegressor(
        n_neighbors=3,
        weights="distance"
    )

resultats["K-Nearest Neighbors (KNN)"] = train_model(knn, 5)


ks = range(1,31)

scores = []

for k in ks:
    knn = KNeighborsRegressor(
        n_neighbors=k,
        weights="distance"
    )

    resultats["K-Nearest Neighbors (KNN)"] = train_model(knn, 5)
    scores.append(resultats["K-Nearest Neighbors (KNN)"]["R2"])

plt.plot(ks, scores)
plt.show()


print("=== Support Vector Regression (SVR) ===")

# ============================================================
# Support Vector Regression (SVR)
# ============================================================

# Le SVR (Support Vector Regression) est l'adaptation de la
# méthode des Support Vector Machines (SVM) au problème de
# régression.

# Contrairement à la régression linéaire qui cherche à minimiser
# directement l'erreur quadratique, le SVR cherche une fonction
# aussi "plate" que possible tout en autorisant une erreur de
# prédiction inférieure à ε (epsilon).

# ------------------------------------------------------------
# Principe
# ------------------------------------------------------------

# On construit un tube de largeur 2ε autour de la fonction de
# régression.

# Les observations situées à l'intérieur du tube ne sont pas
# pénalisées.

# Seules les observations situées en dehors du tube contribuent
# au coût.

# Les observations qui touchent ou dépassent ce tube sont les
# "Support Vectors".

# Ce sont elles qui déterminent entièrement la solution.

# ------------------------------------------------------------
# Fonction objectif
# ------------------------------------------------------------

# Le SVR résout

#
#      1
# min --- ||w||² + C Σ Lε(yi-f(xi))
#      2
#
#
# où

# Lε(r)=max(0, |r|-ε)

# est la perte ε-insensible.

# ------------------------------------------------------------
# Paramètres importants
# ------------------------------------------------------------

# kernel :
#   "linear"
#   "poly"
#   "rbf"
#   "sigmoid"

# C :
# pénalité des erreurs.
#
# petit C :
#    modèle simple
#
# grand C :
#    modèle plus complexe.

# epsilon :
# largeur du tube d'erreur.

# gamma :
# uniquement pour les noyaux non linéaires.
#
# petit gamma :
#    influence large.
#
# grand gamma :
#    influence très locale.

# ------------------------------------------------------------
# Avantages
# ------------------------------------------------------------

# Très performant sur des jeux de données de taille moyenne.

# Gère naturellement les relations non linéaires via les kernels.

# Bonne capacité de généralisation.

# ------------------------------------------------------------
# Inconvénients
# ------------------------------------------------------------

# Très sensible au choix des paramètres.

# Nécessite une standardisation des variables.

# Devient lent lorsque le nombre d'observations devient grand.

# Peu interprétable comparé à une régression linéaire.

svr = SVR(
    kernel="linear",
    C=100000,
    epsilon=1,
    gamma="scale"
)

resultats["SVR"] = train_model(svr, 3)

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





"""print("=== Arbre de décision ===")

# =============================================================================
# Arbre de décision (DecisionTreeRegressor)
# =============================================================================
#
# Principe
# --------
# Un arbre de décision construit récursivement une partition de l'espace des
# variables explicatives afin d'obtenir des feuilles les plus homogènes
# possibles.
#
# Chaque nœud pose une question du type :
#
#     Surface < 120 m² ?
#     DistanceCentre < 5 km ?
#     Ville ∈ {Paris, Lyon} ?
#
# Les observations sont alors réparties dans les deux sous-arbres.
#
#
# Construction
# ------------
# Pour chaque nœud :
#
#   Pour chaque variable :
#
#       - tester toutes les séparations possibles ;
#       - calculer la qualité de chaque séparation ;
#       - conserver la meilleure ;
#
# puis recommencer récursivement sur chacun des deux groupes.
#
#
# Variables numériques
# --------------------
# Les valeurs sont d'abord triées.
#
# Si les valeurs observées sont
#
#     2   4   6   8
#
# seuls les milieux
#
#     3   5   7
#
# sont testés comme seuils.
#
# Cela évite de tester une infinité de valeurs.
#
#
# Variables catégorielles
# -----------------------
# Théoriquement, on cherche la meilleure partition des k catégories.
#
# Nombre de partitions possibles :
#
#     2^(k-1) - 1
#
# Lorsque k est grand, cela devient rapidement très coûteux.
#
# En pratique on utilise souvent :
#
#   - OneHotEncoder ;
#   - OrdinalEncoder (si un ordre naturel existe) ;
#   - ou des algorithmes spécialisés (CatBoost, LightGBM...).
#
#
# Critère de qualité
# ------------------
# En classification :
#
#   - indice de Gini
#
#         G = 1 - Σ p_i²
#
#   - ou entropie
#
#         H = - Σ p_i log(p_i)
#
# Les deux mesurent l'impureté d'un nœud.
#
# L'arbre choisit la séparation qui minimise
# l'impureté moyenne pondérée des deux enfants.
#
#
# En régression
# -------------
# On ne parle plus d'impureté de Gini.
#
# Une feuille prédit simplement la moyenne des observations
# qu'elle contient.
#
# Le critère consiste à minimiser la variance (ou,
# de manière équivalente, la somme des carrés des erreurs)
# dans les feuilles.
#
#
# Pourquoi une moyenne pondérée ?
# -------------------------------
# Une feuille contenant 100 observations doit avoir beaucoup plus
# d'importance qu'une feuille contenant seulement 2 observations.
#
#
# Paramètres principaux
# ---------------------
#
# max_depth
#     Profondeur maximale de l'arbre.
#
# min_samples_split
#     Nombre minimal d'observations pour pouvoir couper un nœud.
#
# min_samples_leaf
#     Nombre minimal d'observations dans chaque feuille.
#
# max_leaf_nodes
#     Nombre maximal de feuilles.
#
# max_features
#     Nombre maximal de variables testées à chaque séparation.
#     (Très important pour Random Forest.)
#
#
# Surapprentissage
# ----------------
# Sans contrainte, l'arbre continue de se diviser jusqu'à obtenir
# des feuilles presque pures.
#
# Il finit alors par mémoriser les données d'entraînement
# (overfitting).
#
# Les paramètres précédents servent essentiellement
# à limiter cette complexité.
#
#
# Importance des variables
# ------------------------
# Contrairement à la régression linéaire, il n'existe pas de
# coefficients β.
#
# L'arbre fournit à la place une importance de chaque variable :
#
#     feature_importances_
#
# correspondant à la diminution totale d'erreur apportée
# par cette variable au cours des différentes séparations.
#
#
# Prétraitement
# -------------
# Les arbres ne nécessitent pas de normalisation.
#
# En effet, ils utilisent uniquement des comparaisons du type
#
#     x < seuil
#
# Une transformation monotone (StandardScaler, MinMaxScaler...)
# modifie les valeurs du seuil mais ne change pas les partitions.
#
#
# Avantages
# ---------
# - très intuitif ;
# - facilement interprétable ;
# - peu de prétraitement ;
# - gère naturellement les relations non linéaires ;
# - accepte les interactions entre variables.
#
#
# Inconvénients
# -------------
# - surapprend facilement ;
# - très sensible aux données d'entraînement ;
# - forte variance ;
# - souvent moins performant qu'une méthode d'ensemble
#   (Random Forest, Gradient Boosting...).
#
#
# Idée fondamentale
# -----------------
# Un arbre de décision construit récursivement une partition
# de l'espace des données afin de minimiser l'erreur (régression)
# ou de maximiser la pureté des feuilles (classification).
#
#
# Pseudo-code
# -----------
#
# meilleur_score = +∞
#
# pour chaque variable :
#
#     trier les valeurs
#
#     pour chaque seuil possible :
#
#         couper les données
#
#         calculer le score des deux groupes
#
#         calculer le score pondéré
#
#         si meilleur :
#
#             mémoriser ce seuil
#
# créer deux nouveaux nœuds
#
# recommencer récursivement
#
# =============================================================================

for depth in [6]:
    arbre = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    #resultats[f"Arbre de décision - Profondeur {depth}"] = train_model(arbre, 1)


depths = [2,3,4,5,6,7,8,10]
r2 = [0.789,0.845,0.889,0.903,0.912,0.906,0.898,0.869]

plt.plot(depths, r2, marker="o")
plt.xlabel("max_depth")
plt.ylabel("R²")
plt.grid(True)
plt.show()

print("=== Random Forest ===")

# =============================================================================
# Random Forest Regressor
# =============================================================================
#
# Principe
# --------
# Une Random Forest (forêt aléatoire) est un ensemble d'arbres de décision.
#
# Au lieu de construire un seul arbre, on construit plusieurs centaines
# d'arbres puis on moyenne leurs prédictions.
#
# Pour la régression :
#
#        prédiction finale
#              =
# moyenne des prédictions de tous les arbres
#
#
# Pourquoi ?
# ----------
# Un arbre unique possède une forte variance :
# une légère modification des données peut produire un arbre très différent.
#
# En moyennant de nombreux arbres, les erreurs individuelles se compensent.
#
# L'objectif principal est donc de diminuer la variance tout en conservant
# un biais relativement faible.
#
#
# Construction d'un arbre
# -----------------------
# Chaque arbre est construit sur un échantillon Bootstrap.
#
# Bootstrap :
#
# - on tire n observations parmi les n observations du jeu d'entraînement ;
# - le tirage est effectué AVEC remise ;
# - certaines observations apparaissent plusieurs fois ;
# - environ 36 % des observations ne sont pas utilisées dans un arbre donné
#   (Out-Of-Bag samples).
#
#
# Sélection aléatoire des variables
# ---------------------------------
# A chaque séparation, l'arbre ne teste pas toutes les variables.
#
# Il en choisit aléatoirement seulement quelques-unes
# (paramètre max_features).
#
# Cette étape rend les arbres moins corrélés entre eux,
# ce qui améliore la réduction de variance.
#
#
# Algorithme
# ----------
#
# Pour i = 1,...,B :
#
#     tirer un échantillon Bootstrap
#
#     construire un arbre complet
#
#         à chaque nœud :
#
#             sélectionner aléatoirement quelques variables
#
#             rechercher la meilleure séparation
#
# Fin
#
# La prédiction finale est la moyenne des B arbres.
#
#
# Paramètres principaux
# ---------------------
#
# n_estimators
#     Nombre d'arbres.
#
# max_depth
#     Profondeur maximale de chaque arbre.
#
# min_samples_split
#     Nombre minimal d'observations pour couper un nœud.
#
# min_samples_leaf
#     Nombre minimal d'observations dans une feuille.
#
# max_features
#     Nombre de variables testées à chaque séparation.
#
# bootstrap
#     Utilisation ou non du Bootstrap.
#
# random_state
#     Graine du générateur pseudo-aléatoire.
#
#
# Importance des variables
# ------------------------
# Comme pour un arbre unique, Random Forest calcule
#
#     feature_importances_
#
# correspondant à la diminution moyenne de l'erreur
# apportée par chaque variable sur l'ensemble de la forêt.
#
#
# Out Of Bag (OOB)
# ----------------
# Les observations non utilisées pour construire un arbre
# permettent d'obtenir une estimation de l'erreur de généralisation
# sans créer explicitement un jeu de validation.
#
# Dans scikit-learn :
#
#     oob_score=True
#
#
# Prétraitement
# -------------
# Les arbres ne nécessitent pas de normalisation.
#
# Un SimpleImputer est généralement suffisant.
#
# Les variables catégorielles doivent néanmoins être encodées
# (One-Hot Encoder par exemple).
#
#
# Avantages
# ---------
# - très bonnes performances ;
# - réduit fortement le surapprentissage d'un arbre unique ;
# - robuste au bruit ;
# - gère naturellement les relations non linéaires ;
# - peu sensible aux valeurs aberrantes ;
# - fournit une importance des variables.
#
#
# Inconvénients
# -------------
# - moins interprétable qu'un arbre unique ;
# - plus coûteux en temps de calcul ;
# - nécessite davantage de mémoire ;
# - les prédictions sont une moyenne de nombreux arbres,
#   il n'existe plus de règles de décision simples.
#
#
# Différence avec un arbre unique
# -------------------------------
#
# Arbre :
#
#     forte variance
#
#     facilement en surapprentissage
#
# Random Forest :
#
#     faible variance
#
#     meilleure généralisation
#
#     prédictions plus stables
#
#
# Idée fondamentale
# -----------------
# Une Random Forest construit de nombreux arbres indépendants
# sur des échantillons Bootstrap et moyenne leurs prédictions
# afin de réduire la variance d'un arbre de décision.
# =============================================================================

for estimators in [10, 50, 100, 300]:
    for depth in [2, 3, 4, 5, 6, 7, 8, 10, None]:
        rf = RandomForestRegressor(
            n_estimators=estimators,
            max_depth=depth,
            max_features="sqrt",
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )

        resultats[f"Random Forest - Profondeur {depth} - n_estimators {estimators}"] = train_model(rf, 3)

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    max_features=None,
    min_samples_leaf = 1,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

param_grid = {
    "regression__n_estimators":[100,300,500],
    "regression__max_depth":[5,8,10,None],
    "regression__max_features":[
        "sqrt",
        None,
        5,
        8,
        10
    ],
    "regression__min_samples_leaf":[
        1,
        2,
        5
    ]
}

#optimize_model(RandomForestRegressor(), param_grid)


#resultats["Random Forest - Estimator 300 - depth 10 - Feature None - min_samples_leaf 1"] = train_model(rf, 2)

print("=== Gradient Boosting ===")

# =============================================================================
# Gradient Boosting Regressor
# =============================================================================
#
# Principe général
# ----------------
# Le Gradient Boosting est une méthode d'ensemble basée sur des arbres
# construits séquentiellement.
#
# Contrairement à la Random Forest (arbres indépendants), ici chaque arbre
# corrige les erreurs des arbres précédents.
#
#
# Idée fondamentale
# -----------------
# On cherche à approximer une fonction f(x) en construisant une somme
# d'arbres faibles :
#
#     f(x) = f0(x) + η * h1(x) + η * h2(x) + ... + η * hM(x)
#
# où :
# - f0(x) est une première approximation (souvent la moyenne des y)
# - h_m(x) est un petit arbre de décision
# - η (learning rate) contrôle la vitesse d'apprentissage
#
#
# Interprétation intuitive
# -------------------------
# Chaque nouvel arbre apprend à prédire les résidus :
#
#     r_i = y_i - f(x_i)
#
# Autrement dit, chaque arbre corrige les erreurs du modèle courant.
#
#
# Lien avec optimisation
# ----------------------
# Le Gradient Boosting peut être vu comme une descente de gradient
# dans l'espace des fonctions.
#
# À chaque étape, on ajoute une fonction (arbre) qui suit la direction
# du gradient négatif de la fonction de perte.
#
#
# Hyperparamètres principaux
# --------------------------
#
# n_estimators
#     Nombre d'arbres successifs.
#     Plus il est grand, plus le modèle est flexible.
#
# learning_rate (η)
#     Taille du pas de correction.
#     Petit η → apprentissage lent mais plus robuste.
#     Grand η → apprentissage rapide mais risque de surapprentissage.
#
# max_depth
#     Profondeur des arbres (souvent faible : 2 à 5).
#     Les arbres sont des "weak learners".
#
#
# Différence avec Random Forest
# -----------------------------
#
# Random Forest :
# - arbres indépendants
# - réduction de la variance
# - moyenne des prédictions
#
# Gradient Boosting :
# - arbres dépendants
# - réduction du biais
# - correction séquentielle des erreurs
#
#
# Avantages
# ---------
# - très bonnes performances en pratique
# - capture des relations complexes non linéaires
# - souvent meilleur que Random Forest sur données structurées
#
#
# Inconvénients
# -------------
# - entraînement séquentiel (moins parallélisable)
# - sensible aux hyperparamètres
# - risque de surapprentissage si mal réglé
#
#
# Résumé final
# ------------
# Le Gradient Boosting construit une somme de petits arbres,
# où chaque arbre apprend à corriger les erreurs du précédent,
# permettant de réduire progressivement l'erreur globale.
# =============================================================================

for rate in [0.05]:
    for depth in [2]:
        gbr = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=rate,
            max_depth=depth,
            subsample = 0.8,
            random_state=42
        )

    #resultats[f"Gradient Boosting - Rate {rate} - Depth {depth} - Sample 0.8"] = train_model(gbr, 3)

param_grid = {
    "regression__learning_rate":[
        0.01,
        0.03,
        0.05,
        0.1
    ],
    "regression__max_depth":[
        2,
        3,
        4
    ],
    "regression__n_estimators":[
        100,
        300,
        500
    ],
    "regression__subsample":[
        0.8,
        1.0
    ]
}


#optimize_model(GradientBoostingRegressor(), param_grid)

    
print("=== XGBoost ===")

# ========================= XGBoost =========================
#
# XGBoost (Extreme Gradient Boosting) est une amélioration du Gradient Boosting.
#
# Principe :
#
# Initialiser le modèle
#
#     F0(x) = moyenne(y)
#
# Pour chaque itération :
#
#     1. Calculer le gradient (et la dérivée seconde de la fonction de coût).
#
#     2. Construire un petit arbre qui approxime ces gradients.
#
#     3. Choisir les séparations maximisant un gain prenant en compte :
#            - l'erreur,
#            - une pénalité sur la complexité de l'arbre.
#
#     4. Élaguer les branches dont le gain est insuffisant (gamma).
#
#     5. Ajouter l'arbre au modèle :
#
#            F(x) ← F(x) + learning_rate x arbre
#
# À la fin, la prédiction est la somme des prédictions de tous les arbres.
#
# Contrairement au Gradient Boosting classique :
#   - utilise les gradients ET les dérivées secondes (Hessiennes) ;
#   - ajoute une régularisation L1/L2 ;
#   - possède un élagage ("pruning") intégré ;
#   - peut gérer automatiquement les valeurs manquantes ;
#   - est fortement optimisé pour la vitesse.
#
# Paramètres importants :
#
# n_estimators      : nombre d'arbres
# learning_rate     : poids de chaque nouvel arbre
# max_depth         : profondeur maximale des arbres
# subsample         : proportion des observations utilisées
# colsample_bytree  : proportion des variables utilisées
# gamma             : gain minimal pour effectuer une séparation (par défaut 0)
# reg_alpha         : pénalisation L1 (par défaut 0)
# reg_lambda        : pénalisation L2 (par défaut 1)
#
# Avantages :
#   - très performant ;
#   - robuste au surapprentissage ;
#   - capture des relations non linéaires ;
#   - fournit des importances de variables ;
#   - compatible avec SHAP.
#
# Inconvénients :
#   - davantage d'hyperparamètres ;
#   - entraînement plus long ;
#   - interprétation moins simple qu'une régression linéaire.
# ===========================================================


xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=2,
    subsample=0.8,
    colsample_bytree=1,
    gamma = 0,
    reg_alpha = 0.1,
    reg_lambda = 10,
    random_state=42
)

#resultats[f"XGBoost - Rate 0.03 - Depth 2"] = train_model(xgb, 3)


param_grid = {
    "regression__n_estimators": [500],
    "regression__learning_rate": [0.03],
    "regression__max_depth": [2],
    "regression__subsample": [0.7, 0.8, 1.0],
    "regression__colsample_bytree": [0.7, 0.8, 1.0],
    "regression__gamma": [0, 0.5, 1],
    "regression__reg_alpha": [0, 0.1, 1],
    "regression__reg_lambda": [1, 3, 10]
}

#optimize_model(XGBRegressor(), param_grid)




print("=== LightGBM ===")

# ========================= LightGBM =========================
#
# LightGBM (Light Gradient Boosting Machine) est une amélioration
# du Gradient Boosting conçue pour être plus rapide et plus
# économe en mémoire que XGBoost.
#
# Principe :
#
# Initialiser le modèle
#
#     F0(x) = moyenne(y)
#
# Pour chaque itération :
#
#     1. Calculer les gradients de la fonction de coût.
#
#     2. Construire un nouvel arbre de décision.
#
#     3. Développer uniquement la feuille dont le gain est maximal
#        (croissance "leaf-wise"), contrairement aux arbres
#        classiques qui se développent niveau par niveau
#        ("level-wise").
#
#     4. Ajouter ce nouvel arbre :
#
#            F(x) ← F(x) + learning_rate x arbre
#
# La prédiction finale est la somme des prédictions de tous les arbres.
#
# Optimisations propres à LightGBM :
#
#   - croissance feuille par feuille (leaf-wise) ;
#   - histogrammes pour accélérer la recherche des seuils ;
#   - GOSS (Gradient-based One-Side Sampling) qui conserve
#     principalement les observations ayant les plus grands gradients ;
#   - EFB (Exclusive Feature Bundling) qui fusionne des variables
#     peu utilisées simultanément afin de réduire la dimension.
#
# Paramètres importants :
#
# n_estimators      : nombre d'arbres
# learning_rate     : poids de chaque arbre
# max_depth         : profondeur maximale
# num_leaves        : nombre maximal de feuilles
# min_child_samples : taille minimale d'une feuille
# subsample         : proportion d'observations utilisées
# colsample_bytree  : proportion de variables utilisées
#
# Avantages :
#   - très rapide ;
#   - faible consommation mémoire ;
#   - excellent sur les grands jeux de données ;
#   - très bonnes performances ;
#   - compatible avec SHAP.
#
# Inconvénients :
#   - davantage de paramètres à régler ;
#   - peut surapprendre si num_leaves est trop grand ;
#   - un peu moins interprétable qu'un arbre classique.
# ===========================================================

for leaves in [3, 7, 15]:
    lgbm = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        num_leaves = leaves,
        min_child_samples=5,
        importance_type="gain",
        random_state=42,
        verbose=-1
    )

    #resultats[f"LightGBM - Rate 0.05 - Depth 3 - Leaves {leaves}"] = train_model(lgbm, 2)


print("=== CatBoost ===")


CatBoost (Categorical Boosting)

CatBoost est une méthode de Gradient Boosting développée par Yandex,
spécialisée dans le traitement des variables catégorielles.

Principe :

- Initialiser un modèle constant.
- Répéter :
    - calculer les résidus ;
    - construire un arbre peu profond sur ces résidus ;
    - ajouter cet arbre au modèle avec un facteur learning_rate.
- La prédiction finale est la somme de tous les arbres.

Contrairement aux autres méthodes de boosting, CatBoost :

- traite directement les variables catégorielles sans One-Hot Encoding ;
- encode les catégories de manière progressive (target statistics) ;
- utilise l'Ordered Boosting afin d'éviter les fuites d'information
  lors du calcul des résidus.

Paramètres principaux :

iterations      : nombre d'arbres.
learning_rate   : poids de chaque arbre ajouté.
depth           : profondeur maximale des arbres.
l2_leaf_reg     : régularisation L2.
loss_function   : fonction de coût (RMSE en régression).
random_seed     : reproductibilité.
verbose         : affichage de l'apprentissage.

Avantages :

- excellent avec de nombreuses variables catégorielles ;
- peu de prétraitement ;
- très robuste au surapprentissage ;
- souvent parmi les meilleurs algorithmes sur les données tabulaires.

Inconvénients :

- apprentissage plus lent que LightGBM ;
- davantage d'hyperparamètres ;
- interprétation plus difficile qu'un modèle linéaire.

Pseudo-code :

Initialiser un modèle constant

Pour t = 1 ... T :

    Calculer les résidus

    Construire un arbre sur ces résidus

    Ajouter learning_rate x arbre au modèle

Retourner la somme de tous les arbres

cat = CatBoostRegressor(
    iterations=300,
    learning_rate=0.05,
    depth=3,
    loss_function="RMSE",
    random_seed=42,
    verbose=False
)

# En utilisant onehot encoder
#resultats[f"CatBoost - Rate 0.05 - Depth 3 - Avec OneHot"] = train_model(cat, 2)

# Sans utiliser onehot encoder (Sans utiliser le pipeline)

X_train2 = X_train.copy()
X_test2 = X_test.copy()

# uniquement imputation
for c in colonnes_num:
    X_train2[c] = X_train2[c].fillna(X_train2[c].median())
    X_test2[c] = X_test2[c].fillna(X_train2[c].median())

for c in colonnes_cat:
    mode = X_train2[c].mode()[0]
    X_train2[c] = X_train2[c].fillna(mode)
    X_test2[c] = X_test2[c].fillna(mode)

for depth in [3]:
    for rate in [0.03]:
        for nb in [1000]:
            cat = CatBoostRegressor(
                iterations=nb,
                learning_rate=rate,
                depth=depth,
                loss_function="RMSE",
                random_seed=42,
                verbose=False
            )

            cat.fit(
                X_train2,
                y_train,
                cat_features=colonnes_cat.tolist(),
                verbose=False
            )

            y_pred2 = cat.predict(X_test2)

            resultats[f"CatBoost - Rate {rate} - Depth {depth} - Iterations {nb} - Sans OneHot"] = regression_metrics(y_test, y_pred2)
   


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


# Test Git