# 🏠 House Price Prediction

Projet de Machine Learning visant à prédire le prix d'un logement à partir de ses caractéristiques.

Le projet compare plusieurs algorithmes de régression de scikit-learn ainsi que des bibliothèques spécialisées telles que XGBoost, LightGBM et CatBoost.

---

## Objectifs

- Générer un jeu de données réaliste.
- Réaliser une analyse exploratoire des données (EDA).
- Construire des pipelines de prétraitement.
- Comparer plusieurs modèles de régression.
- Optimiser les hyperparamètres.
- Interpréter les modèles à l'aide des coefficients, des importances de variables, des permutation importances et des valeurs SHAP.

---

## Jeu de données

Le jeu de données est généré artificiellement.

Chaque logement est décrit par :

- Surface
- Nombre de chambres
- Nombre de salles de bain
- Ville
- Type de logement
- Garage
- Jardin
- Année de construction
- Distance au centre-ville

Le prix est calculé à partir d'un modèle linéaire auquel est ajouté un bruit gaussien.

---

## Modèles étudiés

- Régression linéaire
- Ridge
- Lasso
- Elastic Net
- Arbre de décision
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- K-Nearest Neighbors
- Support Vector Regression

---

## Évaluation

Les modèles sont comparés à l'aide de plusieurs métriques :

- MAE
- RMSE
- Coefficient de détermination (R²)

Les meilleurs modèles atteignent un R² proche de **0,96**.

---

## Interprétabilité

Le projet comporte également plusieurs outils d'interprétation :

- coefficients de régression
- feature importance
- permutation importance
- SHAP values
- arbres de décision
- analyse des résidus

---

## Technologies utilisées

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- SHAP
- Jupyter Notebook

---

## Installation

Créer un environnement virtuel :

```bash
python -m venv .venv
```

L'activer puis installer les dépendances :

```bash
pip install -r requirements.txt
```

---

## Lancer le projet

Ouvrir le notebook :

```bash
jupyter notebook
```

ou

```bash
code .
```

puis exécuter le notebook principal.

---

## Auteur

Eddy Bernard