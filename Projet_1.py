
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, zscore


# Chargement des données

df = pd.read_csv("projet_prix_maisons.csv")

# Affichage des 5 premières lignes

print(df.head())

# Dimension du jeu de données

print(f"La dimension du jeu de données est {df.shape}.")

# Informations sur les colonnes 

print(f"Il y a en tout {len(df.columns)} colonnes, dont voici les intitulés:")

print(df.columns)

print("Voici ci-dessous leur types:")

print(df.dtypes)

# Nombre de ligne

print(f"Il y a en tout {len(df.index)} lignes dans le jeu de données.")

# Statistiques descriptive du jeu de données

print("Voici les statistiques descriptives des variables numériques:")

print(df.describe())

colonnes_num = df.select_dtypes(include="number").columns

for col in colonnes_num:

    df.boxplot(column=col)
    plt.show()

print("Voici les statistiques descriptives des variables catégoritielles:")

colonnes_cat = df.select_dtypes(include="str").columns # On récupérère le nom des colonnes catégoritielles (on peut aussi utiliser "object" à la place de "str")

for col in colonnes_cat:
    print(f"\n===== {col} =====")
    print(df[col].value_counts(normalize=True) * 100) # On affiche la proportions de toutes les variables qualitatives.

# Valeurs manquantes par colonne

print("Voici le nombre de valeurs manquantes par colonne:")

print(df.isna().sum())

# Remarque: df.info regroupe toutes ces infos sauf les statistiques descriptives


# Répartition des varaibles numériques

for col in colonnes_num:
    print(f"\n===== {col} =====")

    n, bins, patches = plt.hist(df[col], bins=10)

    print("n =", n)
    print("somme =", sum(n))
    print("bins =", bins)
    plt.title(f"{col}")

    plt.show()

# Répartition des variables catégoritielles

for col in colonnes_cat:
    print(f"\n===== {col} =====")
    vc = df[col].value_counts()
    labels = vc.index
    valeurs = vc.values
    plt.bar(labels, valeurs)
    plt.title(f"{col}")

    plt.show()

# Corrélation entre les variables numériques

print("Voici la corrélation entre les variables:")

corr = df.corr(numeric_only=True)

print(corr)

print(corr["Prix"].sort_values(ascending=False))

# Effet des variables catégorielle sur le prix

print("Effet de la variable catégorielle sur le prix:")

for col in colonnes_cat:
    print(f"\n===== {col} =====")
    print(df.groupby(col)["Prix"].agg(["count", "mean", "median", "std"])) # Calcul le prix moyen, médian et l'écart type pour les différentes valeurs des variables catégoritielles
    
    df.boxplot(column="Prix", by=col) # La moustache supérieure est la plus grande observation qui ne dépasse pas la borne supérieure. 

    plt.title(f"Prix selon {col}")
    plt.suptitle("")   # enlève le titre automatique de pandas
    plt.xlabel(f"{col}")
    plt.ylabel("Prix")
    plt.show() 

# Effet des variables numériques sur le prix

print("Effet des variables numériques sur le prix:")


X = df.drop(columns="Prix")
var_num = X.select_dtypes(include="number").columns

for var in var_num:
    plt.scatter(df[var], df["Prix"])
    plt.xlabel(var)
    plt.ylabel("Prix")
    plt.show() 

# ANOVA (test statistique)

groupes = [
    g["Prix"].dropna().values
    for _, g in df.groupby("Ville")
]

F, p = f_oneway(*groupes)

print(F) # F=variation intragroupe/ variation intergroupe​ (plus F est grand plus la variable semble avoir de l'effet sur le prix, F=1 pas d'effet)
print(p) # Probabilité d'obtenir une statistique F, où l'on considère toute les moyennes des groupes égales, au moins aussi grande que celle observée (p proche de zéro, on rejette l'hypothèse des moyennes égales)


# Recherche des valeurs abérrantes (outliers) statistiquement

# A) Méthode IQR

for col in colonnes_num:

    Q1 = df[col].quantile(0.25)

    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    borne_inf = Q1 - 1.5 * IQR
    borne_sup = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < borne_inf)
        |
        (df[col] > borne_sup)
    ]

    print(f"Les outliers (IQR) pour la variable {col} sont {outliers[col].tolist()}. Il y en a en tout {len(outliers)}, soit une proportion {len(outliers) / len(df)}")
    print(f"Borne inf : {borne_inf:.2f}")
    print(f"Borne sup : {borne_sup:.2f}")

# B) Métode z-score

for col in colonnes_num:

    z = zscore(df[col], nan_policy="omit")
    outliers = df[abs(z) > 3]

    print(f"Les outliers (z-score) pour la variable {col} sont {outliers[col].tolist()}. Il y en a en tout {len(outliers)}, soit une proportion {len(outliers) / len(df)}")


# Recherche des valeurs aberrantes métiers (domain knowledge)

contraintes = {
    "Surface": lambda s: s < 0,
    "Prix": lambda s: s < 0,
    "Chambres": lambda s: s <= 0,
    "SallesDeBain": lambda s: s <= 0,
    "DistanceCentre": lambda s: s < 0,
    "AnneeConstruction": lambda s: s > 2026,
}

for col, test in contraintes.items():
    aberrantes = df[test(df[col])]
    print(f"{col} : {len(aberrantes)} valeurs aberrantes métiers")


# Remarques: Il peut aussi avoir des incohérences, par exemple Garage = Non et SurfaceGarage = 35 m² qui sont indétectable statisitquement