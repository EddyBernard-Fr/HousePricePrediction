import pandas as pd
import random

random.seed(42)

villes = ["Paris", "Lyon", "Marseille", "Toulouse", "Lille"]
types = ["Appartement", "Maison", "Loft"]

donnees = []

for _ in range(250):

    surface = random.randint(25, 220)
    chambres = random.randint(1, 6)
    salles_bain = random.randint(1, 3)
    annee = random.randint(1960, 2024)

    ville = random.choice(villes)
    type_maison = random.choice(types)

    garage = random.choice(["Oui", "Non"])
    jardin = random.choice(["Oui", "Non"])

    distance = round(random.uniform(0.5, 30), 1)

    prix = (
        50000
        + surface * 3200
        + chambres * 8000
        + salles_bain * 12000
        + (15000 if garage == "Oui" else 0)
        + (18000 if jardin == "Oui" else 0)
        + (30000 if type_maison == "Maison" else 20000 if type_maison == "Loft" else 0)
        + {
            "Paris": 180000,
            "Lyon": 80000,
            "Marseille": 20000,
            "Toulouse": 40000,
            "Lille": 30000,
        }[ville]
        - distance * 2500
        + (annee - 1960) * 500
        + random.gauss(0, 30000)
    )

    donnees.append([
        surface,
        chambres,
        salles_bain,
        annee,
        ville,
        type_maison,
        garage,
        jardin,
        distance,
        round(prix)
    ])

df = pd.DataFrame(
    donnees,
    columns=[
        "Surface",
        "Chambres",
        "SallesDeBain",
        "AnneeConstruction",
        "Ville",
        "TypeMaison",
        "Garage",
        "Jardin",
        "DistanceCentre",
        "Prix"
    ]
)

# Valeurs manquantes
for col, n in [
    ("Surface", 8),
    ("Ville", 6),
    ("TypeMaison", 5),
    ("DistanceCentre", 5),
]:
    indices = random.sample(range(len(df)), n)
    df.loc[indices, col] = None

df.to_csv("projet_prix_maisons.csv", index=False)

print(df.head())
print()
print(df.isna().sum())