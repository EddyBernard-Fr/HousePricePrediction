# House Price Prediction

Projet de Machine Learning permettant de prédire le prix de maisons à partir de différentes caractéristiques.

## Fonctionnalités

- Prétraitement des données
- Entraînement de plusieurs modèles
- Évaluation des performances
- Importance des variables
- Génération de l'arbre de décision au format PDF

---

## Structure du projet

```text
.
├── data/
├── figures/
├── notebooks/
├── src/
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/<ton-utilisateur>/HousePricePrediction.git
cd HousePricePrediction
```

### 2. Créer un environnement virtuel

#### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Dépendances système

### Graphviz

La génération de l'arbre de décision nécessite Graphviz.

### Windows

Télécharger Graphviz :

https://graphviz.org/download/

Puis ajouter le dossier `bin` au `PATH`.

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install graphviz
```

### macOS

```bash
brew install graphviz
```

---

## Exécution avec Jupyter

```bash
jupyter lab
```

Puis ouvrir :

```
notebooks/HousePricePrediction.ipynb
```

---

## Exécution avec Docker

Construire l'image :

```bash
docker build -t house-price-prediction .
```

Lancer Jupyter Lab :

```bash
docker run --rm -p 8888:8888 house-price-prediction
```

Puis ouvrir :

```
http://localhost:8888
```

---

## Tests

```bash
pytest
```

Avec couverture :

```bash
pytest --cov=src --cov-report=term-missing
```

---

## Technologies utilisées

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- LightGBM
- CatBoost
- SHAP
- Graphviz
- Jupyter Lab
- Docker
- GitHub Actions

## Auteur

Eddy Bernard