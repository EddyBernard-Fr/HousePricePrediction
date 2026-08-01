# House Price Prediction using Machine Learning

## Highlights

- End-to-end supervised regression project
- Data preprocessing and feature engineering
- Comparison of multiple machine learning models
- Hyperparameter tuning
- Feature importance analysis
- Decision Tree visualization
- Model interpretation using SHAP

This project aims to predict house prices from various property characteristics using supervised machine learning algorithms.

The project follows a complete machine learning workflow, including data preprocessing, model comparison, hyperparameter tuning, model interpretation, and visualization.

## Project Overview

The objective of this project is to build and evaluate regression models capable of accurately predicting house prices.

Several machine learning algorithms are trained and compared in order to identify the best-performing model while gaining insight into the importance of each feature.

## Features

The project includes:

- Data preprocessing
- Training multiple machine learning models
- Model performance evaluation
- Feature importance analysis
- SHAP explainability
- Decision Tree visualization in PDF format

## Project Structure

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

## Technologies

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

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/EddyBernard-Fr/HousePricePrediction.git
cd HousePricePrediction
```

### 2. Create a virtual environment

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

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

## System Dependencies

### Graphviz

Graphviz is required to generate the Decision Tree visualization.

### Windows

Download Graphviz:

https://graphviz.org/download/

Then add the `bin` directory to your system `PATH`.

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install graphviz
```

### macOS

```bash
brew install graphviz
```

## Running with Jupyter

```bash
jupyter lab
```

Then open:

```text
notebooks/HousePricePrediction.ipynb
```

## Running with Docker

Build the Docker image:

```bash
docker build -t house-price-prediction .
```

Run Jupyter Lab:

```bash
docker run --rm -p 8888:8888 house-price-prediction
```

Then open:

```text
http://localhost:8888
```

## Running the Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

## Educational Objectives

This project was mainly developed to strengthen my understanding of:

- Machine learning workflows
- Regression algorithms
- Feature engineering
- Hyperparameter tuning
- Model explainability with SHAP
- Reproducible machine learning projects
- Docker-based development environments

## Future Work

Possible improvements include:

- Cross-validation for all regression models
- Automated hyperparameter optimization
- Model deployment with FastAPI
- Interactive dashboard for predictions
- Additional explainability techniques
- CI/CD pipeline improvements

## Author

**Eddy Bernard**

PhD in Mathematics & Theoretical Chemistry

GitHub: https://github.com/EddyBernard-Fr

This repository is part of my machine learning portfolio.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.