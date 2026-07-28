import pytest

from pathlib import Path
import sys

ROOT = Path.cwd().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import generer_donnees, pipeline_regression, pipeline_arbre


@pytest.fixture
def dataset():
    return generer_donnees()

@pytest.fixture
def df(dataset):
    return dataset[0]

@pytest.fixture
def X():

    _, X, _ = generer_donnees()

    return X

@pytest.fixture
def y():

    _, _, y = generer_donnees()

    return y

@pytest.fixture
def preprocessor_regression(X):
    return pipeline_regression(X.select_dtypes(include="number").columns, X.select_dtypes(include="str").columns)


@pytest.fixture
def preprocessor_arbre(X):
    return pipeline_arbre(X.select_dtypes(include="number").columns, X.select_dtypes(include="str").columns)


@pytest.fixture
def Xt_r(X, preprocessor_regression):
    return preprocessor_regression.fit_transform(X)

@pytest.fixture
def Xt_a(X, preprocessor_arbre):
    return preprocessor_arbre.fit_transform(X)