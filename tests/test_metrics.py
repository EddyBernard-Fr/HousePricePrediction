from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path.cwd().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import regression_metrics


def test_prediction_parfaite():

    y = np.array([1,2,3,4,5])

    resultat = regression_metrics(y, y, verbose=False)

    assert resultat["MAE"] == 0

    assert resultat["RMSE"] == 0

    assert resultat["R2"] == 1


def test_rmse_positive():

    y = np.array([1,2,3,4])

    y_pred = np.array([2,3,4,5])

    resultat = regression_metrics(y, y_pred, verbose=False)

    assert resultat["RMSE"] >= 0



def test_rmse_augmente_si_prediction_mauvaise():

    y = np.array([1,2,3,4])

    y1 = np.array([1,2,3,4])

    y2 = np.array([10,20,30,40])

    rmse1 = regression_metrics(y,y1,verbose=False)["RMSE"]

    rmse2 = regression_metrics(y,y2,verbose=False)["RMSE"]

    assert rmse2 > rmse1


@pytest.mark.parametrize(
    "y_true, y_pred",
    [
        (
            np.array([1,2,3]),
            np.array([1,2,3])
        ),
        (
            np.array([10,20,30]),
            np.array([12,19,31])
        ),
        (
            np.array([5,6,7]),
            np.array([0,0,0])
        )
    ]
)

def test_mae_positive(y_true, y_pred):

    resultat = regression_metrics(
        y_true,
        y_pred,
        verbose=False
    )

    assert resultat["MAE"] >= 0