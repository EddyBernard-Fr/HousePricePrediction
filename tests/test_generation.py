from pathlib import Path
import pandas as pd
import sys

ROOT = Path.cwd().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import generer_donnees


def test_nombre_de_lignes(df):

    assert len(df) == 250

def test_df_is_dataframe(df):
    
    assert isinstance(df, pd.DataFrame)

def test_colonnes_attendues(df):

    colonnes = [
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

    assert list(df.columns) == colonnes


def test_prix_non_manquant(df):

    assert df["Prix"].notna().all()


def test_prix_positif(df):

    assert (df["Prix"] > 0).all()


def test_generation_reproductible():

    df1, X1, y1 = generer_donnees()

    df2, X2, y2 = generer_donnees()

    assert df1.equals(df2)

def test_shape(Xt_r, Xt_a):
<<<<<<< HEAD

    assert Xt_r.shape == (250,13)
    assert Xt_a.shape == (250,13)
=======
    
    assert Xt_r.shape == (500,13)
    assert Xt_a.shape == (500,13)
>>>>>>> d2f1784797c9e5797cd0849740eef3428f0bee0b
