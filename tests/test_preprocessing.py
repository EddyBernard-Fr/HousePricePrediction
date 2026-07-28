
import numpy as np

def test_preprocessing_supprime_les_nan(Xt_r, Xt_a):

    assert not np.isnan(Xt_r).any()
    assert not np.isnan(Xt_a).any()


def test_preprocessing_conserve_le_nombre_de_lignes(Xt_r, Xt_a, df):

    assert Xt_r.shape[0] == len(df)
    assert Xt_a.shape[0] == len(df)


def test_nombre_de_variables_transformees(Xt_r, Xt_a):

    assert Xt_r.shape[1] == 13
    assert Xt_a.shape[1] == 13


def test_feature_names_regression(X, preprocessor_regression, preprocessor_arbre):

    preprocessor_regression.fit(X)
    preprocessor_arbre.fit(X)

    assert "num__Surface" in preprocessor_regression.get_feature_names_out()
    assert "cat__Ville_Paris" in preprocessor_regression.get_feature_names_out()
    assert "num__Surface" in preprocessor_arbre.get_feature_names_out()
    assert "cat__Ville_Paris" in preprocessor_arbre.get_feature_names_out()
