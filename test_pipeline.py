"""Tests unitaires : une fonction = un ou plusieurs tests.

Lancer :  pytest -v test_pipeline.py     (ou : python test_pipeline.py)
Les tests utilisent de petites données générées : aucun fichier réel requis.
"""

import os
import zipfile

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from model_pipeline import (
    evaluate_model, load_model, predict_model, prepare_data, save_model,
    train_model,
)


def _make_adult_zip(folder, n=600):
    """Crée un mini adult.zip au format UCI (adult.data + adult.test)."""
    rng = np.random.RandomState(0)

    def row():
        return ", ".join(map(str, [
            rng.randint(17, 90), rng.choice(["Private", "State-gov", "?"]),
            rng.randint(10000, 900000), "Bachelors", rng.choice([9, 13, 14]),
            rng.choice(["Married-civ-spouse", "Never-married"]),
            rng.choice(["Sales", "Craft-repair", "?"]),
            rng.choice(["Husband", "Own-child"]), "White",
            rng.choice(["Male", "Female"]), rng.choice([0, 0, 5000, 99999]), 0,
            rng.randint(20, 60), rng.choice(["United-States", "Mexico", "?"]),
            rng.choice(["<=50K", ">50K"]),
        ]))

    data, test = os.path.join(folder, "adult.data"), os.path.join(folder, "adult.test")
    open(data, "w").write("\n".join(row() for _ in range(n)) + "\n")
    open(test, "w").write("|1x3 Cross validator\n"
                          + "\n".join(row() + "." for _ in range(n // 3)) + "\n")
    zip_path = os.path.join(folder, "adult.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(data, "adult.data")
        z.write(test, "adult.test")
    return zip_path


@pytest.fixture
def small_xy():
    X, y = make_classification(n_samples=200, n_features=6, random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(6)]), pd.Series(y)


# ---------- prepare_data ----------
def test_prepare_data_sorties(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)          # l'extraction se fait dans le dossier courant
    zip_path = _make_adult_zip(str(tmp_path))
    X_train, X_test, y_train, y_test = prepare_data(zip_path)
    assert len(X_train) == len(y_train) and len(X_test) == len(y_test)
    assert len(X_test) / (len(X_train) + len(X_test)) == pytest.approx(0.25, abs=0.01)
    assert list(X_train.columns) == list(X_test.columns)
    assert not X_train.isnull().any().any()
    assert set(y_train.unique()) <= {0, 1}
    assert "fnlwgt" not in X_train.columns and "education" not in X_train.columns
    assert {"has_capital_gain", "is_us"} <= set(X_train.columns)


def test_prepare_data_entrees_invalides(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_data("inexistant.zip")
    with pytest.raises(ValueError):
        prepare_data(str(tmp_path), test_size=1.5)


# ---------- train_model ----------
@pytest.mark.parametrize("model_type", ["gb", "knn"])
def test_train_model(small_xy, model_type):
    X, y = small_xy
    model = train_model(X, y, model_type=model_type)
    assert len(model.predict(X)) == len(X)


def test_train_model_entrees_invalides(small_xy):
    X, y = small_xy
    with pytest.raises(ValueError):
        train_model(X, y, model_type="inconnu")
    with pytest.raises(ValueError):
        train_model(X, y.iloc[:10])            # tailles différentes
    with pytest.raises(ValueError):
        train_model(X, y, model_type="knn", k=0)


# ---------- evaluate_model ----------
def test_evaluate_model(small_xy):
    X, y = small_xy
    res = evaluate_model(train_model(X, y), X, y)
    assert set(res) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0 <= v <= 1 for v in res.values())


def test_evaluate_model_entrees_invalides(small_xy):
    X, y = small_xy
    with pytest.raises(ValueError):
        evaluate_model(train_model(X, y), X, y.iloc[:5])


# ---------- predict_model ----------
def test_predict_model(small_xy):
    X, y = small_xy
    preds = predict_model(train_model(X, y), X.head(10))
    assert len(preds) == 10 and set(preds) <= {0, 1}
    with pytest.raises(ValueError):
        predict_model(train_model(X, y), X.iloc[0:0])


# ---------- save_model / load_model ----------
def test_save_puis_load(small_xy, tmp_path):
    X, y = small_xy
    model = train_model(X, y)
    path = str(tmp_path / "m.joblib")
    save_model(model, path)
    assert os.path.exists(path)
    assert (load_model(path).predict(X) == model.predict(X)).all()


def test_save_load_erreurs(tmp_path):
    with pytest.raises(TypeError):
        save_model(None, str(tmp_path / "m.joblib"))
    with pytest.raises(FileNotFoundError):
        load_model(str(tmp_path / "absent.joblib"))


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-v", __file__]))
