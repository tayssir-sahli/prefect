"""Pipeline ML modulaire - Adult Census Income (prédire income > 50K).

Fonctions : prepare_data, train_model, evaluate_model, save_model, load_model.
Le prétraitement reproduit celui du notebook Colab.
"""

import os
import zipfile

import joblib
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]
CAT_TO_ENCODE = [
    "workclass",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "age_group",
]
NUMERIC_TO_SCALE = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]


def _check_xy(X, y):
    """Vérifie que X et y sont non vides, de même taille et sans valeur manquante."""
    if len(X) == 0:
        raise ValueError("X est vide")
    if len(X) != len(y):
        raise ValueError(
            f"X ({len(X)} lignes) et y ({len(y)}) n'ont pas la même taille"
        )
    if pd.DataFrame(X).isnull().any().any() or pd.Series(y).isnull().any():
        raise ValueError("X ou y contient des valeurs manquantes")


def _find_file(folder, name):
    """Cherche récursivement un fichier dans un dossier (None s'il est absent)."""
    for root, _, files in os.walk(folder):
        if name in files:
            return os.path.join(root, name)
    return None


def _load_raw(data_path, extract_dir="adult_data"):
    """Charge adult.data (+ adult.test s'il existe) depuis un zip ou un dossier."""
    if data_path.endswith(".zip"):
        if not os.path.exists(extract_dir):
            with zipfile.ZipFile(data_path, "r") as zf:
                zf.extractall(extract_dir)
        folder = extract_dir
    else:
        folder = data_path

    train_file = _find_file(folder, "adult.data")
    if train_file is None:
        raise FileNotFoundError(f"adult.data introuvable dans '{folder}'")

    read_args = dict(
        header=None, names=COLUMNS, sep=r",\s*", engine="python", na_values="?"
    )
    frames = [pd.read_csv(train_file, **read_args)]

    test_file = _find_file(folder, "adult.test")
    if test_file is not None:
        df_test = pd.read_csv(test_file, skiprows=1, **read_args)
        # adult.test écrit '>50K.' au lieu de '>50K' : on retire le point final
        df_test["income"] = df_test["income"].str.replace(".", "", regex=False)
        frames.append(df_test)

    return pd.concat(frames, ignore_index=True)


def prepare_data(data_path="adult.zip", test_size=0.25, random_state=RANDOM_STATE):
    """Charge et prétraite le dataset Adult, puis sépare train/test.

    Étapes (comme dans le notebook) : imputation par le mode, suppression de
    fnlwgt et education, indicateurs has_capital_gain/loss, plafonnement de
    capital_gain, age_group, is_us, one-hot encoding, standardisation.
    Le StandardScaler est ajusté sur le train uniquement (pas de fuite de données).

    Args:
        data_path (str): chemin de adult.zip ou d'un dossier contenant adult.data.
        test_size (float): proportion du jeu de test.
        random_state (int): graine aléatoire.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # Vérification des entrées
    if not isinstance(data_path, str) or not os.path.exists(data_path):
        raise FileNotFoundError(f"Chemin introuvable : {data_path!r}")
    if not 0 < test_size < 1:
        raise ValueError("test_size doit être strictement entre 0 et 1")

    df = _load_raw(data_path)

    # 1) Valeurs manquantes : mode pour les 3 colonnes catégorielles
    for col in ["workclass", "occupation", "native_country"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 2) Colonnes redondantes / non explicatives
    df = df.drop(columns=["fnlwgt", "education"])

    # 3) Valeurs aberrantes : indicateurs binaires + plafonnement de capital_gain
    df["has_capital_gain"] = (df["capital_gain"] > 0).astype(int)
    df["has_capital_loss"] = (df["capital_loss"] > 0).astype(int)
    cap_99 = df.loc[df["capital_gain"] < 99999, "capital_gain"].quantile(0.99)
    df["capital_gain"] = df["capital_gain"].clip(upper=cap_99)

    # 4) Feature engineering
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["<=25", "26-35", "36-45", "46-55", "56-65", "65+"],
    )
    df["is_us"] = (df["native_country"] == "United-States").astype(int)
    df = df.drop(columns=["native_country"])

    # 5) Cible binaire + encodage one-hot
    df["income_binary"] = (df["income"] == ">50K").astype(int)
    df = pd.get_dummies(df, columns=CAT_TO_ENCODE, drop_first=True)
    df = df.drop(columns=["income"])

    X = df.drop(columns=["income_binary"])
    y = df["income_binary"]

    # 6) Séparation train/test (stratifiée), puis normalisation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[NUMERIC_TO_SCALE] = scaler.fit_transform(X_train[NUMERIC_TO_SCALE])
    X_test[NUMERIC_TO_SCALE] = scaler.transform(X_test[NUMERIC_TO_SCALE])

    # Vérification des sorties
    if X_train.isnull().any().any() or X_test.isnull().any().any():
        raise ValueError("Des valeurs manquantes subsistent après le prétraitement")
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Colonnes différentes entre train et test")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, model_type="gb", k=5, random_state=RANDOM_STATE):
    """Entraîne un modèle et le retourne.

    Args:
        model_type (str): "gb" (Gradient Boosting) ou "knn".
        k (int): nombre de voisins si model_type="knn".
    """
    _check_xy(X_train, y_train)
    if model_type == "knn" and (not isinstance(k, int) or k < 1):
        raise ValueError("k doit être un entier >= 1")

    if model_type == "gb":
        model = GradientBoostingClassifier(random_state=random_state)
    elif model_type == "knn":
        model = KNeighborsClassifier(n_neighbors=k)
    else:
        raise ValueError("model_type doit valoir 'gb' ou 'knn'")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Évalue le modèle (accuracy, précision, rappel, F1, ROC-AUC, confusion).

    Returns:
        dict: métriques calculées.
    """
    _check_xy(X_test, y_test)
    if not hasattr(model, "predict_proba"):
        raise TypeError("Le modèle doit avoir une méthode predict_proba")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results = {
        "accuracy": metrics.accuracy_score(y_test, y_pred),
        "precision": metrics.precision_score(y_test, y_pred),
        "recall": metrics.recall_score(y_test, y_pred),
        "f1": metrics.f1_score(y_test, y_pred),
        "roc_auc": metrics.roc_auc_score(y_test, y_proba),
    }
    for name, value in results.items():
        print(f"{name:10s}: {value:.3f}")
    print("\nMatrice de confusion :")
    print(metrics.confusion_matrix(y_test, y_pred))
    print("\nRapport de classification :")
    print(metrics.classification_report(y_test, y_pred))
    return results


def predict_model(model, X):
    """Prédit la classe (0 = <=50K, 1 = >50K) pour chaque ligne de X.

    Returns:
        numpy.ndarray: prédictions, une par ligne de X.
    """
    if len(X) == 0:
        raise ValueError("X est vide")
    return model.predict(X)


def save_model(model, path="model.joblib"):
    """Sauvegarde le modèle entraîné avec joblib."""
    if model is None or not hasattr(model, "predict"):
        raise TypeError("save_model attend un modèle entraîné (avec predict)")
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        raise FileNotFoundError(f"Dossier inexistant : {folder}")
    joblib.dump(model, path)
    print(f"Modèle sauvegardé dans {path}")


def load_model(path="model.joblib"):
    """Charge un modèle sauvegardé avec joblib."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle introuvable : {path} (lancez d'abord --train)")
    model = joblib.load(path)
    print(f"Modèle chargé depuis {path}")
    return model
