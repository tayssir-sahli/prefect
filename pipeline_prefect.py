"""Pipeline Prefect : orchestration des étapes du projet ML (Atelier 3).

Flows : install, code, train, evaluate, predict, all, clean.

Lancement :
    python pipeline_prefect.py --flow all
    python pipeline_prefect.py --flow train          (alias : entrainement)
    python pipeline_prefect.py --flow evaluate
    python pipeline_prefect.py --flow code
"""

import argparse
import os
import shutil
import subprocess  # noqa: S404 (appels d'outils locaux, sans shell)
import sys

from prefect import flow, get_run_logger, task

from model_pipeline import (
    evaluate_model,
    load_model,
    predict_model,
    prepare_data,
    save_model,
    train_model,
)

# Les étapes qualité/sécurité ne ciblent que ces fichiers (pas venv/, adult_data/...)
PROJECT_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]
DATA_PATH = "adult.zip"
MODEL_PATH = "model.joblib"


def _run(cmd):
    """Exécute une commande (liste d'arguments, sans shell) et journalise la sortie.

    Raises:
        RuntimeError: si la commande retourne un code d'erreur.
    """
    logger = get_run_logger()
    logger.info("Commande : %s", " ".join(cmd))
    result = subprocess.run(  # arguments fixes, sans shell, pas d'entrée utilisateur
        cmd, capture_output=True, text=True, check=False
    )
    output = (result.stdout + result.stderr).strip()
    if output:
        logger.info("\n%s", output)
    if result.returncode != 0:
        raise RuntimeError(f"Échec ({result.returncode}) : {' '.join(cmd)}")
    return output


# ----------------------------------------------------------------- TASKS
@task(name="Récupérer le projet (git)")
def fetch_project(repo_url, dest="."):
    """Clone le dépôt git, ou fait un `git pull` s'il est déjà présent."""
    if os.path.isdir(os.path.join(dest, ".git")):
        _run(["git", "-C", dest, "pull"])
    else:
        _run(["git", "clone", repo_url, dest])


@task(name="Installer les dépendances")
def install_dependencies():
    """Installe les paquets de requirements.txt dans l'environnement actif."""
    _run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="Formatage du code (black)")
def format_code():
    """Formate les fichiers du projet avec black."""
    _run([sys.executable, "-m", "black", *PROJECT_FILES])


@task(name="Qualité du code (flake8)")
def check_quality():
    """Analyse le style et les erreurs avec flake8 (compatible black)."""
    _run(
        [
            sys.executable,
            "-m",
            "flake8",
            "--max-line-length=88",
            "--extend-ignore=E203,W503",
            *PROJECT_FILES,
        ]
    )


@task(name="Sécurité du code (bandit)")
def check_security():
    """Cherche des failles courantes avec bandit (sévérité moyenne et plus)."""
    _run([sys.executable, "-m", "bandit", "-ll", "-q", *PROJECT_FILES])


@task(name="Tests unitaires (pytest)")
def run_unit_tests():
    """Exécute les tests unitaires de test_pipeline.py."""
    _run([sys.executable, "-m", "pytest", "-q", "test_pipeline.py"])


@task(name="Préparation des données")
def prepare_task(data_path=DATA_PATH):
    """Charge et prétraite les données (appelle prepare_data)."""
    logger = get_run_logger()
    X_train, X_test, y_train, y_test = prepare_data(data_path)
    logger.info("train=%s, test=%s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


@task(name="Entraînement du modèle")
def train_task(X_train, y_train, model_type="gb"):
    """Entraîne le modèle (appelle train_model)."""
    return train_model(X_train, y_train, model_type=model_type)


@task(name="Sauvegarde du modèle")
def save_task(model, path=MODEL_PATH):
    """Sauvegarde le modèle (appelle save_model)."""
    save_model(model, path)


@task(name="Chargement du modèle")
def load_task(path=MODEL_PATH):
    """Charge le modèle sauvegardé (appelle load_model)."""
    return load_model(path)


@task(name="Évaluation du modèle")
def evaluate_task(model, X_test, y_test):
    """Évalue le modèle (appelle evaluate_model) et retourne les métriques."""
    results = evaluate_model(model, X_test, y_test)
    get_run_logger().info("Métriques : %s", results)
    return results


@task(name="Prédiction")
def predict_task(model, X_test, y_test, n=5):
    """Prédit les n premières lignes du jeu de test et compare à la réalité."""
    preds = predict_model(model, X_test.head(n))
    get_run_logger().info(
        "Prédictions : %s | Réel : %s",
        [int(v) for v in preds],
        [int(v) for v in y_test.head(n)],
    )
    return preds


@task(name="Nettoyage")
def clean_task():
    """Supprime les fichiers générés (modèle, données extraites, caches)."""
    for path in [MODEL_PATH, "adult_data", ".pytest_cache", "__pycache__"]:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)


# ----------------------------------------------------------------- FLOWS
@flow(name="install")
def install_flow():
    """Flow : installation des dépendances."""
    install_dependencies()


@flow(name="code")
def code_flow():
    """Flow : formatage, qualité, sécurité et tests unitaires."""
    format_code()
    check_quality()
    check_security()
    run_unit_tests()


@flow(name="train")
def train_flow(data_path=DATA_PATH, model_type="gb"):
    """Flow : préparation des données, entraînement et sauvegarde du modèle."""
    X_train, _, y_train, _ = prepare_task(data_path)
    model = train_task(X_train, y_train, model_type)
    save_task(model)


@flow(name="evaluate")
def evaluate_flow(data_path=DATA_PATH):
    """Flow : chargement du modèle et évaluation (le jeu de test vient de prepare)."""
    _, X_test, _, y_test = prepare_task(data_path)
    model = load_task()
    return evaluate_task(model, X_test, y_test)


@flow(name="predict")
def predict_flow(data_path=DATA_PATH):
    """Flow : chargement du modèle et prédiction sur quelques lignes de test."""
    _, X_test, _, y_test = prepare_task(data_path)
    model = load_task()
    return predict_task(model, X_test, y_test)


@flow(name="all")
def all_flow(data_path=DATA_PATH, model_type="gb", repo_url=None):
    """Flow complet : (git), dépendances, code, données, modèle, évaluation."""
    if repo_url:
        fetch_project(repo_url)
    install_dependencies()
    format_code()
    check_quality()
    check_security()
    run_unit_tests()
    X_train, X_test, y_train, y_test = prepare_task(data_path)
    model = train_task(X_train, y_train, model_type)
    save_task(model)
    return evaluate_task(model, X_test, y_test)


@flow(name="clean")
def clean_flow():
    """Flow : suppression des fichiers générés."""
    clean_task()


FLOWS = {
    "all": all_flow,
    "install": install_flow,
    "code": code_flow,
    "train": train_flow,
    "entrainement": train_flow,
    "evaluate": evaluate_flow,
    "predict": predict_flow,
    "clean": clean_flow,
}


def main():
    """Point d'entrée CLI : lance le flow demandé."""
    parser = argparse.ArgumentParser(description="Pipeline Prefect")
    parser.add_argument("--flow", required=True, choices=FLOWS.keys())
    parser.add_argument("--data", default=DATA_PATH, help="adult.zip ou dossier")
    parser.add_argument("--model-type", default="gb", choices=["gb", "knn"])
    args = parser.parse_args()

    kwargs = {}
    if args.flow in ("all", "train", "entrainement"):
        kwargs["model_type"] = args.model_type
    if args.flow not in ("install", "code", "clean"):
        kwargs["data_path"] = args.data
    FLOWS[args.flow](**kwargs)


if __name__ == "__main__":
    main()
