"""Point d'entrée : exécute les étapes du pipeline via la ligne de commande.

Exemples :
    python main.py --prepare
    python main.py --train
    python main.py --train --model-type knn --k 11 --model knn.joblib
    python main.py --evaluate
    python main.py --prepare --train --evaluate
"""

import argparse

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


def main():
    parser = argparse.ArgumentParser(description="Pipeline ML - Adult Census Income")
    parser.add_argument(
        "--data", default="adult.zip", help="adult.zip ou dossier contenant adult.data"
    )
    parser.add_argument(
        "--model", default="model.joblib", help="Chemin du modèle sauvegardé"
    )
    parser.add_argument(
        "--model-type", default="gb", choices=["gb", "knn"], help="gb ou knn"
    )
    parser.add_argument("--k", type=int, default=5, help="Nombre de voisins (KNN)")
    parser.add_argument("--prepare", action="store_true", help="Préparer les données")
    parser.add_argument(
        "--train", action="store_true", help="Entraîner et sauvegarder le modèle"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Évaluer le modèle sauvegardé"
    )
    args = parser.parse_args()

    if not (args.prepare or args.train or args.evaluate):
        parser.error(
            "Choisissez au moins une action : --prepare, --train ou --evaluate"
        )

    X_train, X_test, y_train, y_test = prepare_data(args.data)

    if args.prepare:
        print(f"Données prêtes : train={X_train.shape}, test={X_test.shape}")
        print(f"Proportion >50K : train={y_train.mean():.3f}, test={y_test.mean():.3f}")

    if args.train:
        model = train_model(X_train, y_train, model_type=args.model_type, k=args.k)
        save_model(model, args.model)

    if args.evaluate:
        model = load_model(args.model)
        evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
