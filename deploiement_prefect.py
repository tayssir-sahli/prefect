"""Déploiement Prefect : planifie les flows (exécution quotidienne).

Usage (voir les 3 terminaux de l'atelier) :
    Terminal 1 : prefect server start
    Terminal 2 : prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
                 python deploiement_prefect.py
    Terminal 3 : prefect deployment ls
                 prefect deployment run 'all/ml-pipeline-all'
"""

from prefect import serve

from pipeline_prefect import (
    all_flow,
    clean_flow,
    code_flow,
    evaluate_flow,
    install_flow,
    train_flow,
)

if __name__ == "__main__":
    deployments = [
        # Pipeline complet : tous les jours à 02:00
        all_flow.to_deployment(
            name="ml-pipeline-all",
            cron="0 2 * * *",
            tags=["mlops", "full-pipeline"],
        ),
        # Ré-entraînement : tous les jours à 03:00
        train_flow.to_deployment(
            name="ml-pipeline-train",
            cron="0 3 * * *",
            tags=["mlops", "training"],
        ),
        # Qualité du code : tous les jours à 01:00
        code_flow.to_deployment(
            name="ml-pipeline-code",
            cron="0 1 * * *",
            tags=["mlops", "code"],
        ),
        # Déploiements sans planification (lancement manuel)
        evaluate_flow.to_deployment(
            name="ml-pipeline-evaluate", tags=["mlops", "evaluation"]
        ),
        install_flow.to_deployment(name="ml-pipeline-install", tags=["mlops", "setup"]),
        clean_flow.to_deployment(name="ml-pipeline-clean", tags=["mlops", "maintenance"]),
    ]
    serve(*deployments)
