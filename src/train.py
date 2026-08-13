"""
train.py – Trainings-Einstiegspunkt der ML-Pipeline.

Was passiert hier?
  1. Daten laden (Iris-Datensatz aus scikit-learn)
  2. Train/Test-Split durchführen
  3. MLflow-Run starten → alles wird protokolliert
  4. Modell trainieren (RandomForest)
  5. Metriken berechnen und in MLflow loggen
  6. Trainiertes Modell als Artefakt in MLflow speichern

MLflow-Konzept:
  Ein "Run" ist eine einzelne Ausführung des Trainings.
  Jeder Run bekommt eine eindeutige ID und speichert:
    - Parameter (was wurde eingestellt?)
    - Metriken (wie gut wurde das Modell?)
    - Artefakte (das Modell selbst, Plots, etc.)
"""

import logging
import os

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Logger einrichten
# Warum logging statt print()?
#   - Loglevel (DEBUG, INFO, WARNING, ERROR) steuerbar
#   - Timestamps automatisch
#   - In Produktions-Containern lassen sich Logs zentral sammeln (z.B. ELK-Stack)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("train")


def load_data():
    """
    Daten laden und in Features (X) und Labels (y) aufteilen.

    Iris-Datensatz:
      - 150 Blumenproben
      - 4 Features: Kelchblatt-/Blütenblatt-Länge und -Breite
      - 3 Klassen: Setosa, Versicolor, Virginica
    """
    logger.info("Lade Iris-Datensatz …")
    iris = load_iris()
    X, y = iris.data, iris.target
    logger.info("Datensatz geladen: %d Samples, %d Features, %d Klassen",
                X.shape[0], X.shape[1], len(np.unique(y)))
    return X, y, iris.target_names


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Datensatz in Trainings- und Testmenge aufteilen.

    random_state=42: Fixierter Zufallswert → reproduzierbare Splits.
    test_size=0.2: 20 % der Daten werden zum Testen zurückgehalten.

    Warum aufteilen?
      Das Modell soll auf *ungesehenen* Daten bewertet werden.
      Ohne Split würde man nur messen, wie gut es auswendig gelernt hat (Overfitting).
    """
    logger.info("Trenne Daten: %.0f%% Training / %.0f%% Test …",
                (1 - test_size) * 100, test_size * 100)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(X_train, y_train, n_estimators, max_depth, random_state):
    """
    RandomForestClassifier trainieren.

    RandomForest:
      - Trainiert viele Entscheidungsbäume (n_estimators) auf zufälligen Datensatz-Teilmengen
      - Jeder Baum trifft eine Vorhersage; das Ergebnis ist der Mehrheitsentscheid
      - Robust gegen Overfitting, gut für tabellarische Daten

    max_depth: Maximale Tiefe jedes Baumes.
      Zu tief → Overfitting. Zu flach → Underfitting.
    """
    logger.info("Trainiere RandomForest: n_estimators=%d, max_depth=%s",
                n_estimators, max_depth)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    logger.info("Training abgeschlossen.")
    return model


def evaluate_model(model, X_test, y_test, target_names):
    """
    Modell auf dem Testset auswerten und Metriken berechnen.

    accuracy_score: Anteil korrekt klassifizierter Samples (simpelste Metrik).
    f1_score (macro): Harmonisches Mittel aus Precision und Recall,
                      gemittelt über alle Klassen.
                      Besser bei unbalancierten Datensätzen als reine Accuracy.

    Gibt ein Dict zurück, damit die Metriken strukturiert an MLflow übergeben werden.
    """
    logger.info("Bewerte Modell auf Testset …")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, average="macro")

    logger.info("Accuracy: %.4f | F1-Score (macro): %.4f", accuracy, f1)
    logger.info("\n%s", classification_report(y_test, y_pred, target_names=target_names))

    return {"accuracy": accuracy, "f1_macro": f1}


def run_pipeline():
    """
    Hauptfunktion – orchestriert den gesamten Trainings-Ablauf.

    Warum alles in einer Funktion statt global?
      - Testbar (Unit-Tests können run_pipeline() mocken oder isoliert aufrufen)
      - Kein unbeabsichtigter Seiteneffekt beim Import des Moduls

    MLflow-Run als Context-Manager (with-Block):
      - Beim Betreten: neuer Run wird gestartet, Run-ID vergeben
      - Beim Verlassen (auch bei Exceptions): Run wird sauber abgeschlossen
      - Kein manuelles mlflow.end_run() nötig → kein Risiko eines offenen Runs
    """
    # Hyperparameter aus Umgebungsvariablen lesen.
    # Warum Umgebungsvariablen?
    #   Docker und Jenkins können Werte von außen injizieren,
    #   ohne den Code zu ändern. Ideal für Grid-Search oder CI-Overrides.
    n_estimators  = int(os.getenv("N_ESTIMATORS",  "100"))
    max_depth_raw = os.getenv("MAX_DEPTH", "None")
    max_depth     = None if max_depth_raw == "None" else int(max_depth_raw)
    random_state  = int(os.getenv("RANDOM_STATE", "42"))
    test_size     = float(os.getenv("TEST_SIZE",   "0.2"))

    logger.info("=== Starte ML-Pipeline-Run ===")
    logger.info("Hyperparameter: n_estimators=%d, max_depth=%s, random_state=%d, test_size=%.2f",
                n_estimators, max_depth, random_state, test_size)

    # MLflow Tracking-Server-URL (standardmäßig lokal auf Port 5000).
    # In Docker wird diese URL als Umgebungsvariable übergeben.
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("iris-classification")  # Experiment gruppiert mehrere Runs

    # Daten laden und aufteilen
    X, y, target_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)

    # MLflow-Run starten – ab hier wird alles protokolliert
    with mlflow.start_run() as run:
        logger.info("MLflow Run-ID: %s", run.info.run_id)

        # ── Parameter loggen (Einstellungen, die das Training steuern) ──────
        mlflow.log_params({
            "n_estimators":  n_estimators,
            "max_depth":     str(max_depth),
            "random_state":  random_state,
            "test_size":     test_size,
            "model_type":    "RandomForestClassifier",
        })

        # ── Modell trainieren ────────────────────────────────────────────────
        model = train_model(X_train, y_train, n_estimators, max_depth, random_state)

        # ── Metriken loggen (Ergebnisse der Evaluation) ──────────────────────
        metrics = evaluate_model(model, X_test, y_test, target_names)
        mlflow.log_metrics(metrics)

        # ── Modell als Artefakt speichern ────────────────────────────────────
        # mlflow.sklearn.log_model speichert:
        #   - das serialisierte Modell (pickle)
        #   - MLmodel-Metadatei (Format, Python-Version, Requirements)
        #   - conda.yaml / requirements.txt für Reproduzierbarkeit
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_model",
            registered_model_name="IrisClassifier",  # in der Model Registry registrieren
        )
        logger.info("Modell in MLflow gespeichert: artifact_path='random_forest_model'")

        # Schwellwert-Check: Pipeline schlägt fehl, wenn Qualität nicht ausreicht.
        # Jenkins wertet den Exit-Code aus – Exit 1 = Stage fehlgeschlagen.
        min_accuracy = float(os.getenv("MIN_ACCURACY", "0.90"))
        if metrics["accuracy"] < min_accuracy:
            logger.error(
                "Accuracy %.4f unter Mindestschwelle %.4f → Deployment abgebrochen!",
                metrics["accuracy"], min_accuracy
            )
            raise SystemExit(1)  # Jenkins-Pipeline wird gestoppt

        logger.info("=== Pipeline-Run erfolgreich abgeschlossen ===")


if __name__ == "__main__":
    run_pipeline()
