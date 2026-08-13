"""
serve.py – Stellt das trainierte Modell als REST-API bereit.

Was passiert hier?
  1. Das zuletzt in MLflow registrierte Modell wird geladen
  2. Ein Flask-Webserver wird gestartet
  3. Unter /predict können JSON-Anfragen mit Iris-Features beantwortet werden
  4. Unter /health können Load-Balancer den Container-Status prüfen

Flask vs. FastAPI:
  Flask ist einfacher zu verstehen (kein async/await).
  FastAPI wäre in Produktion vorzuziehen (automatische OpenAPI-Docs, Validierung).
  Für dieses Beispiel reicht Flask.

Warum Modell aus MLflow laden statt von Disk?
  Die MLflow Model Registry ist die "Single Source of Truth".
  So weiß immer jeder, welche Version im Einsatz ist,
  wann sie registriert wurde und was ihre Metriken waren.
"""

import logging
import os

import mlflow.sklearn
import numpy as np
from flask import Flask, jsonify, request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("serve")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Modell beim Start laden (einmalig, nicht bei jeder Anfrage)
# Warum global?
#   Das Laden kostet Zeit (~Sekunden). Bei 1000 Anfragen/Minute wäre
#   pro-Request-Loading ein Flaschenhals. Einmal laden, n-mal nutzen.
# ---------------------------------------------------------------------------
MODEL = None
IRIS_CLASSES = ["setosa", "versicolor", "virginica"]


def load_model_from_registry():
    """
    Lädt die neueste Produktionsversion des Modells aus der MLflow Model Registry.

    models:/IrisClassifier/Production:
      - "IrisClassifier" = Name des registrierten Modells
      - "Production" = Stage (Staging → Production → Archived)

    Alternativ: models:/IrisClassifier/1 (konkrete Versionsnummer)
    """
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)

    model_uri  = os.getenv("MODEL_URI", "models:/IrisClassifier/Production")
    logger.info("Lade Modell aus MLflow: %s", model_uri)

    model = mlflow.sklearn.load_model(model_uri)
    logger.info("Modell erfolgreich geladen: %s", type(model).__name__)
    return model


@app.before_request
def startup():
    """
    Flask lädt das Modell beim ersten Request (Lazy Loading).
    before_request wird vor jedem Request ausgeführt.
    Der global-Check stellt sicher, dass nur einmal geladen wird.
    """
    global MODEL
    if MODEL is None:
        MODEL = load_model_from_registry()


@app.route("/health", methods=["GET"])
def health():
    """
    Health-Endpoint für Docker/Kubernetes-Liveness-Probes.

    Kubernetes fragt periodisch: "Lebt der Container noch?"
    Antwortet dieser Endpoint nicht mit 200, wird der Container neu gestartet.

    Wir prüfen, ob das Modell geladen ist – ist es das nicht,
    ist der Container zwar "lebendig" aber nicht "bereit" (ready).
    """
    if MODEL is None:
        return jsonify({"status": "not ready", "model_loaded": False}), 503
    return jsonify({"status": "ok", "model_loaded": True}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Vorhersage-Endpoint.

    Erwartet JSON-Body:
      {
        "features": [[5.1, 3.5, 1.4, 0.2]]   ← Liste von Samples (2D)
      }

    Antwortet mit:
      {
        "predictions": ["setosa"],
        "probabilities": [[0.97, 0.02, 0.01]]
      }

    Warum 2D-Liste?
      Ein Modell kann mehrere Samples gleichzeitig vorhersagen (Batch).
      [[5.1, 3.5, 1.4, 0.2], [6.3, 2.8, 5.1, 1.5]] → 2 Vorhersagen in einem Request.
    """
    if MODEL is None:
        return jsonify({"error": "Modell noch nicht geladen"}), 503

    # Request-Body parsen und validieren
    body = request.get_json(silent=True)
    if body is None or "features" not in body:
        logger.warning("Ungültige Anfrage – kein 'features'-Feld im Body")
        return jsonify({"error": "Body muss 'features' als 2D-Array enthalten"}), 400

    try:
        X = np.array(body["features"], dtype=float)
        if X.ndim != 2 or X.shape[1] != 4:
            raise ValueError(f"Erwartet Shape (N, 4), erhalten: {X.shape}")
    except (ValueError, TypeError) as exc:
        logger.warning("Feature-Parsing fehlgeschlagen: %s", exc)
        return jsonify({"error": str(exc)}), 422

    # Vorhersage
    y_pred   = MODEL.predict(X)
    y_proba  = MODEL.predict_proba(X)

    predictions   = [IRIS_CLASSES[p] for p in y_pred]
    probabilities = y_proba.round(4).tolist()

    logger.info("Vorhersage für %d Sample(s): %s", len(predictions), predictions)

    return jsonify({
        "predictions":   predictions,
        "probabilities": probabilities,
    }), 200


if __name__ == "__main__":
    # debug=False in Produktion – niemals debug=True in einem Container!
    # Warum? debug=True aktiviert den Werkzeug-Debugger, der beliebigen
    # Code im Browser ausführen lässt → Sicherheitslücke.
    port = int(os.getenv("PORT", "8080"))
    logger.info("Starte Flask-Server auf Port %d …", port)
    app.run(host="0.0.0.0", port=port, debug=False)
