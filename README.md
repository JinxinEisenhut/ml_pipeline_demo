# ML Pipeline Demo – Docker · Jenkins · MLflow

Ein vollständiges Beispielprojekt für eine End-to-End ML-Pipeline
mit dem Iris-Klassifikations-Datensatz.

## Projektstruktur

```
ml_pipeline_demo/
├── src/
│   ├── train.py        # Trainings-Skript mit MLflow-Tracking
│   └── serve.py        # Flask REST-API für Inferenz
├── tests/
│   └── test_train.py   # Unit-Tests (pytest)
├── docker/
│   ├── Dockerfile.train     # Container für Training
│   ├── Dockerfile.serve     # Container für Inferenz
│   └── docker-compose.yml   # Lokale Entwicklungsumgebung
├── jenkins/
│   └── Jenkinsfile          # CI/CD-Pipeline als Code
└── requirements.txt         # Python-Abhängigkeiten (fixierte Versionen)
```

## Schnellstart (lokal ohne Jenkins)

### 1. Python-Umgebung einrichten
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. MLflow-Server starten (eigenes Terminal)
```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000
# UI erreichbar unter: http://localhost:5000
```

### 3. Training starten
```bash
# Standard-Hyperparameter
python src/train.py

# Mit eigenen Hyperparametern
N_ESTIMATORS=200 MAX_DEPTH=5 python src/train.py
```

### 4. Tests ausführen
```bash
pytest tests/ -v --tb=short
```

### 5. Serving-Server starten (nach Training)
```bash
# Modell muss zuerst in MLflow auf "Production" gesetzt werden!
# MLflow UI → Models → IrisClassifier → Version X → "Transition to Production"
python src/serve.py
```

### 6. API testen
```bash
# Einzelnes Sample
curl -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [[5.1, 3.5, 1.4, 0.2]]}'

# Batch (mehrere Samples gleichzeitig)
curl -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [[5.1, 3.5, 1.4, 0.2], [6.3, 2.8, 5.1, 1.5]]}'

# Health-Check
curl http://localhost:8080/health
```

## Mit Docker Compose (empfohlen für vollständige Pipeline)

```bash
# Alles starten: MLflow + Training + Serving
cd docker
docker compose up --build

# Nur MLflow starten (für manuelle Training-Runs)
docker compose up mlflow

# Einzelnen Service neu bauen
docker compose build training
docker compose up training
```

**Services nach docker compose up:**
- MLflow UI:      http://localhost:5000
- Inferenz-API:   http://localhost:8080/predict
- Health-Check:   http://localhost:8080/health

## MLflow-Konzepte im Überblick

| Konzept         | Erklärung                                              |
|-----------------|--------------------------------------------------------|
| **Experiment**  | Gruppe von Runs (hier: "iris-classification")          |
| **Run**         | Eine Trainings-Ausführung mit eigener ID               |
| **Parameter**   | Einstellungen vor dem Training (n_estimators etc.)     |
| **Metric**      | Messwerte nach dem Training (accuracy, f1_macro)       |
| **Artifact**    | Gespeicherte Dateien (Modell, Plots, Datensätze)       |
| **Model Stage** | Lebenszyklus: None → Staging → Production → Archived   |

## Jenkins-Pipeline-Ablauf

```
git push
    ↓
Jenkins Webhook
    ↓
[Checkout]   → Neuesten Code holen
[Setup]      → Python venv + pip install
[Tests]      → pytest (schlägt hier fehl → Pipeline stoppt)
[Build]      → docker build Dockerfile.train
[Training]   → docker run (MLflow trackt alles)
              ↓ Accuracy >= 0.90?
              Ja → weiter
              Nein → Pipeline fehlgeschlagen (Exit 1)
[Serve Build] → docker build Dockerfile.serve
[Push]        → Images in Registry (nur main-Branch)
[Deploy]      → Kubernetes/ECS update (nur production)
```

## Umgebungsvariablen

| Variable              | Default              | Beschreibung                          |
|-----------------------|----------------------|---------------------------------------|
| `MLFLOW_TRACKING_URI` | http://localhost:5000| MLflow-Server-URL                     |
| `N_ESTIMATORS`        | 100                  | Anzahl Bäume im RandomForest          |
| `MAX_DEPTH`           | None                 | Maximale Baumtiefe (None = unbegrenzt)|
| `RANDOM_STATE`        | 42                   | Zufallsseed für Reproduzierbarkeit    |
| `TEST_SIZE`           | 0.2                  | Anteil der Testdaten (0.0–1.0)        |
| `MIN_ACCURACY`        | 0.90                 | Schwellwert für Deployment            |
| `MODEL_URI`           | models:/IrisClassifier/Production | Modell-URI für Serving |
| `PORT`                | 8080                 | Flask-Server-Port                     |
