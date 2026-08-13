# SETUP.md – Jenkins von Null bis zur laufenden Pipeline

Diese Anleitung richtet alles ein was gebraucht wird:
- Lokale Docker-Registry (Speicher für Images)
- MLflow-Server (Experiment-Tracking)
- GitHub-Repository (Code-Hosting)
- Jenkins (CI/CD-Server)
- Webhook-Verbindung zwischen GitHub und Jenkins

---

## Voraussetzungen

Auf dem Rechner muss installiert sein:
- Docker Desktop (oder Docker Engine + Docker Compose)
- Git
- Python 3.11+
- Ein GitHub-Account (kostenlos)

Überprüfen:
```bash
docker --version      # Docker version 24.x oder neuer
git --version         # git version 2.x
python3 --version     # Python 3.11.x oder 3.12.x
```

---

## Schritt 1: Lokale Docker-Registry starten

Eine Docker-Registry ist ein lokaler Server der Docker-Images speichert.
Ohne Registry müsste man Images manuell per `docker save/load` übertragen.

```bash
# Registry auf Port 5050 starten (5000 ist oft von MLflow belegt)
docker run -d --name local-registry --restart unless-stopped -p 5050:5000 registry:2

# Prüfen ob sie läuft
curl http://localhost:5050/v2/_catalog
# Erwartete Ausgabe: {"repositories":[]}
```

Docker muss die lokale Registry als "insecure" kennen (kein HTTPS).
Datei öffnen oder anlegen:
- Linux:   `/etc/docker/daemon.json`
- Mac:     Docker Desktop → Settings → Docker Engine
- Windows: Docker Desktop → Settings → Docker Engine

Inhalt:
```json
{
  "insecure-registries": ["localhost:5050"]
}
```

Docker danach neu starten (Docker Desktop: Restart; Linux: `sudo systemctl restart docker`).

---

## Schritt 2: MLflow-Server starten

```bash
# Im Projektverzeichnis
cd ml_pipeline_demo

pip install mlflow

mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000
```

MLflow-UI ist jetzt erreichbar unter: http://localhost:5000

Tipp: MLflow in einem separaten Terminal-Fenster laufen lassen
oder als systemd-Service/Docker-Container einrichten (siehe unten).

MLflow als Docker-Container (empfohlen, damit er im Hintergrund läuft):
```bash
docker run -d \
    --name mlflow-server \
    --restart unless-stopped \
    -p 5000:5000 \
    -v $(pwd)/mlflow-data:/mlflow \
    ghcr.io/mlflow/mlflow:v2.10.0 \
    mlflow server \
        --backend-store-uri sqlite:////mlflow/mlflow.db \
        --default-artifact-root /mlflow/artifacts \
        --host 0.0.0.0 \
        --port 5000
```

---

## Schritt 3: Code nach GitHub pushen

### 3a. GitHub-Repository anlegen

1. Auf https://github.com einloggen
2. Grüner Button "New" (oben links)
3. Repository-Name: `ml-pipeline-demo`
4. Visibility: Public (für Webhook ohne Pro-Account nötig)
   ODER Private (dann Jenkins braucht einen Deploy-Key)
5. "Create repository" klicken
6. Die Repository-URL notieren:
   `https://github.com/DEIN-USERNAME/ml-pipeline-demo.git`

### 3b. Lokales Repo mit GitHub verbinden

```bash
cd ml_pipeline_demo

# Remote-URL setzen (DEIN-USERNAME ersetzen!)
git remote add origin https://github.com/DEIN-USERNAME/ml-pipeline-demo.git

# Alle Dateien zum ersten Commit vorbereiten
git add .

# Status prüfen: Was wird committet? Was nicht? (.gitignore greift hier)
git status

# Erster Commit
git commit -m "Initial commit: Iris ML pipeline with Docker, Jenkins, MLflow"

# Auf GitHub pushen
git push -u origin main
```

Nach dem Push sollte man auf GitHub alle Dateien sehen.
.venv/, mlflow.db, __pycache__/ dürfen NICHT sichtbar sein (dank .gitignore).

---

## Schritt 4: Jenkins installieren und starten

### Option A: Jenkins als Docker-Container (empfohlen für lokales Testen)

```bash
# Verzeichnis für Jenkins-Daten (bleibt bei Container-Neustart erhalten)
mkdir -p ~/jenkins-data

docker run -d \
    --name jenkins \
    --restart unless-stopped \
    -p 8080:8080 \
    -p 50000:50000 \
    -v ~/jenkins-data:/var/jenkins_home \
    -v /var/run/docker.sock:/var/run/docker.sock \
    jenkins/jenkins:lts-jdk17
```

Wichtig: `-v /var/run/docker.sock:/var/run/docker.sock`
Das gibt dem Jenkins-Container Zugriff auf Docker des Hosts.
Ohne das kann Jenkins keine Docker-Befehle ausführen (docker build, docker run etc.).

### Option B: Jenkins nativ installieren (Linux)

```bash
# Jenkins-Repository hinzufügen
curl -fsSL https://pkg.jenkins.io/debian-stable/jenkins.io-2023.key \
    | sudo tee /usr/share/keyrings/jenkins-keyring.asc > /dev/null
echo "deb [signed-by=/usr/share/keyrings/jenkins-keyring.asc] \
    https://pkg.jenkins.io/debian-stable binary/" \
    | sudo tee /etc/apt/sources.list.d/jenkins.list > /dev/null

sudo apt-get update
sudo apt-get install -y jenkins java-17-openjdk

sudo systemctl enable jenkins
sudo systemctl start jenkins
```

### 4a. Jenkins-Ersteinrichtung

1. http://localhost:8080 im Browser öffnen
2. Initiales Admin-Passwort anzeigen:
   ```bash
   # Docker-Container:
   docker exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword
   # Nativ:
   sudo cat /var/jenkins_home/secrets/initialAdminPassword
   ```
3. Passwort eingeben
4. "Install suggested plugins" wählen (dauert 2-3 Minuten)
5. Admin-Benutzer anlegen (Benutzername + Passwort merken!)
6. Jenkins-URL bestätigen: http://localhost:8080

### 4b. Docker in Jenkins verfügbar machen

Wenn Jenkins als Docker-Container läuft, muss man Docker-CLI installieren:
```bash
docker exec -u root jenkins bash -c "
    apt-get update -qq &&
    apt-get install -y docker.io &&
    usermod -aG docker jenkins
"
docker restart jenkins
```

Test:
```bash
docker exec jenkins docker --version
# Erwartete Ausgabe: Docker version 24.x...
```

---

## Schritt 5: Jenkins konfigurieren

### 5a. Plugins installieren

Jenkins-UI → "Jenkins verwalten" → "Plugins" → "Available plugins"

Folgende Plugins suchen und installieren (falls nicht vorhanden):
- **Pipeline** (meist vorinstalliert)
- **Git** (meist vorinstalliert)
- **Docker Pipeline** ← wichtig für docker.withRegistry() im Jenkinsfile
- **GitHub** ← für Webhook-Integration

Nach Installation: Jenkins neu starten.

### 5b. Credentials anlegen

Jenkins speichert Passwörter und Tokens verschlüsselt im Credential Store.
Das Jenkinsfile referenziert sie per ID, niemals als Klartext.

Jenkins-UI → "Jenkins verwalten" → "Credentials"
→ "(global)" → "Add Credentials"

**Credential 1: MLflow-URL**
- Kind: Secret text
- Secret: `http://localhost:5000`
  (oder die Netzwerk-IP des Hosts wenn Jenkins in Docker läuft:
   `http://172.17.0.1:5000` – IP mit `ip route show default` ermitteln)
- ID: `mlflow-tracking-uri`  ← genau diese ID steht im Jenkinsfile!
- Description: MLflow Tracking Server URL

**Credential 2: Docker-Registry**
- Kind: Username with password
- Username: (leer, lokale Registry braucht kein Login)
- Password: (leer)
- ID: `docker-registry-creds`
- Description: Local Docker Registry

Wenn man eine echte Registry (Docker Hub, GitHub Container Registry) nutzt:
- Username: Docker Hub Benutzername
- Password: Docker Hub Access Token (NICHT das Account-Passwort!)

---

## Schritt 6: Jenkins-Job anlegen

Jenkins-UI → "Neues Element" (linke Seitenleiste)

1. **Name:** `ml-pipeline-iris`
2. **Typ:** "Multibranch Pipeline" wählen
   - Vorteil: Erkennt automatisch alle Branches und Pull Requests im Repo
   - Für jeden Branch mit einem Jenkinsfile wird automatisch ein Job erstellt
3. "OK" klicken

### Job konfigurieren:

**Branch Sources → Add Source → GitHub**

```
Repository HTTPS URL: https://github.com/DEIN-USERNAME/ml-pipeline-demo.git
```

Bei privatem Repository: Credentials hinzufügen
- GitHub-Benutzername + Personal Access Token (nicht Passwort!)
- Token erstellen unter: GitHub → Settings → Developer Settings →
  Personal Access Tokens → "repo" Scope auswählen

**Scan Multibranch Pipeline Triggers:**
- "Periodically if not otherwise run": 5 Minuten
  (SCM Polling als Fallback, falls Webhook nicht funktioniert)

**Speichern** klicken.

Jenkins scannt jetzt das Repository und findet das Jenkinsfile im `jenkins/`-Ordner.
Dabei muss man Jenkins mitteilen wo das Jenkinsfile liegt:

"Properties" → "Pipeline" →
- "Markerfile": leer lassen
- Script Path: `jenkins/Jenkinsfile`

---

## Schritt 7: Webhook einrichten (für sofortigen Trigger bei git push)

Ohne Webhook: Jenkins pollt alle 5 Minuten (pollSCM im Jenkinsfile).
Mit Webhook: Jenkins wird sofort nach jedem Push benachrichtigt (< 1 Sekunde).

**Voraussetzung:** Jenkins muss vom Internet erreichbar sein.
- Bei öffentlicher IP/Domain: direkt nutzbar
- Lokal (ohne öffentliche IP): ngrok als Tunnel nutzen (siehe unten)

### 7a. ngrok für lokales Testing (optional)

ngrok erstellt einen öffentlichen Tunnel zu deinem lokalen Jenkins:

```bash
# ngrok installieren: https://ngrok.com/download
# Kostenlosen Account erstellen, Auth-Token holen

ngrok config add-authtoken DEIN_AUTH_TOKEN
ngrok http 8080
```

ngrok zeigt eine URL wie: `https://abc123.ngrok-free.app`
Diese URL ist temporär und ändert sich beim Neustart.

### 7b. Webhook in GitHub anlegen

1. GitHub → Repository → "Settings" → "Webhooks" → "Add webhook"
2. Payload URL: `https://abc123.ngrok-free.app/github-webhook/`
   (oder deine Jenkins-URL + `/github-webhook/`)
3. Content type: `application/json`
4. Secret: (leer lassen für Testzwecke)
5. Events: "Just the push event"
6. "Add webhook" klicken

GitHub sendet sofort einen Test-Request (mit grünem Haken wenn OK).

### 7c. Jenkins für Webhook-Trigger konfigurieren

Im Jenkins-Job:
"Configure" → "Build Triggers" →
Haken setzen bei: "GitHub hook trigger for GITScm polling"

Ab jetzt: `git push origin main` → GitHub → Webhook → Jenkins startet sofort.

---

## Schritt 8: Ersten Build manuell starten und beobachten

1. Jenkins-UI → Job `ml-pipeline-iris` → Branch `main`
2. "Build with Parameters" klicken
3. Parameter:
   - N_ESTIMATORS: 100
   - MAX_DEPTH: None
   - ENVIRONMENT: staging
4. "Build" klicken

### Build beobachten:

- Linke Seite: Build-Liste mit Nummer (#1, #2, ...)
- Klick auf Build-Nummer → "Console Output"
- Man sieht die Ausgabe in Echtzeit (wie ein Terminal)

Erwarteter Ablauf:
```
[Pipeline] stage (Checkout)
[Pipeline] checkout ... OK
[Pipeline] stage (Setup)
[Pipeline] sh ... Installing packages ... OK
[Pipeline] stage (Tests)
[Pipeline] sh ... 10 passed in 3.73s ... OK
[Pipeline] stage (Docker Build)
[Pipeline] sh ... Successfully built ... OK
[Pipeline] stage (Training)
[Pipeline] sh ... Accuracy: 0.9667 | F1-Score: 0.9667 ... OK
[Pipeline] stage (Build Serving Image)
[Pipeline] sh ... Successfully built ... OK
[Pipeline] stage (Deploy)
[Pipeline] sh ... Healthcheck erfolgreich! ... OK
[Pipeline] End of Pipeline
Finished: SUCCESS
```

---

## Schritt 9: Automatischen Trigger testen

```bash
# Kleine Änderung im Code machen
echo "# Build trigger test" >> src/train.py

# Committen und pushen
git add src/train.py
git commit -m "test: trigger CI pipeline"
git push origin main
```

**Mit Webhook:** Jenkins startet in < 5 Sekunden.
**Mit SCM Polling:** Jenkins startet in max. 5 Minuten.

---

## Häufige Probleme und Lösungen

### "docker: command not found" in Jenkins

Ursache: Docker-CLI nicht im Jenkins-Container installiert.
Lösung: Schritt 4b nochmals ausführen.

### "Cannot connect to MLflow" beim Training

Ursache: MLflow läuft auf localhost:5000, aber der Docker-Container
sieht einen anderen localhost.

Lösung 1 (Docker auf Linux): Host-IP verwenden
```bash
# Host-IP ermitteln
ip route show default | awk '{print $3}'  # z.B. 172.17.0.1
# Credential 'mlflow-tracking-uri' auf http://172.17.0.1:5000 ändern
```

Lösung 2 (Docker Desktop auf Mac/Windows): host.docker.internal verwenden
```
http://host.docker.internal:5000
```

### "denied: requested access to the resource is denied" beim docker push

Ursache: Docker kann sich nicht bei der Registry authentifizieren.

Lösung: insecure-registries korrekt konfiguriert? (Schritt 1)
```bash
docker info | grep -A 5 "Insecure Registries"
# Muss localhost:5050 enthalten
```

### Webhook kommt nicht an (GitHub zeigt rotes X)

Ursache: Jenkins nicht erreichbar von GitHub.
Lösung: ngrok starten (Schritt 7a) und Webhook-URL aktualisieren.

### "Branch does not match" – Pipeline startet nicht

Ursache: Jenkinsfile hat `when { branch 'main' }` für den Deploy-Stage,
aber der Branch heißt im Job `origin/main`.

Lösung: Im Jenkinsfile `branch 'main'` zu `branch '*/main'` ändern,
oder Branch-Quellen im Job korrekt konfigurieren.

---

## Zusammenfassung: Was womit kommuniziert

```
Entwickler
   │ git push
   ▼
GitHub (Code-Repository)
   │ HTTP POST (Webhook) bei git push
   ▼
Jenkins :8080 (CI/CD-Server)
   │ checkout scm (git clone/fetch)
   ▼
Lokales Dateisystem (Workspace)
   │ docker build
   ▼
Lokale Docker-Registry :5050 (Image-Speicher)
   │ docker run (Training-Container)
   ▼
MLflow :5000 (Experiment-Tracking)
   │ Modell registriert
   ▼
Docker-Container iris-serve :8080 (REST-API)
   │ HTTP /predict
   ▼
Endnutzer / Anwendung
```
