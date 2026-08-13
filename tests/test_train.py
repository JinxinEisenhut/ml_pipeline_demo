"""
test_train.py – Unit-Tests für die Trainings-Pipeline.

Warum Tests in einer ML-Pipeline?
  ML-Code hat dieselben Bugs wie normaler Code:
  - Falsche Array-Shapes
  - Stille Fehler bei Datentypen (int statt float)
  - Logikfehler in der Metrik-Berechnung
  Plus ML-spezifische Bugs:
  - Data Leakage (Testdaten versehentlich im Training)
  - Schlechte Generalisierung durch Bug im Split

Jenkins führt diese Tests BEVOR das Modell trainiert wird aus.
Schlagen Tests fehl → kein Training, kein Deployment.
Das spart teure GPU-Stunden für fehlerhafte Runs.

pytest:
  Standard-Test-Framework für Python.
  Funktionen, die mit "test_" beginnen, werden automatisch erkannt.
  Assertions werden als normale Python-assert geschrieben.
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from train import evaluate_model, load_data, split_data, train_model


# ---------------------------------------------------------------------------
# Tests für load_data()
# ---------------------------------------------------------------------------
def test_load_data_returns_correct_shapes():
    """
    Stellt sicher, dass load_data() die erwarteten Dimensionen zurückgibt.

    Iris hat 150 Samples, 4 Features, 3 Klassen.
    Ändert sich der Datensatz unerwartet, schlägt dieser Test an.
    """
    X, y, target_names = load_data()

    assert X.shape == (150, 4),   f"Erwartet (150, 4), erhalten {X.shape}"
    assert y.shape == (150,),     f"Erwartet (150,), erhalten {y.shape}"
    assert len(target_names) == 3, f"Erwartet 3 Klassen, erhalten {len(target_names)}"


def test_load_data_no_nan():
    """
    Stellt sicher, dass keine NaN-Werte im Datensatz sind.

    NaN in Features führt bei sklearn zu kryptischen Fehlern oder
    silenten Fehlklassifikationen. Frühzeitig prüfen!
    """
    X, y, _ = load_data()
    assert not np.isnan(X).any(), "Datensatz enthält NaN-Werte in X"
    assert not np.isnan(y).any(), "Datensatz enthält NaN-Werte in y"


# ---------------------------------------------------------------------------
# Tests für split_data()
# ---------------------------------------------------------------------------
def test_split_data_size():
    """
    Stellt sicher, dass der Split die erwartete Anzahl Samples ergibt.

    Bei 150 Samples und test_size=0.2:
      - Training: 120 Samples
      - Test:      30 Samples
    """
    X, y, _ = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    assert len(X_train) == 120, f"Erwartet 120 Trainings-Samples, erhalten {len(X_train)}"
    assert len(X_test)  == 30,  f"Erwartet 30 Test-Samples, erhalten {len(X_test)}"


def test_split_no_leakage():
    """
    Data-Leakage-Prüfung: kein Sample darf gleichzeitig in Train UND Test sein.

    Leakage ist ein häufiger Bug: das Modell "sieht" Testdaten während des
    Trainings, die Metriken sind dadurch unrealistisch hoch.
    """
    X, y, _ = load_data()
    X_train, X_test, _, _ = split_data(X, y, test_size=0.2, random_state=42)

    # Konvertiere zu Mengen von Tupeln für einfachen Schnittmengen-Check
    train_set = set(map(tuple, X_train))
    test_set  = set(map(tuple, X_test))
    overlap   = train_set & test_set

    assert len(overlap) == 0, f"Data Leakage! {len(overlap)} Samples überlappen Train/Test"


def test_split_reproducible():
    """
    Stellt sicher, dass derselbe random_state denselben Split ergibt.
    Wichtig für reproduzierbare Experimente.
    """
    X, y, _ = load_data()
    X_train_a, _, _, _ = split_data(X, y, random_state=42)
    X_train_b, _, _, _ = split_data(X, y, random_state=42)

    np.testing.assert_array_equal(X_train_a, X_train_b,
                                  err_msg="Split ist nicht reproduzierbar!")


# ---------------------------------------------------------------------------
# Tests für train_model()
# ---------------------------------------------------------------------------
def test_train_model_returns_fitted_classifier():
    """
    Stellt sicher, dass train_model() ein trainiertes sklearn-Modell zurückgibt.

    Ein "fitted" Modell hat das Attribut "classes_" (gesetzt beim fit()).
    Ist es nicht gesetzt, wurde fit() nicht aufgerufen.
    """
    X, y, _ = load_data()
    X_train, _, y_train, _ = split_data(X, y)

    model = train_model(X_train, y_train, n_estimators=10, max_depth=3, random_state=42)

    assert isinstance(model, RandomForestClassifier), "Kein RandomForestClassifier zurückgegeben"
    assert hasattr(model, "classes_"), "Modell wurde nicht trainiert (kein classes_-Attribut)"
    assert len(model.classes_) == 3, f"Erwartet 3 Klassen, erhalten {len(model.classes_)}"


def test_train_model_can_predict():
    """
    Smoke-Test: Das trainierte Modell muss auf dem Testset Vorhersagen treffen können
    ohne Exception.
    """
    X, y, _ = load_data()
    X_train, X_test, y_train, _ = split_data(X, y)

    model = train_model(X_train, y_train, n_estimators=10, max_depth=3, random_state=42)
    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test), "Anzahl Vorhersagen stimmt nicht mit Test-Samples überein"


# ---------------------------------------------------------------------------
# Tests für evaluate_model()
# ---------------------------------------------------------------------------
def test_evaluate_model_returns_expected_keys():
    """
    Stellt sicher, dass evaluate_model() die erwarteten Metriken zurückgibt.

    Werden Schlüssel umbenannt (z.B. "f1_macro" → "f1"),
    würde mlflow.log_metrics() falsche Daten speichern.
    Dieser Test fängt solche Umbenennungen frühzeitig ab.
    """
    X, y, target_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train, n_estimators=10, max_depth=None, random_state=42)

    metrics = evaluate_model(model, X_test, y_test, target_names)

    assert "accuracy" in metrics, "Schlüssel 'accuracy' fehlt in Metriken"
    assert "f1_macro" in metrics, "Schlüssel 'f1_macro' fehlt in Metriken"


def test_evaluate_model_accuracy_range():
    """
    Sanity-Check: Accuracy muss zwischen 0 und 1 liegen.

    Klingt trivial, aber Implementierungsfehler (z.B. Prozent statt Anteil)
    können Werte wie 93.4 statt 0.934 erzeugen.
    """
    X, y, target_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train, n_estimators=10, max_depth=None, random_state=42)

    metrics = evaluate_model(model, X_test, y_test, target_names)

    assert 0.0 <= metrics["accuracy"] <= 1.0, \
        f"Accuracy außerhalb [0, 1]: {metrics['accuracy']}"
    assert 0.0 <= metrics["f1_macro"] <= 1.0, \
        f"F1-Score außerhalb [0, 1]: {metrics['f1_macro']}"


def test_evaluate_model_minimum_quality():
    """
    Integrations-Test: Das Modell muss eine Mindest-Accuracy von 90% erreichen.

    Iris ist ein einfacher Datensatz – 90% sind leicht erreichbar.
    Fällt die Accuracy darunter, deutet das auf einen Bug hin,
    nicht auf ein schwieriges Problem.

    Dieser Test simuliert den MIN_ACCURACY-Check in der Pipeline.
    """
    X, y, target_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, random_state=42)
    model = train_model(X_train, y_train, n_estimators=100, max_depth=None, random_state=42)

    metrics = evaluate_model(model, X_test, y_test, target_names)

    assert metrics["accuracy"] >= 0.90, \
        f"Accuracy {metrics['accuracy']:.4f} unter Mindestschwelle 0.90"
