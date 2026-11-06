from unittest.mock import patch

import numpy as np

from pipeline.datasets import get_dataset_availability, get_dataset_catalog, load_digits, load_iris, load_mice_protein, load_synthetic_signal7


def test_catalog_includes_availability_fields():
    with patch("pipeline.datasets.get_dataset_availability") as mock_availability:
        mock_availability.side_effect = lambda key: {
            "available": key == "breast_cancer",
            "reason": None if key == "breast_cancer" else "mock unavailable",
        }
        catalog = get_dataset_catalog()

    breast = next(item for item in catalog if item["key"] == "breast_cancer")
    iris = next(item for item in catalog if item["key"] == "iris")
    assert breast["available"] is True
    assert breast["unavailable_reason"] is None
    assert iris["available"] is False
    assert iris["unavailable_reason"] == "mock unavailable"


def test_local_multiclass_replacements_load():
    X_iris, y_iris, labels_iris, info_iris = load_iris()
    X_digits, y_digits, labels_digits, info_digits = load_digits()
    X_syn, y_syn, labels_syn, info_syn = load_synthetic_signal7()

    assert X_iris.shape[0] == len(y_iris) == 150
    assert info_iris["n_classes"] == 3
    assert labels_iris == ["setosa", "versicolor", "virginica"]

    assert X_digits.shape[0] == len(y_digits) == 1797
    assert info_digits["n_classes"] == 10
    assert labels_digits[0] == "0"

    assert X_syn.shape[0] == len(y_syn) == 2800
    assert info_syn["n_classes"] == 7
    assert labels_syn[0] == "Program 0"
    assert np.unique(y_syn).size == 7


def test_mice_protein_loads_from_restored_mirror_source():
    import pandas as pd
    df = pd.DataFrame({
        "feat_a": [1.0, 1.5],
        "feat_b": [2.0, 2.5],
        "class": ["c-CS-m", "t-CS-m"],
        "behavior": [1.0, 2.0],
    })

    with patch("pipeline.datasets._module_available", return_value=True), patch(
        "pipeline.datasets._fetch_first_available_url", return_value=(b"xls-bytes", "https://mirror.example/mice.xls")
    ), patch("pipeline.datasets.pd.read_excel", return_value=df):
        X, y, labels, info = load_mice_protein()

    assert X.shape == (2, 2)
    assert info["n_classes"] == 2
    assert labels == ["c-CS-m", "t-CS-m"]


def test_replacement_datasets_are_reported_available():
    for key in ["iris", "wine", "digits", "synthetic_blobs5", "synthetic_aniso6", "synthetic_signal7"]:
        availability = get_dataset_availability(key)
        assert availability["available"] is True
        assert availability["reason"] is None


def test_mice_protein_availability_uses_mirror_probe():
    with patch("pipeline.datasets._module_available", return_value=True), patch(
        "pipeline.datasets._first_reachable_url", return_value=("https://mirror.example/mice.xls", None)
    ):
        availability = get_dataset_availability("mice_protein")

    assert availability["available"] is True
    assert availability["reason"] is None