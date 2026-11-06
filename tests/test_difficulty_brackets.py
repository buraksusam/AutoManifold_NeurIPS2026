from pipeline.datasets import DATASET_DIFFICULTY


def test_paper_authoritative_brackets():
    assert DATASET_DIFFICULTY["pbmc3k"] == "Moderate"
    assert DATASET_DIFFICULTY["kidney_scrnaseq"] == "Moderate"
    assert DATASET_DIFFICULTY["cortex_development"] == "Hard"
