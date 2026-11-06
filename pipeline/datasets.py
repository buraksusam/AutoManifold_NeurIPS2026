"""
pipeline/datasets.py — Built-in bio/chem dataset loaders.

Every loader returns (X, y, label_names, info):
    X           np.ndarray  float32  (n, d)
    y           np.ndarray  int32    (n,)
    label_names list[str]            class name per integer label
    info        dict                 metadata (name, description, n_samples, …)
"""
from __future__ import annotations

from functools import lru_cache
import importlib.util
import io
import logging
from pathlib import Path
import urllib.request
from urllib.error import HTTPError, URLError
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DatasetReturn = Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]


class DatasetUnavailableError(RuntimeError):
    """Raised when a built-in dataset cannot be loaded and should not silently fall back."""


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PBMC3K_PATH = DATA_DIR / "pbmc3k_processed.h5ad"
MICE_PROTEIN_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls",
    "https://raw.githubusercontent.com/emanuelecavalleri/MiceProteinExpression/main/Data_Cortex_Nuclear.xls",
    "https://raw.githubusercontent.com/VishalPatnaik/Mice-Protein-Analysis/main/Data_Cortex_Nuclear.xls",
    "https://raw.githubusercontent.com/Mahpara/Visual-Data-Analysis/master/Data_Cortex_Nuclear.xls",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pack(X: np.ndarray, y: np.ndarray, label_names: List[str],
          name: str, description: str, n_features_original: int | None = None) -> DatasetReturn:
    return X, y, label_names, {
        "name": name,
        "description": description,
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(np.unique(y))),
        "n_features_original": int(n_features_original or X.shape[1]),
    }


def _encode(series: "pd.Series") -> Tuple[np.ndarray, List[str]]:
    """Encode a string series to integer codes; return (y, label_names)."""
    label_names = sorted(series.astype(str).unique().tolist())
    y = np.array([label_names.index(v) for v in series.astype(str)], dtype=np.int32)
    return y, label_names


def _pca_reduce(X: np.ndarray, n_components: int = 50) -> np.ndarray:
    from sklearn.decomposition import PCA
    n = min(n_components, X.shape[1], X.shape[0] - 1)
    return PCA(n_components=n).fit_transform(X).astype(np.float32)


def _fetch_url(url: str, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _raise_unavailable(dataset_name: str, exc: Exception | str) -> None:
    message = str(exc)
    logger.warning("%s unavailable: %s", dataset_name, message)
    raise DatasetUnavailableError(f"{dataset_name} unavailable: {message}")


def _make_numeric_label_names(y: np.ndarray, prefix: str = "Class") -> List[str]:
    return [f"{prefix} {int(i)}" for i in sorted(np.unique(y))]


def _stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Class-stratified subsample to `n_samples`. Returns full arrays unchanged if
    the original size is already <= n_samples."""
    if len(X) <= n_samples:
        return X, y
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    proportions = counts / counts.sum()
    # Allocate integer quotas per class (largest-remainder method to hit n_samples exactly)
    raw = proportions * n_samples
    floor_quota = np.floor(raw).astype(int)
    remainder = raw - floor_quota
    deficit = n_samples - int(floor_quota.sum())
    if deficit > 0:
        order = np.argsort(-remainder)
        floor_quota[order[:deficit]] += 1
    idx_parts = []
    for cls, quota in zip(classes, floor_quota):
        cls_idx = np.where(y == cls)[0]
        quota = int(min(quota, len(cls_idx)))
        if quota <= 0:
            continue
        chosen = rng.choice(cls_idx, size=quota, replace=False)
        idx_parts.append(chosen)
    idx = np.concatenate(idx_parts)
    rng.shuffle(idx)
    return X[idx], y[idx]


def _first_reachable_url(urls: List[str], timeout: int = 5) -> tuple[str | None, str | None]:
    for url in urls:
        request = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=timeout):
                return url, None
        except HTTPError as exc:
            if exc.code == 405:
                try:
                    with urllib.request.urlopen(url, timeout=timeout):
                        return url, None
                except Exception as retry_exc:
                    last_error = str(retry_exc)
                else:
                    last_error = None
            else:
                last_error = f"HTTP {exc.code}"
        except URLError as exc:
            last_error = str(exc.reason)
        except Exception as exc:
            last_error = str(exc)
    return None, last_error if 'last_error' in locals() else "no reachable source found"


def _fetch_first_available_url(urls: List[str], timeout: int = 60) -> tuple[bytes, str]:
    errors: List[str] = []
    for url in urls:
        try:
            return _fetch_url(url, timeout=timeout), url
        except Exception as exc:
            errors.append(f"{url} -> {exc}")
    raise RuntimeError("; ".join(errors))


def _pack_sklearn(
    X: np.ndarray,
    y: np.ndarray,
    label_names: List[str],
    name: str,
    description: str,
) -> DatasetReturn:
    return _pack(
        X.astype(np.float32),
        y.astype(np.int32),
        label_names,
        name,
        description,
    )


# ── 1. Breast Cancer Wisconsin ────────────────────────────────────────────────

def load_breast_cancer() -> DatasetReturn:
    from sklearn.datasets import load_breast_cancer as _load
    d = _load()
    return _pack(
        d.data.astype(np.float32), d.target.astype(np.int32),
        list(d.target_names),
        "Breast Cancer Wisconsin",
        "Tumour cell nuclei features; binary malignant/benign classification.",
    )


# ── 2. PBMC 3k single-cell RNA-seq ───────────────────────────────────────────

def load_pbmc3k() -> DatasetReturn:
    if not PBMC3K_PATH.exists():
        _raise_unavailable("PBMC 3k scRNA-seq", f"local file missing: {PBMC3K_PATH}")

    try:
        logger.info("Loading PBMC 3k from local H5AD …")
        if _module_available("anndata"):
            import anndata as ad
            adata = ad.read_h5ad(PBMC3K_PATH)
        elif _module_available("scanpy"):
            import scanpy as sc
            adata = sc.read_h5ad(PBMC3K_PATH)
        else:
            _raise_unavailable("PBMC 3k scRNA-seq", "anndata/scanpy is not installed")

        X = np.asarray(adata.obsm["X_pca"][:, :50], dtype=np.float32)
        y, label_names = _encode(adata.obs["louvain"])
        return _pack(X, y, label_names,
                     "PBMC 3k scRNA-seq",
                     "2,700 PBMCs; 8 Louvain cell-type clusters on top-50 PCs.",
                     getattr(adata, "n_vars", X.shape[1]))
    except Exception as e:
        _raise_unavailable("PBMC 3k scRNA-seq", e)


# ── 3. Iris flower morphology ────────────────────────────────────────────────

def load_iris() -> DatasetReturn:
    from sklearn.datasets import load_iris as _load

    d = _load()
    return _pack_sklearn(
        d.data,
        d.target,
        list(d.target_names),
        "Iris",
        "Classic 3-species flower morphology benchmark; 4 features, 3 classes.",
    )


# ── 4. Wine chemistry ────────────────────────────────────────────────────────

def load_wine() -> DatasetReturn:
    from sklearn.datasets import load_wine as _load

    d = _load()
    return _pack_sklearn(
        d.data,
        d.target,
        list(d.target_names),
        "Wine Recognition",
        "Chemical analysis of wines from 3 cultivars; 13 features, 3 classes.",
    )


# ── 5. Mice Cortical Protein Expression ──────────────────────────────────────

def load_mice_protein() -> DatasetReturn:
    if not _module_available("xlrd"):
        _raise_unavailable("Mice Protein Expression", "xlrd is not installed")

    try:
        raw, source_url = _fetch_first_available_url(MICE_PROTEIN_URLS)
        logger.info("Loading Mice Protein Expression from %s …", source_url)
        df = pd.read_excel(io.BytesIO(raw), engine="xlrd")
        df = df.dropna(axis=1, thresh=int(0.8 * len(df)))
        df = df.fillna(df.median(numeric_only=True))
        y, label_names = _encode(df["class"])
        drop_cols = {"mouseid", "behavior", "genotype", "treatment", "class"}
        feat_cols = [c for c in df.select_dtypes(include=np.number).columns
                     if c.lower() not in drop_cols]
        X = df[feat_cols].values.astype(np.float32)
        return _pack(X, y, label_names,
                     "Mice Protein Expression",
                     "Cortical protein levels in a Down-syndrome mouse model; 8 behaviour classes.")
    except Exception as e:
        _raise_unavailable("Mice Protein Expression", e)


# ── 6. Handwritten digits ────────────────────────────────────────────────────

def load_digits() -> DatasetReturn:
    from sklearn.datasets import load_digits as _load

    d = _load()
    return _pack_sklearn(
        d.data,
        d.target,
        [str(x) for x in d.target_names],
        "Digits",
        "8×8 handwritten digit images flattened to 64 features; 10 classes.",
    )


# ── 6b. Fashion-MNIST (OpenML) ───────────────────────────────────────────────

FASHION_MNIST_DEFAULT_SUBSAMPLE = 5000
FASHION_MNIST_CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def load_fashion_mnist(n_samples: int = FASHION_MNIST_DEFAULT_SUBSAMPLE) -> DatasetReturn:
    """Fashion-MNIST via OpenML; stratified-subsampled to `n_samples` (default 10k)."""
    try:
        from sklearn.datasets import fetch_openml
    except Exception as exc:  # pragma: no cover
        _raise_unavailable("Fashion-MNIST", f"scikit-learn unavailable: {exc}")

    try:
        logger.info(
            "Fetching Fashion-MNIST from OpenML (subsample n_samples=%d)…", n_samples
        )
        bunch = fetch_openml(
            "Fashion-MNIST", version=1, as_frame=False, parser="auto", cache=True
        )
        X_full = np.asarray(bunch.data, dtype=np.float32)
        # OpenML returns string-typed labels; coerce to int.
        y_full = np.asarray(bunch.target).astype(np.int32)
        X, y = _stratified_subsample(X_full, y_full, n_samples=n_samples, random_state=42)
        return _pack(
            X.astype(np.float32),
            y.astype(np.int32),
            list(FASHION_MNIST_CLASS_NAMES),
            "Fashion-MNIST",
            f"28×28 greyscale fashion images flattened to 784 features; "
            f"10 classes (subsampled to {len(X)} of 70 000).",
        )
    except Exception as e:
        _raise_unavailable("Fashion-MNIST", e)


# ── 6c. Swiss Roll (manifold) ────────────────────────────────────────────────

def load_swiss_roll(n_samples: int = 2000) -> DatasetReturn:
    """Classic Swiss-roll manifold; continuous parameter `t` quantised into 6 bands for colouring."""
    try:
        from sklearn.datasets import make_swiss_roll
    except Exception as exc:
        _raise_unavailable("Swiss Roll", f"scikit-learn unavailable: {exc}")

    X, t = make_swiss_roll(n_samples=n_samples, noise=0.05, random_state=42)
    bands = 6
    edges = np.quantile(t, np.linspace(0.0, 1.0, bands + 1)[1:-1])
    y = np.digitize(t, edges).astype(np.int32)
    label_names = [f"Arc band {i}" for i in range(bands)]
    return _pack_sklearn(
        X.astype(np.float32),
        y,
        label_names,
        "Swiss Roll",
        "Classic 3-D rolled-sheet manifold; arc-length quantised into 6 bands; 3 features.",
    )


# ── 6d. MNIST 784 (full size) ────────────────────────────────────────────────

MNIST_FULL_DEFAULT_SUBSAMPLE = 5000


def load_mnist_full(n_samples: int = MNIST_FULL_DEFAULT_SUBSAMPLE) -> DatasetReturn:
    """MNIST 784 via OpenML; stratified-subsampled to `n_samples`."""
    try:
        from sklearn.datasets import fetch_openml
    except Exception as exc:
        _raise_unavailable("MNIST Full", f"scikit-learn unavailable: {exc}")

    try:
        logger.info("Fetching MNIST 784 from OpenML (subsample n_samples=%d)…", n_samples)
        bunch = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto", cache=True)
        X_full = np.asarray(bunch.data, dtype=np.float32)
        y_full = np.asarray(bunch.target).astype(np.int32)
        X, y = _stratified_subsample(X_full, y_full, n_samples=n_samples, random_state=42)
        return _pack(
            X.astype(np.float32),
            y.astype(np.int32),
            [str(i) for i in range(10)],
            "MNIST (full size)",
            f"28×28 greyscale handwritten digits flattened to 784 features; "
            f"10 classes (subsampled to {len(X)} of 70 000).",
        )
    except Exception as e:
        _raise_unavailable("MNIST Full", e)


# ── 6e. Olivetti Faces ───────────────────────────────────────────────────────

def load_olivetti_faces() -> DatasetReturn:
    """AT&T Olivetti faces — 400 64×64 greyscale face images, 40 subjects."""
    try:
        from sklearn.datasets import fetch_olivetti_faces
    except Exception as exc:
        _raise_unavailable("Olivetti Faces", f"scikit-learn unavailable: {exc}")

    try:
        logger.info("Fetching Olivetti Faces…")
        bunch = fetch_olivetti_faces(shuffle=True, random_state=42)
        X = np.asarray(bunch.data, dtype=np.float32)
        y = np.asarray(bunch.target, dtype=np.int32)
        label_names = [f"Subject {i}" for i in range(int(y.max()) + 1)]
        return _pack(
            X,
            y,
            label_names,
            "Olivetti Faces",
            "400 64×64 greyscale face images flattened to 4096 features; 40 subjects.",
        )
    except Exception as e:
        _raise_unavailable("Olivetti Faces", e)


# ── 6g. 20 Newsgroups (text) ─────────────────────────────────────────────────

NEWSGROUPS_DEFAULT_SUBSAMPLE = 4000
NEWSGROUPS_TFIDF_FEATURES = 2000
NEWSGROUPS_PCA_COMPONENTS = 100


def load_20newsgroups(n_samples: int = NEWSGROUPS_DEFAULT_SUBSAMPLE) -> DatasetReturn:
    """20 Newsgroups text corpus → TF-IDF (top 2000) → SVD-reduced to 100 components."""
    try:
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
    except Exception as exc:
        _raise_unavailable("20 Newsgroups", f"scikit-learn unavailable: {exc}")

    try:
        logger.info("Fetching 20 Newsgroups (all subset)…")
        bunch = fetch_20newsgroups(
            subset="all",
            remove=("headers", "footers", "quotes"),
            shuffle=True,
            random_state=42,
        )
        texts = bunch.data
        y_full = np.asarray(bunch.target, dtype=np.int32)
        if len(texts) > n_samples:
            rng = np.random.default_rng(42)
            classes, counts = np.unique(y_full, return_counts=True)
            proportions = counts / counts.sum()
            quotas = np.maximum(1, np.round(proportions * n_samples).astype(int))
            idx_parts = []
            for cls, q in zip(classes, quotas):
                cls_idx = np.where(y_full == cls)[0]
                q = int(min(q, len(cls_idx)))
                idx_parts.append(rng.choice(cls_idx, size=q, replace=False))
            idx = np.concatenate(idx_parts)
            rng.shuffle(idx)
            texts = [texts[i] for i in idx]
            y = y_full[idx]
        else:
            y = y_full

        vec = TfidfVectorizer(
            max_features=NEWSGROUPS_TFIDF_FEATURES,
            stop_words="english",
            min_df=5,
            max_df=0.5,
        )
        X_sparse = vec.fit_transform(texts)
        n_comp = min(NEWSGROUPS_PCA_COMPONENTS, X_sparse.shape[1] - 1, X_sparse.shape[0] - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        X = svd.fit_transform(X_sparse).astype(np.float32)
        return _pack(
            X,
            y.astype(np.int32),
            list(bunch.target_names),
            "20 Newsgroups",
            f"TF-IDF (top {NEWSGROUPS_TFIDF_FEATURES}) → Truncated-SVD ({n_comp}) on a {len(X)}-doc subsample; 20 classes.",
            n_features_original=NEWSGROUPS_TFIDF_FEATURES,
        )
    except Exception as e:
        _raise_unavailable("20 Newsgroups", e)


# ── 6h. Labeled Faces in the Wild ────────────────────────────────────────────

LFW_MIN_FACES = 70
LFW_RESIZE = 0.4


def load_lfw_people() -> DatasetReturn:
    """Labeled Faces in the Wild — only people with ≥70 face images, resized to 0.4×."""
    try:
        from sklearn.datasets import fetch_lfw_people
    except Exception as exc:
        _raise_unavailable("LFW People", f"scikit-learn unavailable: {exc}")

    try:
        logger.info("Fetching Labeled Faces in the Wild (min_faces_per_person=%d)…", LFW_MIN_FACES)
        bunch = fetch_lfw_people(
            min_faces_per_person=LFW_MIN_FACES,
            resize=LFW_RESIZE,
            color=False,
            funneled=True,
        )
        X = np.asarray(bunch.data, dtype=np.float32)
        y = np.asarray(bunch.target, dtype=np.int32)
        return _pack(
            X,
            y,
            list(bunch.target_names),
            "Labeled Faces in the Wild",
            f"Greyscale celebrity face images (resized {LFW_RESIZE}×) flattened to {X.shape[1]} features; "
            f"{int(y.max()) + 1} people with ≥{LFW_MIN_FACES} images each.",
        )
    except Exception as e:
        _raise_unavailable("LFW People", e)


# ── Generic UCSC-style scRNA-seq loader (gene × cell TSV + meta) ────────────

def _load_ucsc_scrnaseq(
    *,
    dataset_dir: Path,
    label_col: str,
    name: str,
    description: str,
    n_samples: int | None = None,
    n_hvg: int = 2000,
    n_pcs: int = 50,
    cell_id_col: str | None = None,
    drop_label_values: tuple[str, ...] = ("nan", "NaN", "NA", ""),
    random_state: int = 42,
) -> DatasetReturn:
    """Load a UCSC Cell Browser–style scRNA-seq dataset.

    Expects:
      - exprMatrix.tsv.gz  (genes × cells, first column = gene name; header row = cell IDs)
      - meta.tsv           (rows = cells; first column or `cell_id_col` = cell ID)

    Pipeline: align cells, drop missing labels, log1p, top-`n_hvg` HVG by variance,
    standardize, PCA(`n_pcs`). Result is cached as `cache_<label>_<n>x<pcs>.npz`.
    """
    expr_path = dataset_dir / "exprMatrix.tsv.gz"
    meta_path = dataset_dir / "meta.tsv"
    if not expr_path.exists():
        _raise_unavailable(name, f"missing {expr_path}")
    if not meta_path.exists():
        _raise_unavailable(name, f"missing {meta_path}")

    cache_tag = f"{label_col}_n{n_samples or 'all'}_h{n_hvg}_p{n_pcs}_s{random_state}"
    cache_path = dataset_dir / f"cache_{cache_tag}.npz"
    if cache_path.exists():
        try:
            logger.info("Loading cached %s from %s", name, cache_path.name)
            blob = np.load(cache_path, allow_pickle=True)
            X = blob["X"].astype(np.float32)
            y = blob["y"].astype(np.int32)
            label_names = [str(s) for s in blob["label_names"].tolist()]
            n_features_original = int(blob["n_features_original"])
            return _pack(X, y, label_names, name, description, n_features_original)
        except Exception as exc:
            logger.warning("Cache for %s unreadable (%s); rebuilding.", name, exc)

    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        _raise_unavailable(name, f"scikit-learn unavailable: {exc}")

    try:
        logger.info("Reading meta for %s …", name)
        meta = pd.read_csv(meta_path, sep="\t", dtype=str)
        if cell_id_col is None:
            cell_id_col = meta.columns[0]
        if label_col not in meta.columns:
            _raise_unavailable(name, f"label column '{label_col}' not in meta ({list(meta.columns)})")
        meta = meta.set_index(cell_id_col)
        meta.index = meta.index.astype(str).str.strip()

        logger.info("Reading expression matrix for %s (this may take a minute) …", name)
        expr = pd.read_csv(expr_path, sep="\t", index_col=0, compression="infer", low_memory=False)
        # Force numeric values only (gene-name index handled separately).
        expr = expr.apply(pd.to_numeric, errors="coerce").astype(np.float32)
        # expr is (genes × cells); transpose to (cells × genes)
        expr = expr.T
        expr.index = expr.index.astype(str).str.strip()

        common = expr.index.intersection(meta.index)
        if len(common) == 0:
            _raise_unavailable(name, "no overlap between expression matrix cells and meta cell IDs")
        expr = expr.loc[common]
        labels = meta.loc[common, label_col].astype(str).str.strip()

        keep = ~labels.isin(set(drop_label_values))
        keep &= labels.notna()
        expr = expr.loc[keep.values]
        labels = labels.loc[keep.values]

        n_features_original = int(expr.shape[1])

        if n_samples is not None and len(expr) > n_samples:
            y_tmp, _ = _encode(labels)
            X_tmp = expr.values
            X_sub, y_sub = _stratified_subsample(X_tmp, y_tmp, n_samples=n_samples, random_state=random_state)
            # rebuild labels series in subsampled order
            inv = sorted(labels.astype(str).unique().tolist())
            labels = pd.Series([inv[i] for i in y_sub])
            expr_values = X_sub
        else:
            expr_values = expr.values

        logger.info("log1p + HVG selection (top %d) for %s …", n_hvg, name)
        X = np.log1p(np.asarray(expr_values, dtype=np.float32))
        if X.shape[1] > n_hvg:
            var = X.var(axis=0)
            top = np.argpartition(var, -n_hvg)[-n_hvg:]
            X = X[:, top]
        X = StandardScaler(with_mean=True, with_std=True).fit_transform(X).astype(np.float32)

        n_comp = min(n_pcs, X.shape[1], X.shape[0] - 1)
        logger.info("PCA(%d) for %s …", n_comp, name)
        X = PCA(n_components=n_comp, random_state=random_state).fit_transform(X).astype(np.float32)

        y, label_names = _encode(labels)

        try:
            np.savez_compressed(
                cache_path,
                X=X, y=y,
                label_names=np.asarray(label_names, dtype=object),
                n_features_original=np.int64(n_features_original),
            )
            logger.info("Cached %s → %s", name, cache_path.name)
        except Exception as exc:
            logger.warning("Could not write cache for %s: %s", name, exc)

        return _pack(X, y, label_names, name, description, n_features_original)
    except DatasetUnavailableError:
        raise
    except Exception as exc:
        _raise_unavailable(name, exc)


# ── Aging human skin (GSE — UCSC Cell Browser) ──────────────────────────────

AGING_SKIN_DEFAULT_SUBSAMPLE = 8000

def load_aging_human_skin(n_samples: int = AGING_SKIN_DEFAULT_SUBSAMPLE) -> DatasetReturn:
    return _load_ucsc_scrnaseq(
        dataset_dir=DATA_DIR / "aging_human_skin",
        label_col="Celltype",
        cell_id_col="cellId",
        name="Aging Human Skin scRNA-seq",
        description=(
            "Human skin scRNA-seq across young/aged donors; "
            f"top-2000 HVG → PCA(50); cell-type labels (subsampled to {AGING_SKIN_DEFAULT_SUBSAMPLE} of ~15.5k)."
        ),
        n_samples=n_samples,
    )


# ── Developing human cortex (Pollen et al., UCSC Cell Browser) ──────────────

def load_cortex_development() -> DatasetReturn:
    return _load_ucsc_scrnaseq(
        dataset_dir=DATA_DIR / "cortex_development",
        label_col="WGCNAcluster",
        cell_id_col="Cell",
        name="Human Cortex Development scRNA-seq",
        description=(
            "Developing human cortex scRNA-seq; top-2000 HVG → PCA(50); "
            "WGCNA cluster labels across ~4.2k cells."
        ),
        n_samples=None,
    )


# ── Mouse incisor (UCSC Cell Browser) ───────────────────────────────────────

def load_mouse_incisor() -> DatasetReturn:
    return _load_ucsc_scrnaseq(
        dataset_dir=DATA_DIR / "mouse_incisor",
        label_col="cluster_annotation",
        cell_id_col="cell_id",
        name="Mouse Incisor scRNA-seq",
        description=(
            "Mouse incisor scRNA-seq; top-2000 HVG → PCA(50); "
            "annotated cell-type clusters across ~2.9k cells."
        ),
        n_samples=None,
    )


# ── Human kidney scRNA-seq (Stewart et al.-style; UCSC Cell Browser) ────────

KIDNEY_SCRNA_DEFAULT_SUBSAMPLE = 8000

def load_kidney_scrnaseq(n_samples: int = KIDNEY_SCRNA_DEFAULT_SUBSAMPLE) -> DatasetReturn:
    return _load_ucsc_scrnaseq(
        dataset_dir=DATA_DIR / "scRNA-seq",
        label_col="cell_type",
        cell_id_col="cellId",
        name="Human Kidney scRNA-seq",
        description=(
            "Human kidney scRNA-seq with detailed ontology cell-type labels; "
            f"top-2000 HVG → PCA(50); subsampled to {KIDNEY_SCRNA_DEFAULT_SUBSAMPLE} of ~20k."
        ),
        n_samples=n_samples,
    )


# ── 7. Synthetic cell states (5 classes) ─────────────────────────────────────

def load_synthetic_blobs5() -> DatasetReturn:
    from sklearn.datasets import make_blobs

    X, y = make_blobs(
        n_samples=2400,
        n_features=24,
        centers=5,
        cluster_std=[1.2, 1.6, 1.5, 1.3, 1.8],
        random_state=42,
    )
    return _pack_sklearn(
        X,
        y,
        _make_numeric_label_names(y, prefix="State"),
        "Synthetic Cell States",
        "Deterministic 5-state expression-like blobs with moderate separation; 24 features, 5 classes.",
    )


# ── 8. Synthetic anisotropic clusters (6 classes) ────────────────────────────

def load_synthetic_aniso6() -> DatasetReturn:
    from sklearn.datasets import make_blobs

    X, y = make_blobs(
        n_samples=3000,
        n_features=18,
        centers=6,
        cluster_std=2.4,
        random_state=17,
    )
    transform = np.array([
        [1.0, 0.4, 0.2],
        [0.0, 1.2, -0.3],
        [0.1, 0.3, 0.9],
    ], dtype=np.float32)
    X_view = X[:, :3] @ transform
    X = np.concatenate([X_view, X[:, 3:]], axis=1)
    return _pack_sklearn(
        X,
        y,
        _make_numeric_label_names(y, prefix="Population"),
        "Synthetic Anisotropic Populations",
        "6 anisotropic clusters with correlated features and partial overlap; 18 features, 6 classes.",
    )


# ── 9. Synthetic signalling programs (7 classes) ─────────────────────────────

def load_synthetic_signal7() -> DatasetReturn:
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=2800,
        n_features=30,
        n_informative=16,
        n_redundant=6,
        n_repeated=0,
        n_classes=7,
        n_clusters_per_class=1,
        class_sep=1.35,
        flip_y=0.015,
        random_state=7,
    )
    return _pack_sklearn(
        X,
        y,
        _make_numeric_label_names(y, prefix="Program"),
        "Synthetic Signalling Programs",
        "7-class multiclass benchmark with informative, redundant, and noisy features; 30 features, 7 classes.",
    )


# ── Registry ──────────────────────────────────────────────────────────────────

DATASETS: Dict[str, Tuple[Callable[[], DatasetReturn], str]] = {
    "breast_cancer":     (load_breast_cancer,    "Breast Cancer Wisconsin — sklearn built-in (oncology)"),
    "iris":              (load_iris,             "Iris flower morphology — 3 species (botany)"),
    "wine":              (load_wine,             "Wine chemistry — 3 cultivars (chemometrics)"),
    "digits":            (load_digits,           "Handwritten digits 8×8 — 10 classes (vision)"),
    "synthetic_blobs5":  (load_synthetic_blobs5, "Synthetic cell states — 5 classes (simulation)"),
    "swiss_roll":        (load_swiss_roll,       "Swiss-roll manifold — 6 colour bands (geometry)"),
    "mnist_full":        (load_mnist_full,       "MNIST 784 — 10 digit classes, subsampled (vision)"),
    "olivetti_faces":    (load_olivetti_faces,   "Olivetti faces — 40 subjects, 4096 features (vision)"),
    "pbmc3k":            (load_pbmc3k,           "PBMC 3k scRNA-seq — 8 cell types, top-50 PCs (genomics)"),
    "mice_protein":      (load_mice_protein,     "Mice Cortical Protein Expression — 8 classes (neuroscience)"),
    "synthetic_aniso6":  (load_synthetic_aniso6, "Synthetic anisotropic populations — 6 classes (simulation)"),
    "synthetic_signal7": (load_synthetic_signal7,"Synthetic signalling programs — 7 classes (simulation)"),
    "fashion_mnist":     (load_fashion_mnist,    "Fashion-MNIST — 10 clothing classes, 784 features, subsampled (vision)"),
    "newsgroups20":      (load_20newsgroups,     "20 Newsgroups text — TF-IDF→SVD, 20 topic classes (NLP)"),
    "lfw_people":        (load_lfw_people,       "Labeled Faces in the Wild — celebrity faces, ≥70 imgs/person (vision)"),
    "aging_human_skin":  (load_aging_human_skin,    "Aging human skin scRNA-seq — Celltype labels, top-2000 HVG → PCA(50) (genomics)"),
    "cortex_development":(load_cortex_development,  "Developing human cortex scRNA-seq — WGCNA cluster labels, PCA(50) (genomics)"),
    "mouse_incisor":     (load_mouse_incisor,       "Mouse incisor scRNA-seq — annotated cell types, PCA(50) (genomics)"),
    "kidney_scrnaseq":   (load_kidney_scrnaseq,     "Human kidney scRNA-seq — ontology cell-type labels, PCA(50), subsampled (genomics)"),
}


# Difficulty brackets — how hard each dataset is to embed faithfully.
DATASET_DIFFICULTY: Dict[str, str] = {
    # Easy: low-dim, well-separated, classical benchmarks.
    "breast_cancer":    "Easy",
    "iris":             "Easy",
    "wine":             "Easy",
    "digits":           "Easy",
    "synthetic_blobs5": "Easy",
    # Moderate: manifolds, mid-size image sets, overlapping clusters.
    "swiss_roll":        "Moderate",
    "mnist_full":        "Moderate",
    "olivetti_faces":    "Moderate",
    "pbmc3k":            "Moderate",
    "mice_protein":      "Moderate",
    "synthetic_aniso6":  "Moderate",
    "synthetic_signal7": "Moderate",
    # Hard: high-dim, many classes, sparse/text or unconstrained natural images.
    "fashion_mnist": "Hard",
    "newsgroups20":  "Hard",
    "lfw_people":    "Hard",
    # scRNA-seq datasets — brackets are PAPER-ASSIGNED (authoritative), not
    # purely kNN-derived.
    "aging_human_skin":   "Easy",
    "kidney_scrnaseq":    "Moderate",
    "cortex_development": "Hard",
    "mouse_incisor":      "Hard",
}

DIFFICULTY_ORDER: List[str] = ["Easy", "Moderate", "Hard"]


DATASET_METADATA: Dict[str, Dict[str, Any]] = {
    "breast_cancer":     {"n_classes": 2},
    "iris":              {"n_classes": 3},
    "wine":              {"n_classes": 3},
    "digits":            {"n_classes": 10},
    "synthetic_blobs5":  {"n_classes": 5},
    "swiss_roll":        {"n_classes": 6},
    "mnist_full":        {"n_classes": 10},
    "olivetti_faces":    {"n_classes": 40},
    "pbmc3k":            {"n_classes": 8},
    "mice_protein":      {"n_classes": 8},
    "synthetic_aniso6":  {"n_classes": 6},
    "synthetic_signal7": {"n_classes": 7},
    "fashion_mnist":     {"n_classes": 10},
    "newsgroups20":      {"n_classes": 20},
    "lfw_people":        {"n_classes": None},  # depends on min_faces_per_person filter
    "aging_human_skin":  {"n_classes": 13, "label_column": "Celltype"},
    "cortex_development":{"n_classes": 47, "label_column": "WGCNAcluster"},
    "mouse_incisor":     {"n_classes": 17, "label_column": "cluster_annotation"},
    "kidney_scrnaseq":   {"n_classes": 13, "label_column": "cell_type"},
}


@lru_cache(maxsize=None)
def get_dataset_availability(name: str) -> Dict[str, Any]:
    """Return fast availability info for a built-in dataset without loading it fully."""
    if name not in DATASETS:
        return {"available": False, "reason": "Unknown dataset"}

    if name in {"breast_cancer", "iris", "wine", "digits",
                "synthetic_blobs5", "synthetic_aniso6", "synthetic_signal7",
                "swiss_roll"}:
        return {"available": True, "reason": None}

    if name in {"fashion_mnist", "mnist_full", "olivetti_faces",
                "newsgroups20", "lfw_people"}:
        if not _module_available("sklearn"):
            return {"available": False, "reason": "scikit-learn not installed"}
        # These all download/cache lazily on first call; don't probe the network here.
        return {"available": True, "reason": None}

    if name in {"aging_human_skin", "cortex_development", "mouse_incisor", "kidney_scrnaseq"}:
        dir_map = {
            "aging_human_skin":   DATA_DIR / "aging_human_skin",
            "cortex_development": DATA_DIR / "cortex_development",
            "mouse_incisor":      DATA_DIR / "mouse_incisor",
            "kidney_scrnaseq":    DATA_DIR / "scRNA-seq",
        }
        d = dir_map[name]
        if not (d / "exprMatrix.tsv.gz").exists():
            return {"available": False, "reason": f"local file missing: {d.name}/exprMatrix.tsv.gz"}
        if not (d / "meta.tsv").exists():
            return {"available": False, "reason": f"local file missing: {d.name}/meta.tsv"}
        if not _module_available("sklearn"):
            return {"available": False, "reason": "scikit-learn not installed"}
        return {"available": True, "reason": None}

    if name == "pbmc3k":
        if not PBMC3K_PATH.exists():
            return {"available": False, "reason": f"local file missing: {PBMC3K_PATH.name}"}
        if not (_module_available("anndata") or _module_available("scanpy")):
            return {"available": False, "reason": "anndata/scanpy is not installed"}
        return {"available": True, "reason": None}

    if name == "mice_protein":
        if not _module_available("xlrd"):
            return {"available": False, "reason": "xlrd is not installed"}
        url, reason = _first_reachable_url(MICE_PROTEIN_URLS)
        return {"available": url is not None, "reason": None if url is not None else f"source unreachable: {reason}"}

    return {"available": True, "reason": None}


def list_datasets() -> None:
    """Print a formatted table of available datasets."""
    print("\nAvailable built-in datasets:")
    print(f"  {'Key':<16}  Description")
    print(f"  {'-'*16}  {'-'*55}")
    for key, (_, desc) in DATASETS.items():
        print(f"  {key:<16}  {desc}")
    print()


def get_dataset(name: str) -> DatasetReturn:
    """Load a built-in dataset by key. Raises ValueError for unknown names."""
    if name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {list(DATASETS.keys())}"
        )
    loader, _ = DATASETS[name]
    logger.info("Loading built-in dataset: %s", name)
    return loader()


def get_dataset_catalog() -> List[Dict[str, Any]]:
    """Return lightweight dataset metadata for UI listing.

    Each item carries a ``difficulty`` field (``Easy`` / ``Moderate`` / ``Hard``)
    so the frontend can group datasets under bracket headers.
    """
    items: List[Dict[str, Any]] = []
    for key, (_, description) in DATASETS.items():
        meta = DATASET_METADATA.get(key, {})
        availability = get_dataset_availability(key)
        items.append({
            "key": key,
            "description": description,
            "n_classes": meta.get("n_classes"),
            "difficulty": DATASET_DIFFICULTY.get(key, "Moderate"),
            "available": availability.get("available", True),
            "unavailable_reason": availability.get("reason"),
        })
    # Sort by difficulty (Easy → Moderate → Hard), preserving insertion order within a bracket.
    order = {d: i for i, d in enumerate(DIFFICULTY_ORDER)}
    items.sort(key=lambda it: order.get(it["difficulty"], 99))
    return items
