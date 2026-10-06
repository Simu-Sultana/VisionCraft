#!/usr/bin/env python3
"""
Computer Vision Project - Sheet 3 / Exercise 3.1
Writer Identification on ICDAR17 Historical WI using SIFT + VLAD + E-SVM.

This is a complete, self-contained solution built from the provided skeleton.
It implements:
  a) SIFT descriptors at SIFT keypoints, keypoint angle fixed to 0, RootSIFT/Hellinger normalization
  b) MiniBatchKMeans codebook generation
  c) VLAD encoding and mAP / Top-1 evaluation
  d) VLAD power normalization + L2 normalization
  e) Exemplar-SVM descriptors

Extra improvement options for the individual part:
  1) Intra-normalization of VLAD blocks (--intranorm)
  2) PCA whitening of global descriptors (--pca_whiten)

The code intentionally writes cache files into --cache_dir so experiments can be resumed.
"""

import argparse
import csv
import gzip
import os
import pickle
import hashlib
import shlex
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

try:
    from joblib import Parallel, delayed
except Exception:  # pragma: no cover - joblib is normally installed with scikit-learn
    Parallel = None
    delayed = None


# -----------------------------
# Utility functions
# -----------------------------

KNOWN_SUFFIXES = [".pkl.gz", ".txt", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".ocvmb", ".csv"]
EPS = 1e-12


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def cache_path(cache_dir: str, name: str) -> str:
    ensure_dir(cache_dir)
    return os.path.join(cache_dir, name)


def save_pickle_gz(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle_gz(path: str):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def strip_known_suffix(file_name: str) -> str:
    # Some file names may contain dots internally; remove only a known final suffix.
    for suffix in KNOWN_SUFFIXES:
        if file_name.endswith(suffix):
            return file_name[: -len(suffix)]
    return file_name


def resolve_file(folder: str, stem: str, preferred_suffix: str, auto_suffix: bool = True) -> str:
    """Return the image path. If preferred suffix is missing, optionally try common suffixes."""
    preferred = os.path.join(folder, stem + preferred_suffix)
    if not auto_suffix or os.path.exists(preferred):
        return preferred

    for suffix in [preferred_suffix, ".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        candidate = os.path.join(folder, stem + suffix)
        if os.path.exists(candidate):
            return candidate
    return preferred


def getFiles(folder: str, pattern: str, labelfile: str, auto_suffix: bool = True) -> Tuple[List[str], List[str]]:
    """
    Return image files and labels by reading the label file.

    Each line is expected to contain: image_stem writer_label
    shlex.split is used, so quoted file names with spaces are allowed.
    """
    files, labels = [], []
    with open(labelfile, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = shlex.split(line)
            if len(parts) < 2:
                raise ValueError(f"Invalid label line {line_no} in {labelfile!r}: {line!r}")
            stem = strip_known_suffix(parts[0])
            label = parts[1]
            files.append(resolve_file(folder, stem, pattern, auto_suffix=auto_suffix))
            labels.append(label)
    return files, labels


def check_files_exist(files: Sequence[str], split_name: str, max_examples: int = 5) -> None:
    missing = [p for p in files if not os.path.exists(p)]
    if missing:
        examples = "\n".join(f"  - {p}" for p in missing[:max_examples])
        raise FileNotFoundError(
            f"{len(missing)} {split_name} images were not found. Examples:\n{examples}\n"
            "Check --in_train/--in_test and --suffix_train/--suffix_test. "
            "You can also keep --auto_suffix enabled to try common image extensions."
        )


def toBinary(img: np.ndarray) -> np.ndarray:
    """Convert a grayscale image to a 0/255 binary image when it is not already binary."""
    if img is None:
        raise ValueError("Cannot binarize an empty image.")
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    unique = np.unique(img)
    if unique.size <= 2:
        # Accept 0/1 and 0/255 binary images.
        if unique.max() <= 1:
            img = (img * 255).astype(np.uint8)
        return img

    _, out = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def create_sift():
    """Create a SIFT extractor in a way that works with current OpenCV builds."""
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create()
    if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "SIFT_create"):
        return cv2.xfeatures2d.SIFT_create()
    raise RuntimeError(
        "SIFT is not available in this OpenCV installation. "
        "Install opencv-contrib-python or a recent OpenCV version with SIFT support."
    )


def rootsift_hellinger(desc: np.ndarray) -> np.ndarray:
    """
    Hellinger / RootSIFT normalization.

    The assignment asks for L1 normalization followed by element-wise signed square root,
    with no additional L2 normalization afterwards. SIFT descriptors are non-negative, but
    sign-sqrt is used for numerical safety.
    """
    desc = desc.astype(np.float32, copy=False)
    l1 = np.sum(np.abs(desc), axis=1, keepdims=True) + EPS
    desc = desc / l1
    desc = np.sign(desc) * np.sqrt(np.abs(desc))
    return desc.astype(np.float32, copy=False)


def dense_fallback_keypoints(img: np.ndarray, step: int = 24, size: int = 16) -> List[cv2.KeyPoint]:
    """Fallback only for rare images where SIFT detects no keypoints."""
    h, w = img.shape[:2]
    keypoints = []
    for y in range(size, max(size + 1, h - size), step):
        for x in range(size, max(size + 1, w - size), step):
            keypoints.append(cv2.KeyPoint(float(x), float(y), float(size), angle=0.0))
    return keypoints


def computeDescs(fname: str, norm_hellinger: bool = False, to_binary: bool = False) -> np.ndarray:
    """
    Load an image and compute SIFT descriptors at SIFT keypoints.

    Required assignment details:
      - SIFT keypoints are detected normally.
      - All keypoint angles are then fixed to 0 before descriptor computation.
      - Optional Hellinger / RootSIFT normalization is applied.
    """
    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {fname}")
    if to_binary:
        img = toBinary(img)

    sift = create_sift()
    keypoints = sift.detect(img, None)
    for kp in keypoints:
        kp.angle = 0.0

    if len(keypoints) == 0:
        # Robustness fallback; it should rarely be used for the ICDAR binarized images.
        keypoints = dense_fallback_keypoints(img)

    _, desc = sift.compute(img, keypoints)
    if desc is None or len(desc) == 0:
        # Final safety fallback: one zero descriptor keeps the pipeline from crashing.
        desc = np.zeros((1, 128), dtype=np.float32)
    else:
        desc = desc.astype(np.float32, copy=False)

    if norm_hellinger:
        desc = rootsift_hellinger(desc)
    return desc


def descriptor_cache_file(fname: str, cache_dir: Optional[str], norm_hellinger: bool, to_binary: bool) -> Optional[str]:
    """Create a stable cache path for local descriptors of one image."""
    if not cache_dir:
        return None
    ensure_dir(cache_dir)
    try:
        mtime = os.path.getmtime(fname)
    except OSError:
        mtime = 0.0
    key = f"{os.path.abspath(fname)}|{mtime:.6f}|hell={int(norm_hellinger)}|bin={int(to_binary)}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    base = os.path.basename(fname).replace(os.sep, "_")
    return os.path.join(cache_dir, f"{base}.{digest}.npy")


def computeDescsCached(
    fname: str,
    norm_hellinger: bool = False,
    to_binary: bool = False,
    desc_cache_dir: Optional[str] = None,
) -> np.ndarray:
    """Compute local descriptors with an optional .npy cache to avoid repeated SIFT extraction."""
    cpath = descriptor_cache_file(fname, desc_cache_dir, norm_hellinger, to_binary)
    if cpath and os.path.exists(cpath):
        return np.load(cpath).astype(np.float32, copy=False)
    desc = computeDescs(fname, norm_hellinger=norm_hellinger, to_binary=to_binary)
    if cpath:
        np.save(cpath, desc.astype(np.float32, copy=False))
    return desc


# -----------------------------
# Dictionary / VLAD
# -----------------------------

def loadRandomDescriptors(
    files: Sequence[str],
    max_descriptors: int = 500_000,
    dict_sample_files: int = 0,
    to_binary: bool = True,
    desc_cache_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Load roughly max_descriptors random SIFT descriptors from the training images.

    If dict_sample_files <= 0, all training files are used. Otherwise a random subset of
    files is used. The descriptor quota is spread across the selected files.
    """
    if len(files) == 0:
        raise ValueError("No training files were provided.")

    if dict_sample_files and dict_sample_files > 0:
        n_files = min(dict_sample_files, len(files))
        file_indices = np.random.choice(len(files), size=n_files, replace=False)
        selected_files = [files[i] for i in file_indices]
    else:
        selected_files = list(files)

    descs_per_file = max(1, int(np.ceil(max_descriptors / float(len(selected_files)))))
    descriptors: List[np.ndarray] = []

    for path in tqdm(selected_files, desc="Sampling descriptors"):
        desc = computeDescsCached(path, norm_hellinger=True, to_binary=to_binary, desc_cache_dir=desc_cache_dir)
        n_take = min(len(desc), descs_per_file)
        if n_take <= 0:
            continue
        idx = np.random.choice(len(desc), size=n_take, replace=False)
        descriptors.append(desc[idx].astype(np.float32, copy=False))

    if not descriptors:
        raise RuntimeError("No descriptors could be extracted for dictionary learning.")

    descriptors = np.vstack(descriptors).astype(np.float32, copy=False)
    if len(descriptors) > max_descriptors:
        idx = np.random.choice(len(descriptors), size=max_descriptors, replace=False)
        descriptors = descriptors[idx]
    return descriptors


def dictionary(descriptors: np.ndarray, n_clusters: int = 100) -> np.ndarray:
    """Compute the VLAD codebook / dictionary using MiniBatchKMeans."""
    if descriptors.ndim != 2:
        raise ValueError("descriptors must have shape N x D")
    if len(descriptors) < n_clusters:
        raise ValueError(f"Need at least {n_clusters} descriptors, got {len(descriptors)}")

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=min(10_000, max(n_clusters * 20, 1_000)),
        n_init=3,
        max_iter=300,
        verbose=0,
        reassignment_ratio=0.01,
    )
    kmeans.fit(descriptors.astype(np.float32, copy=False))
    return kmeans.cluster_centers_.astype(np.float32)


def assignments(descriptors: np.ndarray, clusters: np.ndarray, return_labels: bool = False):
    """
    Compute hard nearest-cluster assignments.

    Returns a T x K one-hot matrix by default, matching the assignment statement.
    With return_labels=True, returns only the nearest cluster index for efficiency.
    """
    descriptors = np.ascontiguousarray(descriptors.astype(np.float32, copy=False))
    clusters = np.ascontiguousarray(clusters.astype(np.float32, copy=False))

    # pairwise_distances_argmin is fast and avoids allocating the full T x K distance matrix.
    labels = pairwise_distances_argmin(descriptors, clusters, metric="euclidean").astype(np.int32)
    if return_labels:
        return labels

    assignment = np.zeros((len(descriptors), len(clusters)), dtype=np.float32)
    assignment[np.arange(len(descriptors)), labels] = 1.0
    return assignment


def power_normalize(x: np.ndarray) -> np.ndarray:
    return (np.sign(x) * np.sqrt(np.abs(x))).astype(np.float32, copy=False)


def l2_normalize_vector(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    if n > EPS:
        return (x / n).astype(np.float32, copy=False)
    return x.astype(np.float32, copy=False)


def vlad(
    files: Sequence[str],
    mus: np.ndarray,
    powernorm: bool = False,
    to_binary: bool = True,
    intranorm: bool = False,
    desc_cache_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Compute one VLAD encoding per file.

    VLAD block k is sum_{x assigned to k} (x - mu_k).
    Optional intra-normalization normalizes each cluster block before global normalization.
    """
    K, D = mus.shape
    mus = mus.astype(np.float32, copy=False)
    encodings: List[np.ndarray] = []

    for path in tqdm(files, desc="Computing VLAD"):
        desc = computeDescsCached(path, norm_hellinger=True, to_binary=to_binary, desc_cache_dir=desc_cache_dir)
        labels = assignments(desc, mus, return_labels=True)

        f_enc = np.zeros((K, D), dtype=np.float32)
        for k in range(K):
            mask = labels == k
            if np.any(mask):
                f_enc[k] = np.sum(desc[mask] - mus[k], axis=0)

        if intranorm:
            f_enc = normalize(f_enc, norm="l2", axis=1, copy=False)

        f_enc = f_enc.reshape(-1)

        if powernorm:
            f_enc = power_normalize(f_enc)

        f_enc = l2_normalize_vector(f_enc)
        encodings.append(f_enc)

    return np.vstack(encodings).astype(np.float32, copy=False)


# -----------------------------
# Evaluation / postprocessing
# -----------------------------

def distances(encs: np.ndarray) -> np.ndarray:
    """Compute cosine distance matrix for L2-normalized global descriptors."""
    encs = normalize(np.asarray(encs, dtype=np.float32), norm="l2", axis=1, copy=False)
    sims = np.matmul(encs, encs.T).astype(np.float32, copy=False)
    dists = 1.0 - sims
    np.fill_diagonal(dists, np.finfo(np.float32).max)
    return dists


def evaluate(encs: np.ndarray, labels: Sequence[str], name: str = "experiment", verbose: bool = True) -> Dict[str, float]:
    """Evaluate Top-1 accuracy and mean average precision (mAP)."""
    labels = np.asarray(labels)
    dist_matrix = distances(encs)
    indices = np.argsort(dist_matrix, axis=1)

    n_encs = len(encs)
    ap_values = []
    correct = 0

    for r in range(n_encs):
        relevant_seen = 0
        precisions = []
        for rank in range(n_encs - 1):
            j = indices[r, rank]
            if labels[j] == labels[r]:
                relevant_seen += 1
                precisions.append(relevant_seen / float(rank + 1))
                if rank == 0:
                    correct += 1
        ap_values.append(float(np.mean(precisions)) if precisions else 0.0)

    result = {
        "name": name,
        "top1": float(correct) / float(n_encs),
        "mAP": float(np.mean(ap_values)),
    }
    if verbose:
        print(f"[{name}] Top-1 accuracy: {result['top1']:.6f} - mAP: {result['mAP']:.6f}")
    return result


def apply_pca_whitening(
    encs_train: np.ndarray,
    encs_test: np.ndarray,
    n_components: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """
    Fit PCA whitening on training descriptors and transform train/test descriptors.

    This avoids fitting the whitening transform on the test labels/writers.
    """
    max_components = min(encs_train.shape[0], encs_train.shape[1])
    if n_components is None or n_components <= 0:
        n_components = min(512, max_components)
    else:
        n_components = min(n_components, max_components)

    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    train_w = pca.fit_transform(encs_train).astype(np.float32)
    test_w = pca.transform(encs_test).astype(np.float32)
    train_w = normalize(train_w, norm="l2", axis=1)
    test_w = normalize(test_w, norm="l2", axis=1)
    return train_w.astype(np.float32), test_w.astype(np.float32), pca


def _fit_one_esvm(i: int, encs_test: np.ndarray, encs_train: np.ndarray, C: float, max_iter: int) -> np.ndarray:
    x_pos = encs_test[i : i + 1]
    X = np.vstack([x_pos, encs_train]).astype(np.float32, copy=False)
    y = np.zeros(X.shape[0], dtype=np.int32)
    y[0] = 1

    clf = LinearSVC(C=C, class_weight="balanced", max_iter=max_iter, dual=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X, y)
    w = clf.coef_.astype(np.float32)
    w = normalize(w, norm="l2", axis=1)
    return w[0].astype(np.float32)


def esvm(
    encs_test: np.ndarray,
    encs_train: np.ndarray,
    C: float = 1000.0,
    n_jobs: int = 1,
    max_iter: int = 10_000,
) -> np.ndarray:
    """Compute one Exemplar-SVM descriptor per test image."""
    encs_test = normalize(np.asarray(encs_test, dtype=np.float32), norm="l2", axis=1)
    encs_train = normalize(np.asarray(encs_train, dtype=np.float32), norm="l2", axis=1)

    if n_jobs != 1 and Parallel is not None:
        rows = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_fit_one_esvm)(i, encs_test, encs_train, C, max_iter)
            for i in tqdm(range(len(encs_test)), desc="E-SVM")
        )
    else:
        rows = [
            _fit_one_esvm(i, encs_test, encs_train, C, max_iter)
            for i in tqdm(range(len(encs_test)), desc="E-SVM")
        ]
    return np.vstack(rows).astype(np.float32, copy=False)


# -----------------------------
# Experiment runner
# -----------------------------

@dataclass(frozen=True)
class VladConfig:
    name: str
    powernorm: bool
    intranorm: bool


def encoding_filename(split: str, cfg: VladConfig, k: int) -> str:
    return f"enc_{split}_K{k}_{cfg.name}.pkl.gz"


def get_or_compute_vlad(
    split: str,
    files: Sequence[str],
    mus: np.ndarray,
    cfg: VladConfig,
    args,
) -> np.ndarray:
    path = cache_path(args.cache_dir, encoding_filename(split, cfg, args.clusters))
    if os.path.exists(path) and not args.overwrite:
        print(f"> loading cached {split} VLAD: {path}")
        return load_pickle_gz(path)
    print(f"> computing {split} VLAD: {cfg.name}")
    enc = vlad(files, mus, powernorm=cfg.powernorm, to_binary=args.to_binary, intranorm=cfg.intranorm, desc_cache_dir=args.desc_cache_dir)
    save_pickle_gz(enc, path)
    return enc


def write_results_csv(results: List[Dict[str, float]], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "top1", "mAP"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def write_report_txt(results: List[Dict[str, float]], args, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("Exercise 3.1: Writer Identification - Results Report\n")
        f.write("====================================================\n\n")
        f.write("Method summary\n")
        f.write("--------------\n")
        f.write("Local features are SIFT descriptors computed at SIFT keypoints. The orientation of every keypoint is fixed to 0 before descriptor computation. Descriptors are Hellinger-normalized by L1 normalization followed by signed square root. A MiniBatchKMeans dictionary is learned on randomly sampled training descriptors. Image-level descriptors are computed with VLAD, followed by optional power normalization and L2 normalization. Retrieval is evaluated on the test set using Top-1 accuracy and mean average precision (mAP). Exemplar classification trains one LinearSVC per test descriptor using that descriptor as the positive sample and all training descriptors as negatives; the L2-normalized SVM weight vector becomes the new descriptor.\n\n")
        f.write("Parameters\n")
        f.write("----------\n")
        f.write(f"Clusters K: {args.clusters}\n")
        f.write(f"Max dictionary descriptors: {args.max_dict_descriptors}\n")
        f.write(f"Dictionary sample files: {'all' if args.dict_sample_files <= 0 else args.dict_sample_files}\n")
        f.write(f"Binary preprocessing: {args.to_binary}\n")
        f.write(f"E-SVM C: {args.C}\n")
        f.write(f"PCA whitening components: {args.pca_components}\n\n")
        f.write("Results\n")
        f.write("-------\n")
        f.write("Experiment, Top-1, mAP\n")
        for r in results:
            f.write(f"{r['name']}, {r['top1']:.6f}, {r['mAP']:.6f}\n")
        f.write("\nDiscussion\n")
        f.write("----------\n")
        f.write("Plain VLAD gives the base retrieval performance. Power normalization is expected to reduce visual burstiness, where frequent local patterns dominate similarity. Intra-normalization normalizes each visual-word block separately and is an additional attempt to reduce the dominance of frequent clusters. PCA whitening is an additional post-processing experiment that decorrelates the global descriptors before the final L2 normalization. Exemplar-SVM changes the representation by learning a discriminative direction for each test image against the training set.\n")


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exercise 3.1 - Writer Identification with SIFT + VLAD + E-SVM")

    parser.add_argument("--labels_test", required=True, help="test label file")
    parser.add_argument("--labels_train", required=True, help="training label file")
    parser.add_argument("--in_test", required=True, help="folder with test images")
    parser.add_argument("--in_train", required=True, help="folder with training images")

    parser.add_argument("-str", "--suffix_train", default=".png", help="training image suffix")
    parser.add_argument("-ste", "--suffix_test", default=".jpg", help="test image suffix")
    parser.add_argument("--auto_suffix", action="store_true", default=True, help="try common suffixes if preferred suffix is missing")
    parser.add_argument("--no_auto_suffix", dest="auto_suffix", action="store_false", help="do not try alternative image suffixes")
    parser.add_argument("--to_binary", action="store_true", default=True, help="use/ensure Otsu binarization")
    parser.add_argument("--no_binary", dest="to_binary", action="store_false", help="do not binarize images")

    parser.add_argument("--clusters", type=int, default=100, help="number of visual words / KMeans clusters")
    parser.add_argument("--max_dict_descriptors", type=int, default=500_000, help="max sampled descriptors for dictionary")
    parser.add_argument("--dict_sample_files", type=int, default=0, help="number of training files to sample for dictionary; 0 means all")
    parser.add_argument("--cache_dir", default="cache_ex3", help="where to store dictionaries, encodings, and results")
    parser.add_argument("--desc_cache_dir", default=None, help="local SIFT descriptor cache; default: <cache_dir>/local_descs")
    parser.add_argument("--overwrite", action="store_true", help="overwrite cached codebook/encodings")

    parser.add_argument("--C", default=1000.0, type=float, help="C parameter for Exemplar-SVM")
    parser.add_argument("--svm_max_iter", default=10_000, type=int, help="maximum iterations for LinearSVC")
    parser.add_argument("--n_jobs", default=1, type=int, help="parallel jobs for E-SVM; use -1 for all CPUs")
    parser.add_argument("--skip_esvm", action="store_true", help="skip Exemplar-SVM experiment")

    parser.add_argument("--pca_whiten", action="store_true", help="run PCA whitening improvement experiment")
    parser.add_argument("--pca_components", type=int, default=512, help="number of PCA whitening components")

    parser.add_argument(
        "--experiment",
        choices=["basic", "all"],
        default="all",
        help="basic: plain VLAD + powernorm; all: also intranorm, optional PCA whitening, and E-SVM",
    )

    return parser.parse_args()


def main() -> None:
    args = parseArgs()
    np.random.seed(42)

    ensure_dir(args.cache_dir)
    if args.desc_cache_dir is None:
        args.desc_cache_dir = os.path.join(args.cache_dir, "local_descs")
    ensure_dir(args.desc_cache_dir)
    start_time = time.time()

    files_train, labels_train = getFiles(args.in_train, args.suffix_train, args.labels_train, args.auto_suffix)
    files_test, labels_test = getFiles(args.in_test, args.suffix_test, args.labels_test, args.auto_suffix)
    print(f"#train: {len(files_train)}")
    print(f"#test:  {len(files_test)}")
    check_files_exist(files_train, "training")
    check_files_exist(files_test, "test")

    # Dictionary / codebook
    mus_file = cache_path(args.cache_dir, f"mus_K{args.clusters}_D{args.max_dict_descriptors}_F{args.dict_sample_files}.pkl.gz")
    if os.path.exists(mus_file) and not args.overwrite:
        print(f"> loading cached dictionary: {mus_file}")
        mus = load_pickle_gz(mus_file)
    else:
        print("> sampling descriptors for dictionary")
        descriptors = loadRandomDescriptors(
            files_train,
            max_descriptors=args.max_dict_descriptors,
            dict_sample_files=args.dict_sample_files,
            to_binary=args.to_binary,
            desc_cache_dir=args.desc_cache_dir,
        )
        print(f"> sampled descriptors: {descriptors.shape}")
        print("> computing MiniBatchKMeans dictionary")
        mus = dictionary(descriptors, args.clusters)
        save_pickle_gz(mus, mus_file)
    print(f"> dictionary shape: {mus.shape}")

    configs = [
        VladConfig("plain_vlad", powernorm=False, intranorm=False),
        VladConfig("powernorm_vlad", powernorm=True, intranorm=False),
    ]
    if args.experiment == "all":
        configs.append(VladConfig("powernorm_intranorm_vlad", powernorm=True, intranorm=True))

    results: List[Dict[str, float]] = []
    cached_encs: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}

    for cfg in configs:
        enc_test = get_or_compute_vlad("test", files_test, mus, cfg, args)
        enc_train = None
        if args.experiment == "all" or (not args.skip_esvm):
            enc_train = get_or_compute_vlad("train", files_train, mus, cfg, args)
        cached_encs[cfg.name] = (enc_test, enc_train)
        results.append(evaluate(enc_test, labels_test, name=cfg.name))

    # PCA whitening improvement on the best normal VLAD variant available.
    if args.experiment == "all" and args.pca_whiten:
        base_name = "powernorm_intranorm_vlad"
        enc_test, enc_train = cached_encs[base_name]
        assert enc_train is not None
        pca_cache = cache_path(args.cache_dir, f"pca_whiten_{base_name}_C{args.pca_components}.pkl.gz")
        if os.path.exists(pca_cache) and not args.overwrite:
            print(f"> loading cached PCA-whitened descriptors: {pca_cache}")
            enc_train_w, enc_test_w = load_pickle_gz(pca_cache)
        else:
            print("> computing PCA whitening improvement")
            enc_train_w, enc_test_w, _ = apply_pca_whitening(enc_train, enc_test, n_components=args.pca_components)
            save_pickle_gz((enc_train_w, enc_test_w), pca_cache)
        results.append(evaluate(enc_test_w, labels_test, name=f"{base_name}_pca_whiten"))

    # Exemplar-SVM on the strongest non-PCA baseline by default.
    if not args.skip_esvm:
        esvm_base = "powernorm_intranorm_vlad" if args.experiment == "all" else "powernorm_vlad"
        enc_test, enc_train = cached_encs[esvm_base]
        assert enc_train is not None
        esvm_file = cache_path(args.cache_dir, f"esvm_{esvm_base}_C{args.C}_K{args.clusters}.pkl.gz")
        if os.path.exists(esvm_file) and not args.overwrite:
            print(f"> loading cached E-SVM descriptors: {esvm_file}")
            enc_test_esvm = load_pickle_gz(esvm_file)
        else:
            print(f"> computing E-SVM descriptors based on {esvm_base}")
            enc_test_esvm = esvm(enc_test, enc_train, C=args.C, n_jobs=args.n_jobs, max_iter=args.svm_max_iter)
            save_pickle_gz(enc_test_esvm, esvm_file)
        results.append(evaluate(enc_test_esvm, labels_test, name=f"esvm_{esvm_base}"))

    results_csv = cache_path(args.cache_dir, "results.csv")
    results_txt = cache_path(args.cache_dir, "results_report.txt")
    write_results_csv(results, results_csv)
    write_report_txt(results, args, results_txt)

    elapsed = time.time() - start_time
    print(f"> wrote {results_csv}")
    print(f"> wrote {results_txt}")
    print(f"> done in {elapsed / 60.0:.2f} minutes")


if __name__ == "__main__":
    main()
