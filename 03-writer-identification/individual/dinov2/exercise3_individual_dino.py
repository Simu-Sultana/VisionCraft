#!/usr/bin/env python3
"""
Individual experiment for CV Project Exercise 3.1:
DINOv2 patch descriptors + VLAD + optional PCA whitening / Exemplar-SVM.

This file is intentionally separate from the required/group SIFT solution.
Use this only for the individual improvement section of the report.

Idea:
  - Replace handcrafted SIFT local descriptors with self-supervised DINOv2 patch tokens.
  - Keep the same retrieval pipeline: codebook -> VLAD -> normalization -> mAP evaluation.

Notes:
  - First run will download DINOv2 through torch.hub.
  - On Mac, the script will use MPS automatically when available.
  - For a first full run, use K=64 and image_size=224 or 336. Then try 518 if time allows.
"""

import argparse
import csv
import gzip
import hashlib
import os
import pickle
import shlex
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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
except Exception:  # pragma: no cover
    Parallel = None
    delayed = None

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
    for suffix in KNOWN_SUFFIXES:
        if file_name.endswith(suffix):
            return file_name[: -len(suffix)]
    return file_name


def resolve_file(folder: str, stem: str, preferred_suffix: str, auto_suffix: bool = True) -> str:
    preferred = os.path.join(folder, stem + preferred_suffix)
    if not auto_suffix or os.path.exists(preferred):
        return preferred
    for suffix in [preferred_suffix, ".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        candidate = os.path.join(folder, stem + suffix)
        if os.path.exists(candidate):
            return candidate
    return preferred


def get_files(folder: str, suffix: str, labelfile: str, auto_suffix: bool = True) -> Tuple[List[str], List[str]]:
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
            files.append(resolve_file(folder, stem, suffix, auto_suffix=auto_suffix))
            labels.append(parts[1])
    return files, labels


def check_files_exist(files: Sequence[str], split_name: str, max_examples: int = 5) -> None:
    missing = [p for p in files if not os.path.exists(p)]
    if missing:
        examples = "\n".join(f"  - {p}" for p in missing[:max_examples])
        raise FileNotFoundError(
            f"{len(missing)} {split_name} images were not found. Examples:\n{examples}\n"
            "Check --in_train/--in_test and --suffix_train/--suffix_test."
        )


def to_binary(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    unique = np.unique(img)
    if unique.size <= 2:
        if unique.max() <= 1:
            img = (img * 255).astype(np.uint8)
        return img
    _, out = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def choose_device(device_arg: str):
    import torch

    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DinoExtractor:
    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        image_size: int = 224,
        device: str = "auto",
        invert: bool = True,
        to_binary_flag: bool = True,
    ):
        import torch

        self.torch = torch
        self.image_size = int(image_size)
        # DINOv2 patch size is 14, so use a multiple of 14.
        if self.image_size % 14 != 0:
            self.image_size = int(round(self.image_size / 14.0) * 14)
        self.invert = invert
        self.to_binary_flag = to_binary_flag
        self.device = choose_device(device)
        print(f"> loading DINOv2 model: {model_name} on {self.device}")
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(self.device)

        # ImageNet normalization used by DINOv2 examples.
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def _load_tensor(self, fname: str):
        torch = self.torch
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {fname}")
        if self.to_binary_flag:
            img = to_binary(img)
        if self.invert:
            # For neural features on binarized documents, script becomes white and background black.
            img = 255 - img
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.repeat(img[None, :, :], 3, axis=0)  # 3 x H x W
        x = torch.from_numpy(img).unsqueeze(0).to(self.device)
        x = (x - self.mean) / self.std
        return x

    def compute(self, fname: str) -> np.ndarray:
        torch = self.torch
        x = self._load_tensor(fname)
        with torch.no_grad():
            # Most DINOv2 hub models support forward_features and return normalized patch tokens.
            try:
                out = self.model.forward_features(x)
                if isinstance(out, dict) and "x_norm_patchtokens" in out:
                    tokens = out["x_norm_patchtokens"]  # 1 x P x D
                else:
                    tokens = self.model.get_intermediate_layers(x, n=1, reshape=False, return_class_token=False)[0]
            except Exception:
                tokens = self.model.get_intermediate_layers(x, n=1, reshape=False, return_class_token=False)[0]
        desc = tokens.squeeze(0).detach().float().cpu().numpy().astype(np.float32)
        # L2-normalize patch descriptors before VLAD. This makes cosine/Euclidean behavior more stable.
        desc = normalize(desc, norm="l2", axis=1).astype(np.float32)
        return desc


def descriptor_cache_file(fname: str, cache_dir: str, model_name: str, image_size: int, invert: bool, to_binary_flag: bool) -> str:
    ensure_dir(cache_dir)
    try:
        mtime = os.path.getmtime(fname)
    except OSError:
        mtime = 0.0
    key = f"{os.path.abspath(fname)}|{mtime:.6f}|model={model_name}|size={image_size}|invert={int(invert)}|bin={int(to_binary_flag)}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    base = os.path.basename(fname).replace(os.sep, "_")
    return os.path.join(cache_dir, f"{base}.{digest}.dino.npy")


def compute_desc_cached(fname: str, extractor: DinoExtractor, args) -> np.ndarray:
    cpath = descriptor_cache_file(fname, args.desc_cache_dir, args.dino_model, args.image_size, args.invert, args.to_binary)
    if os.path.exists(cpath):
        return np.load(cpath).astype(np.float32, copy=False)
    desc = extractor.compute(fname)
    np.save(cpath, desc.astype(np.float32, copy=False))
    return desc


def load_random_descriptors(files: Sequence[str], extractor: DinoExtractor, args) -> np.ndarray:
    if args.dict_sample_files and args.dict_sample_files > 0:
        n_files = min(args.dict_sample_files, len(files))
        idx_files = np.random.choice(len(files), size=n_files, replace=False)
        selected = [files[i] for i in idx_files]
    else:
        selected = list(files)

    descs_per_file = max(1, int(np.ceil(args.max_dict_descriptors / float(len(selected)))))
    chunks: List[np.ndarray] = []
    for path in tqdm(selected, desc="Sampling DINO descriptors"):
        desc = compute_desc_cached(path, extractor, args)
        n_take = min(len(desc), descs_per_file)
        idx = np.random.choice(len(desc), size=n_take, replace=False)
        chunks.append(desc[idx])

    descriptors = np.vstack(chunks).astype(np.float32, copy=False)
    if len(descriptors) > args.max_dict_descriptors:
        idx = np.random.choice(len(descriptors), size=args.max_dict_descriptors, replace=False)
        descriptors = descriptors[idx]
    return descriptors


def dictionary(descriptors: np.ndarray, n_clusters: int) -> np.ndarray:
    if len(descriptors) < n_clusters:
        raise ValueError(f"Need at least {n_clusters} descriptors, got {len(descriptors)}")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=min(8192, max(n_clusters * 20, 1024)),
        n_init=3,
        max_iter=300,
        verbose=0,
        reassignment_ratio=0.01,
    )
    kmeans.fit(descriptors.astype(np.float32, copy=False))
    return kmeans.cluster_centers_.astype(np.float32)


def power_normalize(x: np.ndarray) -> np.ndarray:
    return (np.sign(x) * np.sqrt(np.abs(x))).astype(np.float32, copy=False)


def l2_normalize_vector(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    if n > EPS:
        return (x / n).astype(np.float32, copy=False)
    return x.astype(np.float32, copy=False)


@dataclass(frozen=True)
class VladConfig:
    name: str
    powernorm: bool
    intranorm: bool


def vlad(files: Sequence[str], mus: np.ndarray, cfg: VladConfig, extractor: DinoExtractor, args) -> np.ndarray:
    K, D = mus.shape
    mus = mus.astype(np.float32, copy=False)
    encodings: List[np.ndarray] = []
    for path in tqdm(files, desc=f"Computing {cfg.name}"):
        desc = compute_desc_cached(path, extractor, args)
        labels = pairwise_distances_argmin(desc, mus, metric="euclidean").astype(np.int32)
        enc = np.zeros((K, D), dtype=np.float32)
        for k in range(K):
            mask = labels == k
            if np.any(mask):
                enc[k] = np.sum(desc[mask] - mus[k], axis=0)
        if cfg.intranorm:
            enc = normalize(enc, norm="l2", axis=1, copy=False)
        enc = enc.reshape(-1)
        if cfg.powernorm:
            enc = power_normalize(enc)
        enc = l2_normalize_vector(enc)
        encodings.append(enc)
    return np.vstack(encodings).astype(np.float32, copy=False)


def distances(encs: np.ndarray) -> np.ndarray:
    encs = normalize(np.asarray(encs, dtype=np.float32), norm="l2", axis=1, copy=False)
    sims = np.matmul(encs, encs.T).astype(np.float32, copy=False)
    dists = 1.0 - sims
    np.fill_diagonal(dists, np.finfo(np.float32).max)
    return dists


def evaluate(encs: np.ndarray, labels: Sequence[str], name: str) -> Dict[str, float]:
    labels = np.asarray(labels)
    d = distances(encs)
    order = np.argsort(d, axis=1)
    n = len(labels)
    ap_values = []
    correct = 0
    for i in range(n):
        rel_seen = 0
        precisions = []
        for rank in range(n - 1):
            j = order[i, rank]
            if labels[j] == labels[i]:
                rel_seen += 1
                precisions.append(rel_seen / float(rank + 1))
                if rank == 0:
                    correct += 1
        ap_values.append(float(np.mean(precisions)) if precisions else 0.0)
    result = {"name": name, "top1": float(correct) / n, "mAP": float(np.mean(ap_values))}
    print(f"[{name}] Top-1 accuracy: {result['top1']:.6f} - mAP: {result['mAP']:.6f}")
    return result


def apply_pca_whitening(encs_train: np.ndarray, encs_test: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    max_components = min(encs_train.shape[0], encs_train.shape[1])
    n_components = min(n_components, max_components)
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    tr = pca.fit_transform(encs_train).astype(np.float32)
    te = pca.transform(encs_test).astype(np.float32)
    tr = normalize(tr, norm="l2", axis=1)
    te = normalize(te, norm="l2", axis=1)
    return tr.astype(np.float32), te.astype(np.float32)


def _fit_one_esvm(i: int, encs_test: np.ndarray, encs_train: np.ndarray, C: float, max_iter: int) -> np.ndarray:
    X = np.vstack([encs_test[i : i + 1], encs_train]).astype(np.float32)
    y = np.zeros(X.shape[0], dtype=np.int32)
    y[0] = 1
    clf = LinearSVC(C=C, class_weight="balanced", max_iter=max_iter, dual=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X, y)
    w = normalize(clf.coef_.astype(np.float32), norm="l2", axis=1)
    return w[0].astype(np.float32)


def esvm(encs_test: np.ndarray, encs_train: np.ndarray, C: float, n_jobs: int, max_iter: int) -> np.ndarray:
    encs_test = normalize(np.asarray(encs_test, dtype=np.float32), norm="l2", axis=1)
    encs_train = normalize(np.asarray(encs_train, dtype=np.float32), norm="l2", axis=1)
    if n_jobs != 1 and Parallel is not None:
        rows = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_fit_one_esvm)(i, encs_test, encs_train, C, max_iter)
            for i in tqdm(range(len(encs_test)), desc="E-SVM")
        )
    else:
        rows = [_fit_one_esvm(i, encs_test, encs_train, C, max_iter) for i in tqdm(range(len(encs_test)), desc="E-SVM")]
    return np.vstack(rows).astype(np.float32)


def write_results_csv(results: List[Dict[str, float]], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "top1", "mAP"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def write_report(results: List[Dict[str, float]], args, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("Exercise 3.1 Individual Experiment - DINOv2 + VLAD\n")
        f.write("===================================================\n\n")
        f.write("Method summary\n")
        f.write("--------------\n")
        f.write("This individual experiment replaces SIFT descriptors with self-supervised DINOv2 patch-token descriptors. The remaining retrieval pipeline is kept similar to the group solution: a MiniBatchKMeans codebook is learned from training descriptors, image-level representations are generated with VLAD, and retrieval is evaluated with Top-1 accuracy and mAP. Optional PCA whitening and Exemplar-SVM can be applied afterwards.\n\n")
        f.write("Parameters\n")
        f.write("----------\n")
        f.write(f"DINOv2 model: {args.dino_model}\n")
        f.write(f"Image size: {args.image_size}\n")
        f.write(f"Invert image for neural features: {args.invert}\n")
        f.write(f"Clusters K: {args.clusters}\n")
        f.write(f"Max dictionary descriptors: {args.max_dict_descriptors}\n")
        f.write(f"Dictionary sample files: {'all' if args.dict_sample_files <= 0 else args.dict_sample_files}\n")
        f.write(f"PCA whitening components: {args.pca_components}\n")
        f.write(f"E-SVM C: {args.C}\n\n")
        f.write("Results\n")
        f.write("-------\n")
        f.write("Experiment, Top-1, mAP\n")
        for r in results:
            f.write(f"{r['name']}, {r['top1']:.6f}, {r['mAP']:.6f}\n")
        f.write("\nDiscussion notes\n")
        f.write("----------------\n")
        f.write("The purpose of this experiment is to test whether learned self-supervised patch descriptors capture writer-specific texture and stroke patterns better than handcrafted SIFT. If performance improves, it suggests the learned representation transfers well to historical handwriting. If performance does not improve, possible explanations are domain mismatch, too much resizing of document images, binarized input being far from natural images, limited codebook size, or insufficient tuning of DINO patch descriptors for this retrieval task.\n")


def get_or_compute_vlad(split: str, files: Sequence[str], mus: np.ndarray, cfg: VladConfig, extractor: DinoExtractor, args) -> np.ndarray:
    fname = f"enc_{split}_{args.dino_model}_S{args.image_size}_K{args.clusters}_{cfg.name}.pkl.gz"
    path = cache_path(args.cache_dir, fname)
    if os.path.exists(path) and not args.overwrite:
        print(f"> loading cached {split} encoding: {path}")
        return load_pickle_gz(path)
    enc = vlad(files, mus, cfg, extractor, args)
    save_pickle_gz(enc, path)
    return enc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Individual DINOv2 + VLAD experiment for Exercise 3.1")
    parser.add_argument("--labels_test", required=True)
    parser.add_argument("--labels_train", required=True)
    parser.add_argument("--in_test", required=True)
    parser.add_argument("--in_train", required=True)
    parser.add_argument("--suffix_train", default=".png")
    parser.add_argument("--suffix_test", default=".jpg")
    parser.add_argument("--auto_suffix", action="store_true", default=True)
    parser.add_argument("--no_auto_suffix", dest="auto_suffix", action="store_false")

    parser.add_argument("--dino_model", default="dinov2_vits14", choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"])
    parser.add_argument("--image_size", type=int, default=224, help="input size, rounded to a multiple of 14")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda")
    parser.add_argument("--invert", action="store_true", default=True, help="invert binary documents: white script, black background")
    parser.add_argument("--no_invert", dest="invert", action="store_false")
    parser.add_argument("--to_binary", action="store_true", default=True)
    parser.add_argument("--no_binary", dest="to_binary", action="store_false")

    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--max_dict_descriptors", type=int, default=100_000)
    parser.add_argument("--dict_sample_files", type=int, default=0)
    parser.add_argument("--cache_dir", default="cache_ex3_dino")
    parser.add_argument("--desc_cache_dir", default=None)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--pca_whiten", action="store_true")
    parser.add_argument("--pca_components", type=int, default=512)
    parser.add_argument("--skip_esvm", action="store_true", default=True)
    parser.add_argument("--run_esvm", dest="skip_esvm", action="store_false")
    parser.add_argument("--C", type=float, default=1000.0)
    parser.add_argument("--svm_max_iter", type=int, default=10_000)
    parser.add_argument("--n_jobs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(42)
    ensure_dir(args.cache_dir)
    if args.desc_cache_dir is None:
        args.desc_cache_dir = os.path.join(args.cache_dir, "dino_local_descs")
    ensure_dir(args.desc_cache_dir)
    start = time.time()

    files_train, labels_train = get_files(args.in_train, args.suffix_train, args.labels_train, args.auto_suffix)
    files_test, labels_test = get_files(args.in_test, args.suffix_test, args.labels_test, args.auto_suffix)
    print(f"#train: {len(files_train)}")
    print(f"#test:  {len(files_test)}")
    check_files_exist(files_train, "training")
    check_files_exist(files_test, "test")

    extractor = DinoExtractor(
        model_name=args.dino_model,
        image_size=args.image_size,
        device=args.device,
        invert=args.invert,
        to_binary_flag=args.to_binary,
    )

    mus_file = cache_path(args.cache_dir, f"mus_{args.dino_model}_S{args.image_size}_K{args.clusters}_D{args.max_dict_descriptors}_F{args.dict_sample_files}.pkl.gz")
    if os.path.exists(mus_file) and not args.overwrite:
        print(f"> loading cached dictionary: {mus_file}")
        mus = load_pickle_gz(mus_file)
    else:
        print("> sampling DINO descriptors for dictionary")
        desc = load_random_descriptors(files_train, extractor, args)
        print(f"> sampled descriptors: {desc.shape}")
        print("> computing MiniBatchKMeans dictionary")
        mus = dictionary(desc, args.clusters)
        save_pickle_gz(mus, mus_file)
    print(f"> dictionary shape: {mus.shape}")

    configs = [
        VladConfig("dino_plain_vlad", powernorm=False, intranorm=False),
        VladConfig("dino_powernorm_vlad", powernorm=True, intranorm=False),
        VladConfig("dino_powernorm_intranorm_vlad", powernorm=True, intranorm=True),
    ]

    results: List[Dict[str, float]] = []
    enc_train_best = None
    enc_test_best = None
    for cfg in configs:
        enc_test = get_or_compute_vlad("test", files_test, mus, cfg, extractor, args)
        enc_train = get_or_compute_vlad("train", files_train, mus, cfg, extractor, args)
        results.append(evaluate(enc_test, labels_test, cfg.name))
        if cfg.name == "dino_powernorm_intranorm_vlad":
            enc_train_best = enc_train
            enc_test_best = enc_test

    assert enc_train_best is not None and enc_test_best is not None

    if args.pca_whiten:
        pca_file = cache_path(args.cache_dir, f"pca_dino_S{args.image_size}_K{args.clusters}_C{args.pca_components}.pkl.gz")
        if os.path.exists(pca_file) and not args.overwrite:
            print(f"> loading cached PCA descriptors: {pca_file}")
            enc_train_w, enc_test_w = load_pickle_gz(pca_file)
        else:
            print("> computing PCA whitening")
            enc_train_w, enc_test_w = apply_pca_whitening(enc_train_best, enc_test_best, args.pca_components)
            save_pickle_gz((enc_train_w, enc_test_w), pca_file)
        results.append(evaluate(enc_test_w, labels_test, "dino_powernorm_intranorm_vlad_pca_whiten"))
        esvm_train, esvm_test = enc_train_w, enc_test_w
        esvm_base_name = "dino_pca_whiten"
    else:
        esvm_train, esvm_test = enc_train_best, enc_test_best
        esvm_base_name = "dino_powernorm_intranorm_vlad"

    if not args.skip_esvm:
        esvm_file = cache_path(args.cache_dir, f"esvm_{esvm_base_name}_S{args.image_size}_K{args.clusters}_C{args.C}.pkl.gz")
        if os.path.exists(esvm_file) and not args.overwrite:
            print(f"> loading cached E-SVM descriptors: {esvm_file}")
            enc_esvm = load_pickle_gz(esvm_file)
        else:
            print(f"> computing E-SVM on {esvm_base_name}")
            enc_esvm = esvm(esvm_test, esvm_train, C=args.C, n_jobs=args.n_jobs, max_iter=args.svm_max_iter)
            save_pickle_gz(enc_esvm, esvm_file)
        results.append(evaluate(enc_esvm, labels_test, f"esvm_{esvm_base_name}"))

    write_results_csv(results, cache_path(args.cache_dir, "results_dino.csv"))
    write_report(results, args, cache_path(args.cache_dir, "results_report_dino.txt"))
    print(f"> wrote {cache_path(args.cache_dir, 'results_dino.csv')}")
    print(f"> wrote {cache_path(args.cache_dir, 'results_report_dino.txt')}")
    print(f"> done in {(time.time() - start) / 60.0:.2f} minutes")


if __name__ == "__main__":
    main()
