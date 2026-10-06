from collections.abc import Callable
from typing import Final

import numpy as np
import pandas as pd

from cvproj_exc.config import Config

UNKNOWN_LABEL: Final[int] = -1
_EPS: Final[float] = 1e-12


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < _EPS] = 1.0
    return (x - mean) / std, mean, std


def _standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32) - mean) / std


def _pairwise_squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True).T
    d2 = a2 + b2 - 2.0 * (a @ b.T)
    return np.maximum(d2, 0.0)


def _class_centroids(x: np.ndarray, y: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.vstack([x[y == label].mean(axis=0) for label in labels]).astype(np.float32)


def _calibrate_distance_threshold(
    x: np.ndarray, y: np.ndarray, known_labels: np.ndarray, known_centroids: np.ndarray
) -> float:
    """Calibrate an unknown threshold from known and known-unknown training samples."""
    known_mask = y != UNKNOWN_LABEL
    unknown_mask = ~known_mask
    if not np.any(known_mask):
        return 0.0

    # Distance of known samples to their own class centroid.
    label_to_index = {label: i for i, label in enumerate(known_labels)}
    own_centroid_indices = np.array([label_to_index[label] for label in y[known_mask]], dtype=int)
    known_dist = np.sqrt(
        np.sum((x[known_mask] - known_centroids[own_centroid_indices]) ** 2, axis=1)
    )

    conservative_known_threshold = float(np.percentile(known_dist, 97.5))

    if np.any(unknown_mask):
        unknown_to_known = np.sqrt(_pairwise_squared_distances(x[unknown_mask], known_centroids))
        unknown_nearest = np.min(unknown_to_known, axis=1)
        # Midpoint between the upper tail of known distances and the lower tail of unknown distances.
        # This keeps most known classes accepted but still rejects clear unknowns.
        unknown_low = float(np.percentile(unknown_nearest, 10.0))
        if unknown_low > conservative_known_threshold:
            return 0.5 * (conservative_known_threshold + unknown_low)

    return conservative_known_threshold


def spl_training(
    x_train: np.ndarray, y_train: np.ndarray
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Single pseudo-label open-set model.

    Known classes are represented by class centroids.  All KUC samples share one additional unknown
    centroid.  A calibrated distance threshold rejects samples that are too far from all known class
    centroids, which also helps with unknown-unknown classes.
    """
    x_scaled, mean, std = _standardize_fit(x_train)
    y_train = np.asarray(y_train, dtype=int)

    known_labels = np.unique(y_train[y_train != UNKNOWN_LABEL])
    if known_labels.size == 0:
        def only_unknown_predict_fn(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            n = len(x_test)
            return np.full(n, UNKNOWN_LABEL, dtype=int), np.zeros(n, dtype=float)
        return only_unknown_predict_fn

    known_centroids = _class_centroids(x_scaled, y_train, known_labels)
    threshold = _calibrate_distance_threshold(x_scaled, y_train, known_labels, known_centroids)

    unknown_centroid = None
    if np.any(y_train == UNKNOWN_LABEL):
        unknown_centroid = x_scaled[y_train == UNKNOWN_LABEL].mean(axis=0, keepdims=True)

    def spl_predict_fn(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_test_scaled = _standardize_apply(x_test, mean, std)
        known_dist = np.sqrt(_pairwise_squared_distances(x_test_scaled, known_centroids))
        best_known_idx = np.argmin(known_dist, axis=1)
        best_known_dist = known_dist[np.arange(len(x_test_scaled)), best_known_idx]
        y_pred = known_labels[best_known_idx].astype(int)

        if unknown_centroid is not None:
            unknown_dist = np.sqrt(_pairwise_squared_distances(x_test_scaled, unknown_centroid)).ravel()
            y_pred[unknown_dist < best_known_dist] = UNKNOWN_LABEL

        y_pred[best_known_dist > threshold] = UNKNOWN_LABEL
        y_score = -best_known_dist.astype(float)
        return y_pred.astype(int), y_score

    return spl_predict_fn


def mpl_training(
    x_train: np.ndarray, y_train: np.ndarray
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Multi pseudo-label open-set model.

    Every KUC sample is kept as an individual unknown prototype.  Predictions are made by nearest
    prototype; if the nearest prototype is a KUC prototype, the output is mapped to -1.  A distance
    threshold to the known class centroids rejects far-away UUC samples.
    """
    x_scaled, mean, std = _standardize_fit(x_train)
    y_train = np.asarray(y_train, dtype=int)

    known_labels = np.unique(y_train[y_train != UNKNOWN_LABEL])
    if known_labels.size == 0:
        def only_unknown_predict_fn(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            n = len(x_test)
            return np.full(n, UNKNOWN_LABEL, dtype=int), np.zeros(n, dtype=float)
        return only_unknown_predict_fn

    known_centroids = _class_centroids(x_scaled, y_train, known_labels)
    threshold = _calibrate_distance_threshold(x_scaled, y_train, known_labels, known_centroids)

    prototype_x = x_scaled.astype(np.float32)
    prototype_y = y_train.astype(int)

    def mpl_predict_fn(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_test_scaled = _standardize_apply(x_test, mean, std)

        proto_dist = np.sqrt(_pairwise_squared_distances(x_test_scaled, prototype_x))
        nn_index = np.argmin(proto_dist, axis=1)
        y_pred = prototype_y[nn_index].astype(int)

        # Unknown-unknown safeguard based on nearest known class centroid.
        known_dist = np.sqrt(_pairwise_squared_distances(x_test_scaled, known_centroids))
        best_known_dist = np.min(known_dist, axis=1)
        best_known_idx = np.argmin(known_dist, axis=1)

        # When the nearest prototype is known but the query is very far from all known centroids,
        # reject it as unknown.  Otherwise keep the nearest-prototype class prediction.
        y_pred[best_known_dist > threshold] = UNKNOWN_LABEL

        # If a nearest unknown prototype and a close known centroid disagree, prefer the known class
        # only when it is clearly close enough. This avoids over-rejecting borderline known samples.
        nearest_proto_dist = proto_dist[np.arange(len(x_test_scaled)), nn_index]
        unknown_by_proto = prototype_y[nn_index] == UNKNOWN_LABEL
        prefer_known = unknown_by_proto & (best_known_dist <= 0.85 * nearest_proto_dist)
        y_pred[prefer_known] = known_labels[best_known_idx[prefer_known]]

        y_score = -best_known_dist.astype(float)
        return y_pred.astype(int), y_score

    return mpl_predict_fn


def load_challenge_train_data() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(Config.CHAL_TRAIN_DATA, header=None).values
    x = df[:, :-1]
    y = df[:, -1].astype(int)
    return x, y


def main():
    x_train, y_train = load_challenge_train_data()
    spl_predict_fn = spl_training(x_train, y_train)
    mpl_predict_fn = mpl_training(x_train, y_train)

    x_test = np.random.rand(50, x_train.shape[1])
    y_test = np.random.randint(-1, 5, 50)
    for predict_fn in (spl_predict_fn, mpl_predict_fn):
        y_pred, _ = predict_fn(x_test)
        print("Acc: {}".format(np.equal(y_test, y_pred).sum() / len(x_test)))


if __name__ == "__main__":
    main()
