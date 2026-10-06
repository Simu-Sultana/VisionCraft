import os
import pickle

import cv2
import numpy as np

from cvproj_exc.config import Config


class FaceNet:

    def __init__(self):
        self.facenet = cv2.dnn.readNetFromONNX(str(Config.RESNET50))

    def predict(self, face):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) - (131.0912, 103.8827, 91.4953)
        reshaped = np.moveaxis(face, 2, 0)
        reshaped = np.expand_dims(reshaped, axis=0)
        self.facenet.setInput(reshaped)
        embedding = np.squeeze(self.facenet.forward())
        norm = np.linalg.norm(embedding)
        if norm == 0.0:
            return embedding
        return embedding / norm

    @classmethod
    @property
    def embedding_dimensionality(cls):
        return 128


class FaceRecognizer:

    def __init__(self, num_neighbours=3, max_distance=1.02, min_prob=0.34):
        self.facenet = FaceNet()
        self.num_neighbours = max(1, int(num_neighbours))
        self.max_distance = float(max_distance)
        self.min_prob = float(min_prob)

        self.labels = []
        self.embeddings = np.empty((0, FaceNet.embedding_dimensionality), dtype=np.float32)

        if os.path.exists(Config.REC_GALLERY):
            self.load()

    def save(self):
        print("FaceRecognizer saving: {}".format(Config.REC_GALLERY))
        with open(Config.REC_GALLERY, "wb") as f:
            pickle.dump((self.labels, self.embeddings), f)

    def load(self):
        print("FaceRecognizer loading: {}".format(Config.REC_GALLERY))
        with open(Config.REC_GALLERY, "rb") as f:
            (self.labels, self.embeddings) = pickle.load(f)
        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)

    def partial_fit(self, face, label):
        """Add one labeled face to the gallery.

        The exercise asks to store two embeddings: one from the original BGR face and one from a
        grayscale version converted back to three channels for FaceNet.
        """
        color_embedding = self.facenet.predict(face).astype(np.float32)
        gray_face = self._to_three_channel_gray(face)
        gray_embedding = self.facenet.predict(gray_face).astype(np.float32)

        self.embeddings = np.vstack([self.embeddings, color_embedding, gray_embedding])
        self.labels.extend([label, label])

    def predict(self, face) -> tuple[str, float, float]:
        if len(self.labels) == 0 or self.embeddings.shape[0] == 0:
            return "unknown", 0.0, float("inf")

        query_embeddings = np.vstack(
            [self.facenet.predict(face), self.facenet.predict(self._to_three_channel_gray(face))]
        ).astype(np.float32)

        # Use both query representations.  For each gallery item, keep the smaller distance from
        # the color-query and grayscale-query embeddings.
        distances = np.linalg.norm(
            self.embeddings[None, :, :] - query_embeddings[:, None, :], axis=2
        ).min(axis=0)

        k = min(self.num_neighbours, len(distances))
        nn_indices = np.argsort(distances)[:k]
        nn_labels = np.asarray([self.labels[i] for i in nn_indices], dtype=object)
        nn_distances = distances[nn_indices]

        unique_labels, counts = np.unique(nn_labels, return_counts=True)
        max_count = np.max(counts)
        tied_labels = unique_labels[counts == max_count]

        # Tie break: choose the tied class whose nearest neighbour is closest.
        best_label = min(
            tied_labels,
            key=lambda lab: float(np.min(nn_distances[nn_labels == lab])),
        )
        posterior = float(max_count / k)
        distance_to_prediction = float(np.min(nn_distances[nn_labels == best_label]))

        if distance_to_prediction > self.max_distance or posterior < self.min_prob:
            return "unknown", posterior, distance_to_prediction
        return str(best_label), posterior, distance_to_prediction

    @staticmethod
    def _to_three_channel_gray(face):
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class FaceClustering:

    def __init__(self, num_clusters=2, max_iter=200, n_init=10, tol=1e-6):
        self.facenet = FaceNet()

        self.embeddings = np.empty((0, FaceNet.embedding_dimensionality), dtype=np.float32)
        self.num_clusters = int(num_clusters)
        self.cluster_center = np.empty(
            (self.num_clusters, FaceNet.embedding_dimensionality), dtype=np.float32
        )
        self.cluster_membership = []
        self.objective_history = []
        self.max_iter = int(max_iter)
        self.n_init = max(1, int(n_init))
        self.tol = float(tol)

        if os.path.exists(Config.CLUSTER_GALLERY):
            self.load()

    def save(self):
        print("FaceClustering saving: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.CLUSTER_GALLERY, "wb") as f:
            pickle.dump(
                (
                    self.embeddings,
                    self.num_clusters,
                    self.cluster_center,
                    self.cluster_membership,
                    self.objective_history,
                ),
                f,
            )

    def load(self):
        print("FaceClustering loading: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.CLUSTER_GALLERY, "rb") as f:
            loaded = pickle.load(f)
        if len(loaded) == 4:
            (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership) = loaded
            self.objective_history = []
        else:
            (
                self.embeddings,
                self.num_clusters,
                self.cluster_center,
                self.cluster_membership,
                self.objective_history,
            ) = loaded
        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)
        self.cluster_center = np.asarray(self.cluster_center, dtype=np.float32)

    def partial_fit(self, face):
        embedding = self.facenet.predict(face).astype(np.float32)
        self.embeddings = np.vstack([self.embeddings, embedding])

    def fit(self):
        if self.embeddings.shape[0] < self.num_clusters:
            raise ValueError(
                f"Need at least {self.num_clusters} embeddings, got {self.embeddings.shape[0]}."
            )

        rng = np.random.default_rng(42)
        best_objective = float("inf")
        best_centers = None
        best_labels = None
        best_history = []

        # Multiple restarts make k-means less sensitive to unlucky initialization.
        for _ in range(self.n_init):
            init_indices = rng.choice(self.embeddings.shape[0], size=self.num_clusters, replace=False)
            centers = self.embeddings[init_indices].copy()
            labels = np.full(self.embeddings.shape[0], -1, dtype=int)
            history = []

            for _ in range(self.max_iter):
                sq_distances = self._squared_distances(self.embeddings, centers)
                new_labels = np.argmin(sq_distances, axis=1)

                new_centers = centers.copy()
                for cluster_idx in range(self.num_clusters):
                    members = self.embeddings[new_labels == cluster_idx]
                    if len(members) > 0:
                        new_centers[cluster_idx] = members.mean(axis=0)
                    else:
                        # Empty cluster: re-seed with the point that currently has the largest error.
                        farthest_idx = int(np.argmax(np.min(sq_distances, axis=1)))
                        new_centers[cluster_idx] = self.embeddings[farthest_idx]

                final_sq_distances = self._squared_distances(self.embeddings, new_centers)
                objective = float(np.sum(final_sq_distances[np.arange(len(new_labels)), new_labels]))
                history.append(objective)

                center_shift = float(np.linalg.norm(new_centers - centers))
                if np.array_equal(new_labels, labels) or center_shift <= self.tol:
                    centers = new_centers
                    labels = new_labels
                    break

                centers = new_centers
                labels = new_labels

            if history and history[-1] < best_objective:
                best_objective = history[-1]
                best_centers = centers.copy()
                best_labels = labels.copy()
                best_history = history

        self.cluster_center = best_centers.astype(np.float32)
        self.cluster_membership = best_labels.tolist()
        self.objective_history = best_history
        return self.objective_history

    def predict(self, face) -> tuple[int, np.ndarray]:
        if self.cluster_center.size == 0:
            raise ValueError("FaceClustering has no cluster centers. Call fit() first.")
        embedding = self.facenet.predict(face).astype(np.float32)
        distances = np.linalg.norm(self.cluster_center - embedding[None, :], axis=1)
        best_cluster = int(np.argmin(distances))
        return best_cluster, distances

    @staticmethod
    def _squared_distances(x, centers):
        diff = x[:, None, :] - centers[None, :, :]
        return np.sum(diff * diff, axis=2)
