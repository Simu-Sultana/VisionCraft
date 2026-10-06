import pickle

import numpy as np

from cvproj_exc.classifier import NearestNeighborClassifier

UNKNOWN_LABEL = -1


class OpenSetEvaluation:

    def __init__(
        self,
        classifier=NearestNeighborClassifier(),
        false_alarm_rate_range=np.logspace(-3, 0, 1000, endpoint=True),
    ):
        self.false_alarm_rate_range = false_alarm_rate_range
        self.train_embeddings = []
        self.train_labels = []
        self.test_embeddings = []
        self.test_labels = []
        self.classifier = classifier

    def prepare_input_data(self, train_data_file, test_data_file):
        with open(train_data_file, "rb") as f:
            (self.train_embeddings, self.train_labels) = pickle.load(f, encoding="bytes")
        with open(test_data_file, "rb") as f:
            (self.test_embeddings, self.test_labels) = pickle.load(f, encoding="bytes")

        self.train_embeddings = np.asarray(self.train_embeddings)
        self.train_labels = np.asarray(self.train_labels)
        self.test_embeddings = np.asarray(self.test_embeddings)
        self.test_labels = np.asarray(self.test_labels)

    def run(self):
        train_is_known = self.train_labels != UNKNOWN_LABEL
        self.classifier.fit(self.train_embeddings[train_is_known], self.train_labels[train_is_known])

        prediction_labels, similarities = self.classifier.predict_labels_and_similarities(
            self.test_embeddings
        )
        prediction_labels = prediction_labels.astype(self.test_labels.dtype, copy=False)
        similarities = np.asarray(similarities, dtype=float)

        similarity_thresholds = np.empty(len(self.false_alarm_rate_range), dtype=float)
        identification_rates = np.empty(len(self.false_alarm_rate_range), dtype=float)

        for i, false_alarm_rate in enumerate(self.false_alarm_rate_range):
            threshold = self.select_similarity_threshold(similarities, false_alarm_rate)
            thresholded_predictions = prediction_labels.copy()
            thresholded_predictions[similarities < threshold] = UNKNOWN_LABEL
            similarity_thresholds[i] = threshold
            identification_rates[i] = self.calc_identification_rate(thresholded_predictions)

        return {
            "similarity_thresholds": similarity_thresholds,
            "identification_rates": identification_rates,
        }

    def select_similarity_threshold(self, similarity, false_alarm_rate):
        """Choose threshold so approximately FAR fraction of unknown probes are accepted."""
        similarity = np.asarray(similarity, dtype=float)
        unknown_mask = self.test_labels == UNKNOWN_LABEL
        unknown_similarities = similarity[unknown_mask]
        if unknown_similarities.size == 0:
            unknown_similarities = similarity
        if unknown_similarities.size == 0:
            return np.inf

        false_alarm_rate = float(np.clip(false_alarm_rate, 0.0, 1.0))
        if false_alarm_rate <= 0.0:
            return np.nextafter(np.max(unknown_similarities), np.inf)
        if false_alarm_rate >= 1.0:
            return np.min(unknown_similarities)

        percentile = 100.0 * (1.0 - false_alarm_rate)
        return float(np.percentile(unknown_similarities, percentile))

    def calc_identification_rate(self, prediction_labels):
        """Rank-1 DIR on known test subjects after thresholding."""
        prediction_labels = np.asarray(prediction_labels)
        known_mask = self.test_labels != UNKNOWN_LABEL
        num_known = int(np.sum(known_mask))
        if num_known == 0:
            return 0.0
        correct_known = prediction_labels[known_mask] == self.test_labels[known_mask]
        return float(np.sum(correct_known) / num_known)
