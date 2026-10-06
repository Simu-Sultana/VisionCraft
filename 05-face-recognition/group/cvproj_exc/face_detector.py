from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from mtcnn import MTCNN


@dataclass
class FaceDetectionResult:
    image: np.ndarray
    """The image."""
    rect: tuple[int, int, int, int]
    """The face bounding box (top left x, top left y, width, height)."""
    aligned: np.ndarray
    """The aligned face image."""


class FaceDetector:
    """Face detection, template-matching based tracking, and simple crop alignment."""

    def __init__(
        self,
        tm_window_size: int = 100,
        tm_threshold: float = 0.0,
        aligned_image_size: int = 224,
        redetect_interval: int = 5,
    ) -> None:
        self.detector = MTCNN()
        self.reference: Optional[FaceDetectionResult] = None
        self.aligned_image_size = aligned_image_size

        # Template matching parameters.  TM_CCOEFF_NORMED returns high values for good matches.
        self.tm_method = cv2.TM_CCOEFF_NORMED
        self.tm_window_size = int(tm_window_size)
        self.tm_threshold = float(tm_threshold) if tm_threshold > 0.0 else 0.55

        # Pure template-matching trackers re-anchor their template to the frame they just
        # produced. Any small per-frame offset therefore carries into the next frame's template,
        # and the correlation score stays high even as the tracked box drifts away from the true
        # face (it is only ever compared to *itself*, not to a ground-truth face). Left unchecked,
        # this accumulates until the crop drifts onto background/hair and the recognizer starts
        # extracting embeddings for the wrong content. Forcing a fresh MTCNN detection every
        # `redetect_interval` frames re-anchors tracking to a verified face and bounds the drift.
        self.redetect_interval = max(1, int(redetect_interval))
        self._frames_since_detect = 0

    def track_face(self, image: np.ndarray) -> Optional[FaceDetectionResult]:
        """Track the face from the previous/reference frame using template matching.

        If no reference exists, if template matching is unreliable, or if enough frames have
        passed since the last verified detection, the method (re-)runs MTCNN face detection and
        re-initializes the reference. The periodic re-detection prevents template-matching drift
        from silently accumulating over long sequences.
        """
        if image is None or image.size == 0:
            self.reference = None
            self._frames_since_detect = 0
            return None

        if self.reference is None or self._frames_since_detect >= self.redetect_interval:
            self.reference = self.detect_face(image)
            return self.reference

        x, y, w, h = [int(v) for v in self.reference.rect]
        template = self.crop_face(self.reference.image, (x, y, w, h))
        if template.size == 0 or template.shape[0] < 2 or template.shape[1] < 2:
            self.reference = self.detect_face(image)
            return self.reference

        half_window = max(self.tm_window_size, 1)
        sx1 = max(x - half_window, 0)
        sy1 = max(y - half_window, 0)
        sx2 = min(x + w + half_window, image.shape[1])
        sy2 = min(y + h + half_window, image.shape[0])
        search_region = image[sy1:sy2, sx1:sx2]

        if search_region.shape[0] < template.shape[0] or search_region.shape[1] < template.shape[1]:
            self.reference = self.detect_face(image)
            return self.reference

        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        search_gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        response = cv2.matchTemplate(search_gray, template_gray, self.tm_method)
        _, max_val, _, max_loc = cv2.minMaxLoc(response)

        if max_val < self.tm_threshold:
            self.reference = self.detect_face(image)
            return self.reference

        new_x = sx1 + max_loc[0]
        new_y = sy1 + max_loc[1]
        rect = self._clip_rect((new_x, new_y, w, h), image.shape)
        aligned = self.align_face(image, rect)
        result = FaceDetectionResult(image=image, rect=rect, aligned=aligned)

        # Use the current frame as the next reference so slow motion is followed smoothly.
        self.reference = result
        self._frames_since_detect += 1
        return result

    def detect_face(self, image: np.ndarray) -> Optional[FaceDetectionResult]:
        if not (
            detections := self.detector.detect_faces(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                threshold_pnet=0.85,
                threshold_rnet=0.9,
            )
        ):
            self.reference = None
            self._frames_since_detect = 0
            return None

        largest_detection = int(np.argmax([d["box"][2] * d["box"][3] for d in detections]))
        face_rect = self._clip_rect(detections[largest_detection]["box"], image.shape)
        aligned = self.align_face(image, face_rect)
        self._frames_since_detect = 0
        return FaceDetectionResult(rect=face_rect, image=image, aligned=aligned)

    def align_face(self, image, face_rect):
        crop = self.crop_face(image, face_rect)
        if crop.size == 0:
            return np.zeros((self.aligned_image_size, self.aligned_image_size, 3), dtype=image.dtype)
        return cv2.resize(crop, dsize=(self.aligned_image_size, self.aligned_image_size))

    def crop_face(self, image, face_rect):
        x, y, w, h = [int(v) for v in face_rect]
        top = max(y, 0)
        left = max(x, 0)
        bottom = min(y + h, image.shape[0])
        right = min(x + w, image.shape[1])
        return image[top:bottom, left:right, :]

    @staticmethod
    def _clip_rect(face_rect, image_shape) -> tuple[int, int, int, int]:
        x, y, w, h = [int(round(v)) for v in face_rect]
        x = max(0, min(x, image_shape[1] - 1))
        y = max(0, min(y, image_shape[0] - 1))
        w = max(1, min(w, image_shape[1] - x))
        h = max(1, min(h, image_shape[0] - y))
        return x, y, w, h
