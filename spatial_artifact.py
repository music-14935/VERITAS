import cv2   
import numpy as np


class SpatialArtifactAnalyzer:
    """
    Simple spatial artifact analyzer.
    Produces a numeric score usable by an AI classifier.
    """

    def analyze(self, image: np.ndarray) -> float:
        """
        Analyze spatial artifacts in an image.

        Returns:
            float: artifact score higher = more suspicious
        """

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Normalize to 0..1
        image = image.astype(np.float32) / 255.0

        # Edge response (detects unnatural edges)
        laplacian = cv2.Laplacian(image, cv2.CV_32F)
        edge_variance = laplacian.var()

        # Smoothness (detects over-smoothed regions)
        blur = cv2.GaussianBlur(image, (7, 7), 0)
        smoothness = np.mean(np.abs(image - blur))

        # Final spatial artifact score
        artifact_score = edge_variance / (smoothness + 1e-6)

        return float(artifact_score)
