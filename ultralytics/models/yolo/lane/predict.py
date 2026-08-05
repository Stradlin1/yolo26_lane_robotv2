# Ultralytics 🚀 AGPL-3.0 License

from __future__ import annotations

from ultralytics.models.yolo.detect.predict import DetectionPredictor


class LaneRobotPredictor(DetectionPredictor):
    """Lightweight predictor wrapper. Raw model output is [B, x_grids+1, row_anchors, num_lanes]."""

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        return preds
