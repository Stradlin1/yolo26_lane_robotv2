# Ultralytics 🚀 AGPL-3.0 License

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.lane.dataset import LaneRobotDataset
from ultralytics.models.yolo.lane.plotting import decode_lane, save_lane_grid
from ultralytics.utils import LOGGER


def get_lane_head(model):
    """Return the LaneRobot/LaneRobotV2/LaneRobotV2Independent head from a wrapped, fused, EMA, or raw model object."""
    from ultralytics.nn.modules.head import LaneRobot, LaneRobotV2, LaneRobotV2Independent

    seen = set()

    def walk(obj):
        if obj is None:
            return None
        oid = id(obj)
        if oid in seen:
            return None
        seen.add(oid)
        if isinstance(obj, (LaneRobot, LaneRobotV2, LaneRobotV2Independent)):
            return obj
        seq = None
        if isinstance(obj, (list, tuple, torch.nn.ModuleList, torch.nn.Sequential)):
            seq = obj
        elif hasattr(obj, "model") and getattr(obj, "model") is not obj:
            found = walk(getattr(obj, "model"))
            if found is not None:
                return found
        if seq is not None:
            for item in reversed(seq):
                found = walk(item)
                if found is not None:
                    return found
        if isinstance(obj, torch.nn.Module):
            for m in obj.modules():
                if isinstance(m, (LaneRobot, LaneRobotV2, LaneRobotV2Independent)):
                    return m
        return None

    head = walk(model)
    if head is None:
        raise AttributeError("Could not locate LaneRobot/LaneRobotV2/LaneRobotV2Independent head on validator model.")
    return head


class LaneRobotMetrics:
    """Minimal metrics container expected by the Ultralytics trainer."""

    def __init__(self):
        self.keys = [
            "metrics/lane_mae",
            "metrics/lane_mae_px",
            "metrics/lane_acc_valid_tol1",
            "metrics/lane_acc_valid_tol3",
            "metrics/lane_acc_valid_tol5",
            "metrics/lane_exist_acc",
        ]
        self.lane_mae = 0.0
        self.lane_mae_px = 0.0
        self.lane_acc_valid_tol1 = 0.0
        self.lane_acc_valid_tol3 = 0.0
        self.lane_acc_valid_tol5 = 0.0
        self.lane_exist_acc = 0.0
        self.fitness = 0.0
        self.speed = None
        self.save_dir = None

    @property
    def results_dict(self):
        return {
            "metrics/lane_mae": self.lane_mae,
            "metrics/lane_mae_px": self.lane_mae_px,
            "metrics/lane_acc_valid_tol1": self.lane_acc_valid_tol1,
            "metrics/lane_acc_valid_tol3": self.lane_acc_valid_tol3,
            "metrics/lane_acc_valid_tol5": self.lane_acc_valid_tol5,
            "metrics/lane_exist_acc": self.lane_exist_acc,
            "fitness": self.fitness,
        }

    def mean_results(self):
        return [self.lane_mae, self.lane_mae_px, self.lane_acc_valid_tol1, self.lane_acc_valid_tol3, self.lane_acc_valid_tol5, self.lane_exist_acc]


class LaneRobotValidator(BaseValidator):
    """Validator for row-anchor lane classification."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)
        self.args.task = "lane"
        self.metrics = LaneRobotMetrics()

    def build_dataset(self, img_path):
        return LaneRobotDataset(img_path, self.args, self.data, mode=self.args.split or "val")

    def get_dataloader(self, dataset_path, batch_size):
        dataset = self.build_dataset(dataset_path)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            collate_fn=LaneRobotDataset.collate_fn,
            drop_last=False,
        )

    def preprocess(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=self.device.type == "cuda").float() / 255.0
        batch["lane"] = batch["lane"].to(self.device, non_blocking=self.device.type == "cuda").long()
        if "lane_x" in batch:
            batch["lane_x"] = batch["lane_x"].to(self.device, non_blocking=self.device.type == "cuda").float()
        if "lane_y" in batch:
            batch["lane_y"] = batch["lane_y"].to(self.device, non_blocking=self.device.type == "cuda").float()
        batch["cls"] = batch["cls"].to(self.device, non_blocking=self.device.type == "cuda")
        return batch

    def init_metrics(self, model):
        head = get_lane_head(model)
        self.x_grids = int(head.x_grids)
        self.row_anchors = int(head.row_anchors)
        self.num_lanes = int(head.num_lanes)
        self.no_lane_idx = self.x_grids
        self.mae_sum = 0.0
        self.mae_px_sum = 0.0
        self.valid_total = 0
        self.tol1 = 0
        self.tol3 = 0
        self.tol5 = 0
        self.exist_correct = 0
        self.exist_total = 0

    def _split_preds(self, preds):
        if isinstance(preds, dict):
            return preds["cls"], preds.get("offset", None)
        if isinstance(preds, (list, tuple)) and preds and isinstance(preds[0], dict):
            return preds[0]["cls"], preds[0].get("offset", None)
        return preds, None

    def update_metrics(self, preds, batch):
        logits, offset = self._split_preds(preds)
        pred_xy = decode_lane(
            preds,
            no_lane_idx=self.no_lane_idx,
            topk=int(getattr(self.args, "lane_softargmax_topk", 5)),
            exist_thr=float(getattr(self.args, "lane_exist_thr", 0.5)),
            post_smooth=False,
        )
        pred_x = torch.as_tensor(pred_xy, device=logits.device, dtype=torch.float32)
        target_x = batch.get("lane_x", None)
        if target_x is None:
            target_x = batch["lane"].float()
            target_x = torch.where(batch["lane"] == self.no_lane_idx, torch.full_like(target_x, -1.0), target_x)
        valid = target_x >= 0
        pred_valid = pred_x >= 0

        if valid.any():
            err = (pred_x[valid] - target_x[valid]).abs()
            self.mae_sum += float(err.sum().item())
            img_w = float(batch["img"].shape[-1])
            self.mae_px_sum += float((err * (img_w / max(self.x_grids, 1))).sum().item())
            self.valid_total += int(valid.sum().item())
            self.tol1 += int((err <= 1).sum().item())
            self.tol3 += int((err <= 3).sum().item())
            self.tol5 += int((err <= 5).sum().item())

        self.exist_correct += int((pred_valid == valid).sum().item())
        self.exist_total += int(valid.numel())

    def get_stats(self):
        valid_total = max(self.valid_total, 1)
        mae = self.mae_sum / valid_total
        mae_px = self.mae_px_sum / valid_total
        tol1 = self.tol1 / valid_total
        tol3 = self.tol3 / valid_total
        tol5 = self.tol5 / valid_total
        exist_acc = self.exist_correct / max(self.exist_total, 1)
        self.metrics.lane_mae = float(mae)
        self.metrics.lane_mae_px = float(mae_px)
        self.metrics.lane_acc_valid_tol1 = float(tol1)
        self.metrics.lane_acc_valid_tol3 = float(tol3)
        self.metrics.lane_acc_valid_tol5 = float(tol5)
        self.metrics.lane_exist_acc = float(exist_acc)
        self.metrics.fitness = float(tol3 + 0.5 * tol5 - 0.003 * mae + 0.05 * exist_acc)
        return self.metrics.results_dict

    def finalize_metrics(self):
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir

    def print_results(self):
        stats = self.get_stats()
        LOGGER.info(
            f"Lane MAE={stats['metrics/lane_mae']:.3f}, "
            f"MAE_px={stats['metrics/lane_mae_px']:.2f}, "
            f"Acc@1={stats['metrics/lane_acc_valid_tol1']:.4f}, "
            f"Acc@3={stats['metrics/lane_acc_valid_tol3']:.4f}, "
            f"Acc@5={stats['metrics/lane_acc_valid_tol5']:.4f}, "
            f"Exist={stats['metrics/lane_exist_acc']:.4f}"
        )

    def get_desc(self):
        return ("%22s" + "%11s" * 6) % ("lane", "mae", "mae_px", "acc@1", "acc@3", "acc@5", "exist")

    @property
    def metric_keys(self):
        return self.metrics.keys

    def plot_val_samples(self, batch, ni):
        tgt = batch.get("lane_x", batch["lane"].float()).detach().cpu().numpy().copy()
        tgt[tgt >= self.x_grids] = -1
        save_lane_grid(
            batch["img"].detach().cpu(),
            tgt,
            tgt,
            x_grids=self.x_grids,
            row_anchors=self.row_anchors,
            save_path=self.save_dir / f"val_batch{ni}_labels.jpg",
            row_y=batch.get("lane_y"),
            y_start=float(getattr(self.args, "lane_y_start", 0.67)),
            y_end=float(getattr(self.args, "lane_y_end", 1.0)),
        )

    def plot_predictions(self, batch, preds, ni):
        pred_xy = decode_lane(
            preds,
            no_lane_idx=self.no_lane_idx,
            topk=int(getattr(self.args, "lane_softargmax_topk", 5)),
            exist_thr=float(getattr(self.args, "lane_exist_thr", 0.5)),
            post_smooth=bool(getattr(self.args, "lane_post_smooth", True)),
            poly_degree=int(getattr(self.args, "lane_poly_degree", 2)),
            poly_blend=float(getattr(self.args, "lane_poly_blend", 0.5)),
        )
        tgt = batch.get("lane_x", batch["lane"].float()).detach().cpu().numpy().copy()
        tgt[tgt >= self.x_grids] = -1
        save_lane_grid(
            batch["img"].detach().cpu(),
            pred_xy,
            tgt,
            x_grids=self.x_grids,
            row_anchors=self.row_anchors,
            save_path=self.save_dir / f"val_batch{ni}_pred.jpg",
            row_y=batch.get("lane_y"),
            y_start=float(getattr(self.args, "lane_y_start", 0.67)),
            y_end=float(getattr(self.args, "lane_y_end", 1.0)),
        )
